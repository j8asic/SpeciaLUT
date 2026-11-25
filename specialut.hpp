// SpeciaLUT: run-time choosing of compile-time functions
// Copyright (c) 2022, Josip Basic <j8asic@gmail.com>
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

// C++23: the call operator can be static, so choosing needs no object at all
#if __cplusplus >= 202302L
#if defined(__cpp_static_call_operator) && __cpp_static_call_operator >= 202207L
#define SPECIALUT_HAS_STATIC_CALL_OP 1
#endif
#endif

#ifdef SPECIALUT_HAS_STATIC_CALL_OP
#define SPECIALUT_CALL_STATIC static
#define SPECIALUT_CALL_CONST
#else
#define SPECIALUT_CALL_STATIC
#define SPECIALUT_CALL_CONST const
#endif

// GPU runtime support (CUDA or HIP). It is enabled when the translation unit is
// compiled as CUDA or HIP code, which is the only place a kernel can be
// tabulated. Define SPECIALUT_USE_CUDA or SPECIALUT_USE_HIP to force one of
// them, or SPECIALUT_NO_GPU to leave the GPU part out.
#if !defined(SPECIALUT_NO_GPU)
#if defined(SPECIALUT_USE_HIP) || defined(__HIP__)
#define SPECIALUT_GPU_HIP 1
#elif defined(SPECIALUT_USE_CUDA) || defined(__CUDACC__) || defined(__CUDA__)
#define SPECIALUT_GPU_CUDA 1
#elif defined(__HIPCC__)
#define SPECIALUT_GPU_HIP 1
#endif
#endif

#if defined(SPECIALUT_GPU_CUDA)
#include <cuda_runtime.h>
#elif defined(SPECIALUT_GPU_HIP)
#include <hip/hip_runtime.h>
// CUDA-named aliases, kept for code written against the CUDA spelling. Define
// SPECIALUT_NO_CUDA_ALIASES if your HIP port already provides its own.
#if !defined(SPECIALUT_NO_CUDA_ALIASES)
using cudaStream_t = hipStream_t;
using cudaError_t = hipError_t;
inline constexpr auto cudaSuccess = hipSuccess;
inline constexpr auto cudaErrorInvalidDeviceFunction =
    hipErrorInvalidDeviceFunction;
inline auto cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             hipStream_t stream) {
  return hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}
#endif
#endif

namespace SpeciaLUT {

/// Type of state counts, and of the value a template parameter receives when
/// it is tabulated by its number of states
using arg_t = std::size_t;

namespace detail {

/// Type of `SpeciaLUT::values<...>`
template <auto... Vs> struct value_list {};

/// A specialization that is not in the table was asked for
[[noreturn]] inline void bad_state() {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  throw std::out_of_range(
      "SpeciaLUT: no specialization is tabulated for these run-time states");
#else
  std::abort();
#endif
}

template <typename T, arg_t N>
constexpr auto all_distinct(std::array<T, N> const &a) noexcept -> bool {
  for (arg_t i = 0; i < N; ++i) {
    for (arg_t j = i + 1; j < N; ++j) {
      if (a[i] == a[j]) {
        return false;
      }
    }
  }
  return true;
}

/// One tabulated template parameter: how many states it has (`size`), which
/// compile-time value each state stands for (`value`), and which state a
/// run-time key selects (`index`, >= `size` if the key is not tabulated).
template <auto Spec, typename = decltype(Spec)> struct axis {
  static_assert(std::is_void_v<decltype(Spec)>,
                "SpeciaLUT: describe a template parameter by its number of "
                "states, by an enumerator that counts the states, or by "
                "SpeciaLUT::values<...>");
};

/// `N`: states 0 .. N-1, handed to the template as arg_t
template <auto N, std::integral T> struct axis<N, T> {
  static_assert(!std::same_as<T, bool>,
                "SpeciaLUT: a bool parameter has 2 states, write 2");
  static_assert(N > 0, "SpeciaLUT: a parameter needs at least one state");

  static constexpr arg_t size = static_cast<arg_t>(N);

  static constexpr auto value(arg_t state) noexcept -> arg_t { return state; }

  template <std::convertible_to<arg_t> K>
  static constexpr auto index(K const &key) noexcept -> arg_t {
    return static_cast<arg_t>(key); // a negative key wraps far out of range
  }
};

/// `Enum::Count`: enumerators 0 .. Count-1, handed to the template as Enum
template <auto Count, typename T>
  requires std::is_enum_v<T>
struct axis<Count, T> {
  static_assert(static_cast<std::underlying_type_t<T>>(Count) > 0,
                "SpeciaLUT: a parameter needs at least one state");

  static constexpr arg_t size = static_cast<arg_t>(Count);

  static constexpr auto value(arg_t state) noexcept -> T {
    return static_cast<T>(state);
  }

  static constexpr auto index(T key) noexcept -> arg_t {
    return static_cast<arg_t>(key);
  }
};

/// `values<V0, V1, ...>`: exactly the listed values, in the listed order
template <auto Spec, auto... Vs> struct axis<Spec, value_list<Vs...>> {
  using key_type = std::common_type_t<decltype(Vs)...>;

  static constexpr arg_t size = sizeof...(Vs);
  static constexpr std::array<key_type, size> keys{Vs...};

  static_assert(all_distinct(keys),
                "SpeciaLUT: values<...> lists the same value twice");

  static constexpr auto value(arg_t state) noexcept -> key_type {
    return keys[state];
  }

  template <std::convertible_to<key_type> K>
  static constexpr auto index(K const &key) noexcept -> arg_t {
    const auto wanted = static_cast<key_type>(key);
    arg_t state = 0;
    while (state < size && !(keys[state] == wanted)) {
      ++state;
    }
    return state;
  }
};

/// A run-time value of type K can select a state of the parameter `Spec`
template <typename K, auto Spec>
concept key_for = requires(K const &key) {
  { axis<Spec>::index(key) } -> std::same_as<arg_t>;
};

template <typename P> struct member_class {};
template <typename M, typename C> struct member_class<M C::*> {
  using type = C;
};

/// An object of type C can be used to call the member function pointer P
template <typename C, typename P>
concept object_for =
    std::is_member_function_pointer_v<P> &&
    std::derived_from<std::remove_cv_t<C>, typename member_class<P>::type>;

/// State of parameter `dim`, from the flattened table index. Flattening
/// scheme, for extents (N0, N1, N2) and states (i0, i1, i2):
///   flat = i0 + N0 * (i1 + N1 * i2)
/// so the first template parameter varies fastest.
template <arg_t Rank>
constexpr auto unflatten(arg_t flat, arg_t dim,
                         std::array<arg_t, Rank> const &extents) noexcept
    -> arg_t {
  for (arg_t d = 0; d < dim; ++d) {
    flat /= extents[d];
  }
  return flat % extents[dim];
}

/// Does `Wrapper` make a specialization for these compile-time values? It
/// does not if the template itself rejects them, most likely by a
/// requires-clause.
template <auto Wrapper, auto... Values>
concept specialized = requires { Wrapper.template operator()<Values...>(); };

/// The specializations that `Wrapper` makes for the parameters `Axes`, which
/// are numbered by `D`. All of this is worked out here, and not inside of
/// Chooser, because the front end of nvcc expands the Axes pack wrongly if
/// that happens in a constraint of a member template that has a pack of its
/// own.
template <auto Wrapper, typename Dims, auto... Axes> struct specializations;

template <auto Wrapper, arg_t... D, auto... Axes>
struct specializations<Wrapper, std::index_sequence<D...>, Axes...> {
  static constexpr std::array<arg_t, sizeof...(Axes)> extents{
      axis<Axes>::size...};

  /// Number of combinations of the states
  static constexpr arg_t count = (arg_t{1} * ... * axis<Axes>::size);

  /// Does the specialization at this flattened index exist? The values go
  /// through a concept with a pack of its own: written out here as a
  /// requires-expression, the front end of nvcc takes it for false.
  template <arg_t Flat>
  static constexpr bool exists =
      specialized<Wrapper, axis<Axes>::value(unflatten(Flat, D, extents))...>;

  /// Pointer to the specialization at this flattened index, nullptr if it
  /// does not exist
  template <arg_t Flat> static constexpr auto get() noexcept {
    if constexpr (exists<Flat>) {
      return Wrapper.template
      operator()<axis<Axes>::value(unflatten(Flat, D, extents))...>();
    } else {
      return nullptr;
    }
  }

  template <arg_t... Flat>
  static constexpr auto
  number_of(std::index_sequence<Flat...> /*unused*/) noexcept -> arg_t {
    return (arg_t{0} + ... + (exists<Flat> ? arg_t{1} : arg_t{0}));
  }

  template <arg_t... Flat>
  static constexpr auto
  first_of(std::index_sequence<Flat...> /*unused*/) noexcept -> arg_t {
    arg_t found = 0;
    (void)((exists<Flat> ? (found = Flat, true) : false) || ...);
    return found;
  }

  /// How many of the specializations exist
  static constexpr arg_t existing =
      number_of(std::make_index_sequence<count>{});

  /// Flattened index of the first specialization that exists
  static constexpr arg_t first = first_of(std::make_index_sequence<count>{});
};

template <auto Wrapper, auto... Axes>
using specializations_of =
    specializations<Wrapper, std::make_index_sequence<sizeof...(Axes)>,
                    Axes...>;

/// Pointer type of the specializations, taken from the first one that exists
template <auto Wrapper, auto... Axes> struct signature {
  using type = decltype(specializations_of<Wrapper, Axes...>::template get<
                        specializations_of<Wrapper, Axes...>::first>());
};

} // namespace detail

/// Explicit list of the values a template parameter is tabulated for, e.g.
/// `values<2, 3>` for a dimension or `values<Mode::A, Mode::C>` for a subset
/// of an enum. At run-time the parameter is chosen by value, not by position.
template <auto... Vs> inline constexpr detail::value_list<Vs...> values{};

template <auto Wrapper, auto... Axes> class Chooser;
#if defined(SPECIALUT_GPU_CUDA) || defined(SPECIALUT_GPU_HIP)
template <auto PtrGetter, auto... Axes> class CudaChooser;
#endif

/// One combination of states of the template parameters `Axes`, checked when
/// it is made. It is meant for several functions specialized by the same
/// parameters: describe the parameters once, choose their states once, and
/// take the specialization of every function with `chooser[states]`.
///
///   using Options = SpeciaLUT::States<Kernel::Count, values<2, 3>, 2>;
///   Options::Chooser<TABULATE(density)> density;
///   Options::Chooser<TABULATE(forces)> forces;
///
///   const Options options(kernel, dim, viscous);
///   density[options](particles);
///   forces[options](particles);
///
/// Each of `Axes` describes one template parameter, in order:
///   - `N`            N states; the template gets 0 .. N-1 (use 2 for a bool)
///   - `Enum::Count`  the template gets Enum(0) .. Enum(Count-1), and at
///                    run-time the parameter is chosen by an `Enum`
///   - `values<...>`  the template gets exactly the listed values
template <auto... Axes> class States {
  static_assert(sizeof...(Axes) > 0,
                "SpeciaLUT: tabulate at least one template parameter");

public:
  /// Number of tabulated template parameters
  static constexpr arg_t rank = sizeof...(Axes);

  /// Number of states of each tabulated template parameter
  static constexpr std::array<arg_t, rank> extents{detail::axis<Axes>::size...};

  /// Table of the specializations of one function, for these parameters
  template <auto Wrapper> using Chooser = SpeciaLUT::Chooser<Wrapper, Axes...>;

#if defined(SPECIALUT_GPU_CUDA) || defined(SPECIALUT_GPU_HIP)
  /// Table of the specializations of one kernel, for these parameters
  template <auto PtrGetter>
  using CudaChooser = SpeciaLUT::CudaChooser<PtrGetter, Axes...>;
#endif

  /// Number of combinations of the states
  static constexpr auto size() noexcept -> arg_t { return count; }

  /// Is each of the given run-time parameters inside of its own range?
  template <typename... K>
    requires(sizeof...(K) == rank) && (detail::key_for<K, Axes> && ...)
  [[nodiscard]] static constexpr auto contains(K... keys) noexcept -> bool {
    return ((detail::axis<Axes>::index(keys) < detail::axis<Axes>::size) &&
            ...);
  }

  /// Choose the states by the given run-time parameters. Each parameter is
  /// checked against its own range: a wrong one throws std::out_of_range (or
  /// aborts, if exceptions are disabled).
  template <typename... K>
    requires(sizeof...(K) == rank) && (detail::key_for<K, Axes> && ...)
  constexpr explicit States(K... keys) : index_(flat_index(keys...)) {
    if (!contains(keys...)) {
      detail::bad_state();
    }
  }

  /// Position of this combination in a table. For extents (N0, N1, N2) and
  /// states (i0, i1, i2) it is i0 + N0 * (i1 + N1 * i2).
  [[nodiscard]] constexpr auto index() const noexcept -> arg_t {
    return index_;
  }

  friend constexpr auto operator==(States const &, States const &) noexcept
      -> bool = default;

private:
  template <auto W, auto... A> friend class SpeciaLUT::Chooser;

  static constexpr arg_t count = (arg_t{1} * ... * detail::axis<Axes>::size);

  /// Table stride of each parameter: 1, N0, N0 * N1, ...
  static constexpr std::array<arg_t, rank> strides = [] {
    std::array<arg_t, rank> s{};
    arg_t stride = 1;
    for (arg_t d = 0; d < rank; ++d) {
      s[d] = stride;
      stride *= extents[d];
    }
    return s;
  }();

  /// Flattened table index of the given run-time states
  template <typename... K>
  static constexpr auto flat_index(K const &...keys) noexcept -> arg_t {
    arg_t flat = 0;
    arg_t d = 0;
    ((flat += detail::axis<Axes>::index(keys) * strides[d++]), ...);
    return flat;
  }

  arg_t index_;
};

/// Runtime choosing of specialized template functions.
///
/// `Wrapper` comes from one of the TABULATE macros. Each of `Axes` describes
/// one template parameter of the function, in order; see States.
///
/// A combination of states that the template itself rejects, by a
/// requires-clause, is not compiled and is left out of the table: `find`
/// returns nullptr for it, and choosing it in any other way is an error.
template <auto Wrapper, auto... Axes> class Chooser {
  using all = detail::specializations_of<Wrapper, Axes...>;

  static_assert(all::existing > 0,
                "SpeciaLUT: the function has no specialization for any "
                "combination of the tabulated states; check the number and "
                "the types of its template parameters");

public:
  /// The states of the tabulated parameters. One `state_type(keys...)` chooses
  /// from every table that has the same parameters.
  using state_type = States<Axes...>;

  /// Number of tabulated template parameters
  static constexpr arg_t rank = state_type::rank;

  /// Number of states of each tabulated template parameter
  static constexpr std::array<arg_t, rank> extents = state_type::extents;

  /// Pointer type shared by all the specializations; a member function
  /// pointer if a member function, functor or lambda was tabulated
  using pointer = typename detail::signature<Wrapper, Axes...>::type;

protected:
  /// Number of combinations of the states, which is the length of the table
  static constexpr arg_t count = state_type::size();

  using FnPtr = pointer;
  using FnLUT = std::array<FnPtr, count>;

  /// Get the function pointer from flattened index
  template <arg_t Flat> static constexpr auto entry() noexcept -> FnPtr {
    static_assert(
        std::is_convertible_v<decltype(all::template get<Flat>()), FnPtr>,
        "SpeciaLUT: all specializations must have the same signature");
    return all::template get<Flat>();
  }

  /// Generate the look-up table from all possible combinations
  template <arg_t... Flat>
  static constexpr auto
  make_lut(std::index_sequence<Flat...> /*unused*/) noexcept -> FnLUT {
    return {entry<Flat>()...};
  }

  /// Compile-time-generated table of function pointers, for all states
  /// combinations
  static constexpr FnLUT table = make_lut(std::make_index_sequence<count>{});

  /// A table entry that was chosen for use; an error if the template rejected
  /// that combination. Tables with no such combination pay nothing for it.
  static constexpr auto chosen(FnPtr fn) -> FnPtr {
    if (all::existing != count && !fn) {
      detail::bad_state();
    }
    return fn;
  }

  /// The call goes through std::launder only to keep GCC quiet: for an object
  /// smaller than a pointer it otherwise warns about reading the vtable
  /// pointer the object does not have (-Warray-bounds), on the path of a
  /// virtual member function that is never taken. The code generated is the
  /// same with and without it.
  template <typename C> static constexpr auto bind(C &object, FnPtr fn) {
    return [&object, fn](auto &&...args) -> decltype(auto) {
      return (std::launder(&object)->*fn)(
          std::forward<decltype(args)>(args)...);
    };
  }

public:
  /// Number of specializations in the table. It is less than the number of
  /// combinations of the states if the template rejects some of them.
  static constexpr auto size() noexcept -> arg_t { return all::existing; }

  /// Get the specialization function pointer from the given runtime
  /// parameters, or nullptr if there is none: a parameter is outside of its
  /// range, or the template rejects the combination. Use it to fall back to a
  /// generic implementation for the states that were not tabulated.
  template <typename... K>
    requires(sizeof...(K) == rank) && (detail::key_for<K, Axes> && ...)
  [[nodiscard]] static constexpr auto find(K... keys) noexcept -> FnPtr {
    return state_type::contains(keys...)
               ? table[state_type::flat_index(keys...)]
               : FnPtr{};
  }

  /// See: find
  [[nodiscard]] static constexpr auto find(state_type states) noexcept
      -> FnPtr {
    return table[states.index()];
  }

  /// Get the specialization function pointer from the given runtime
  /// parameters. Each parameter is checked against its own range: a wrong one
  /// throws std::out_of_range (or aborts, if exceptions are disabled), and so
  /// does a combination that the template rejects.
  template <typename... K>
    requires(sizeof...(K) == rank) && (detail::key_for<K, Axes> && ...)
  [[nodiscard]] static constexpr auto at(K... keys) -> FnPtr {
    if (!state_type::contains(keys...)) {
      detail::bad_state();
    }
    return chosen(table[state_type::flat_index(keys...)]);
  }

  /// See: at
  template <typename... K>
    requires(sizeof...(K) == rank) && (detail::key_for<K, Axes> && ...)
  [[nodiscard]] SPECIALUT_CALL_STATIC constexpr FnPtr
  operator()(K... keys) SPECIALUT_CALL_CONST {
    return at(keys...);
  }

  /// Get the specialization function pointer for states that were chosen and
  /// checked before
  [[nodiscard]] constexpr auto operator[](state_type states) const -> FnPtr {
    return chosen(table[states.index()]);
  }

  /// Get the specialization from the given runtime parameters, bound to an
  /// object instance. The object is held by reference.
  template <typename C, typename... K>
    requires(sizeof...(K) == rank) && detail::object_for<C, FnPtr> &&
            (detail::key_for<K, Axes> && ...)
  [[nodiscard]] SPECIALUT_CALL_STATIC constexpr auto
  operator()(C &object, K... keys) SPECIALUT_CALL_CONST {
    return bind(object, at(keys...));
  }

  /// Get the specialization from the given runtime parameters, bound to an
  /// object instance
  template <typename C, typename... K>
    requires(sizeof...(K) == rank) && detail::object_for<C, FnPtr> &&
            (detail::key_for<K, Axes> && ...)
  [[nodiscard]] SPECIALUT_CALL_STATIC constexpr auto
  operator()(C *object, K... keys) SPECIALUT_CALL_CONST {
    return bind(*object, at(keys...));
  }

  /// Get the specialization for states that were chosen and checked before,
  /// bound to an object instance. The object is held by reference.
  template <typename C>
    requires detail::object_for<C, FnPtr>
  [[nodiscard]] SPECIALUT_CALL_STATIC constexpr auto
  operator()(C &object, state_type states) SPECIALUT_CALL_CONST {
    return bind(object, chosen(table[states.index()]));
  }

  /// Get the specialization for states that were chosen and checked before,
  /// bound to an object instance
  template <typename C>
    requires detail::object_for<C, FnPtr>
  [[nodiscard]] SPECIALUT_CALL_STATIC constexpr auto
  operator()(C *object, state_type states) SPECIALUT_CALL_CONST {
    return bind(*object, chosen(table[states.index()]));
  }
};

// Macros for generating tables of function pointers. The return type is
// spelled out, and not left to be deduced, so that a combination of values
// that the template rejects is noticed without being a compile error.
#define SPECIALUT_TABULATE(FnName)                                             \
  []<auto... specialut_args>() constexpr                                       \
      -> decltype(&FnName<specialut_args...>) {                                \
    return &FnName<specialut_args...>;                                         \
  }
#define SPECIALUT_TABULATE_FUNCTOR(FnName)                                     \
  SPECIALUT_TABULATE(FnName::template operator())
#define SPECIALUT_TABULATE_LAMBDA(FnName)                                      \
  SPECIALUT_TABULATE_FUNCTOR(decltype(FnName))

// Short names; define SPECIALUT_NO_SHORT_MACROS if they clash with yours
#if !defined(SPECIALUT_NO_SHORT_MACROS)
#define TABULATE(FnName) SPECIALUT_TABULATE(FnName)
#define TABULATE_FUNCTOR(FnName) SPECIALUT_TABULATE_FUNCTOR(FnName)
#define TABULATE_LAMBDA(FnName) SPECIALUT_TABULATE_LAMBDA(FnName)
#endif

/// Lambdas are instanced objects so use it to directly choose templated
/// function, with no macro and no named function:
///   choose_lambda<2, 3>([&]<bool A, int B>() { run<A, B>(x); }, a, b)();
/// A lambda given as an lvalue is held by reference, a temporary is moved
/// into the result.
template <auto... Axes, typename Lambda, typename... K>
  requires(sizeof...(K) == sizeof...(Axes)) && (detail::key_for<K, Axes> && ...)
[[nodiscard]] constexpr auto choose_lambda(Lambda &&lambda, K... keys) {
  using Closure = std::remove_reference_t<Lambda>;
  constexpr auto wrapper = []<auto... specialut_args>() constexpr
      -> decltype(&Closure::template operator()<specialut_args...>) {
    return &Closure::template operator()<specialut_args...>;
  };
  const auto fn = Chooser<wrapper, Axes...>::at(keys...);
  if constexpr (std::is_lvalue_reference_v<Lambda>) {
    return [&lambda, fn](auto &&...args) -> decltype(auto) {
      return (std::launder(&lambda)->*fn)(
          std::forward<decltype(args)>(args)...);
    };
  } else {
    return [lambda = std::move(lambda),
            fn](auto &&...args) mutable -> decltype(auto) {
      return (std::launder(&lambda)->*fn)(
          std::forward<decltype(args)>(args)...);
    };
  }
}

// GPU kernel support (CUDA or HIP)
#if defined(SPECIALUT_GPU_CUDA) || defined(SPECIALUT_GPU_HIP)

/// Names of the GPU runtime in use, so that the code below reads the same for
/// CUDA and HIP
namespace gpu {
#if defined(SPECIALUT_GPU_CUDA)
using stream_t = cudaStream_t;
using error_t = cudaError_t;
inline constexpr error_t invalid_function = cudaErrorInvalidDeviceFunction;
inline auto launch(const void *fn, dim3 grid_dim, dim3 block_dim, void **args,
                   size_t shmem_bytes, stream_t stream) -> error_t {
  return cudaLaunchKernel(fn, grid_dim, block_dim, args, shmem_bytes, stream);
}
#else
using stream_t = hipStream_t;
using error_t = hipError_t;
inline constexpr error_t invalid_function = hipErrorInvalidDeviceFunction;
inline auto launch(const void *fn, dim3 grid_dim, dim3 block_dim, void **args,
                   size_t shmem_bytes, stream_t stream) -> error_t {
  return hipLaunchKernel(fn, grid_dim, block_dim, args, shmem_bytes, stream);
}
#endif
} // namespace gpu

/// Kernel execution parameters
struct CudaKernelExecution {
  dim3 grid_dim{};
  dim3 block_dim{};
  size_t shmem_bytes = 0;
  gpu::stream_t stream = nullptr;
};

namespace detail {

/// Enqueue the kernel. The runtime reads every argument through a bare
/// pointer, as an object of the kernel's own parameter type, so the arguments
/// are taken here by value as exactly those types: the caller's arguments are
/// converted like in an ordinary call, and a wrong count does not compile.
template <bool NoThrow, typename... Params>
auto launch_kernel(void (*kernel)(Params...) noexcept(NoThrow),
                   CudaKernelExecution const &exec,
                   std::type_identity_t<Params>... args) -> gpu::error_t {
  void *arg_ptrs[sizeof...(Params) + 1] = {static_cast<void *>(&args)...,
                                           nullptr};
  return gpu::launch(reinterpret_cast<const void *>(kernel), exec.grid_dim,
                     exec.block_dim, arg_ptrs, exec.shmem_bytes, exec.stream);
}

} // namespace detail

/// Simple wrapper around chosen specialized CUDA kernel
template <auto PtrGetter, auto... Axes> class CudaKernel {
  using FnPtr = typename Chooser<PtrGetter, Axes...>::pointer;
  FnPtr fn_ = nullptr;
  CudaKernelExecution exec_{};

public:
  CudaKernel(FnPtr const fn, CudaKernelExecution const &exec)
      : fn_(fn), exec_(exec) {}
  ~CudaKernel() = default;

  /// Override inherited execution parameters
  auto prepare(CudaKernelExecution const &exec) -> CudaKernel & {
    exec_ = exec;
    return *this;
  }

  /// Override inherited execution parameters
  auto prepare(dim3 grid_dim, dim3 block_dim, size_t shmem_bytes = 0,
               gpu::stream_t stream = nullptr) -> CudaKernel & {
    exec_ = {grid_dim, block_dim, shmem_bytes, stream};
    return *this;
  }

  /// Launch the kernel with specified execution parameters and run-time
  /// arguments, which are converted to the parameter types of the kernel
  template <typename... Args>
  auto launch(Args &&...args) const -> gpu::error_t {
    if (!fn_) {
      return gpu::invalid_function;
    }
    return detail::launch_kernel(fn_, exec_, std::forward<Args>(args)...);
  }

  /// See: launch
  template <typename... Args>
  auto operator()(Args &&...args) const -> gpu::error_t {
    return launch(std::forward<Args>(args)...);
  }
};

/// Runtime choosing of specialized template CUDA kernels
template <auto PtrGetter, auto... Axes>
class CudaChooser : public Chooser<PtrGetter, Axes...> {
  CudaKernelExecution exec_{};

public:
  CudaChooser() = default;
  ~CudaChooser() = default;

  /// Prepare execution parameters
  auto prepare(CudaKernelExecution const &exec) -> CudaChooser & {
    exec_ = exec;
    return *this;
  }

  /// Prepare execution parameters
  auto prepare(dim3 grid_dim, dim3 block_dim, size_t shmem_bytes = 0,
               gpu::stream_t stream = nullptr) -> CudaChooser & {
    exec_ = {grid_dim, block_dim, shmem_bytes, stream};
    return *this;
  }

  /// Get the specialized kernel pointer, deduced from the given runtime
  /// parameters
  template <typename... K>
    requires(sizeof...(K) == sizeof...(Axes)) &&
            (detail::key_for<K, Axes> && ...)
  [[nodiscard]] auto operator()(K... keys) const
      -> CudaKernel<PtrGetter, Axes...> {
    return {Chooser<PtrGetter, Axes...>::at(keys...), exec_};
  }

  /// Get the specialized kernel pointer for states that were chosen and
  /// checked before
  [[nodiscard]] auto operator[](States<Axes...> states) const
      -> CudaKernel<PtrGetter, Axes...> {
    return {Chooser<PtrGetter, Axes...>::operator[](states), exec_};
  }
};

#endif // end GPU kernel stuff (CUDA/HIP)

} // namespace SpeciaLUT

#undef SPECIALUT_CALL_STATIC
#undef SPECIALUT_CALL_CONST
