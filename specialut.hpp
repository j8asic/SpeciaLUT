// SpeciaLUT: run-time choosing of compile-time functions
// Copyright (c) 2022, Josip Basic <j8asic@gmail.com>
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include <array>
#include <concepts>
#if __has_include(<cuda_runtime.h>)
#    include <cuda_runtime.h>
#endif

namespace SpeciaLUT {

/// Define argument type for template states and LUT index calculation
using arg_t = std::size_t;

namespace detail {

    /// Calculate flattened index in the array from LUT states count and current states
    template<arg_t NP>
    auto flatten(std::array<arg_t, NP> const& n_states, std::array<arg_t, NP> const& state) -> arg_t
    {
        arg_t offset = 0;
        for (int i = NP - 1; i >= 0; --i) {
            offset = state[i] + offset * n_states[i];
        }
        return offset;
    }

    /// Get state index from the flattened index and LUT states count
    template<arg_t NP>
    constexpr auto unflatten(arg_t i, arg_t target_param, std::array<arg_t, NP> const n_states, arg_t param = 0,
                             arg_t product = 1) -> arg_t
    {
        if (param == target_param) {
            return ((i / product) % n_states[param]);
        }
        return unflatten<NP>(i, target_param, n_states, param + 1, product * n_states[param]);
    }

    template<typename T>
    concept not_integral = !std::is_integral_v<T>;

    template<typename T>
    concept convertible_to_arg_t = std::convertible_to<T, arg_t>;

} // namespace detail

/// Runtime choosing of specialized template functions
template<auto Wrapper, arg_t... NS>
class Chooser
{
protected:
    static constexpr arg_t NP = sizeof...(NS); // number of compile-time parameters
    static constexpr arg_t NL = (NS * ...); // total number of function pointers

    using FnPtr = decltype(Wrapper.template operator()<(NS * 0)...>());
    using FnLUT = std::array<FnPtr, NL>;

    /// Get the function pointer from flattened index
    template<arg_t i, arg_t... I>
    static constexpr auto fn_ptr(const std::integer_sequence<arg_t, I...> /*unused*/) -> FnPtr
    {
        return Wrapper.template operator()<detail::unflatten<NP>(i, I, { NS... })...>();
    }

    /// Generate the look-up table from all possible combinations
    template<arg_t... I>
    static constexpr auto make_lut(std::integer_sequence<arg_t, I...> /*unused*/) -> FnLUT
    {
        return { (fn_ptr<I>(std::make_integer_sequence<arg_t, NP>{}))... };
    }

    /// Compile-time-generated table of function pointers, for all states combinations
    static constexpr FnLUT table = make_lut(std::make_integer_sequence<arg_t, NL>{});

public:
    Chooser() = default;
    ~Chooser() = default;

    /// Get the specialization function pointer, from the given runtime parameters
    constexpr auto operator()(detail::convertible_to_arg_t auto... indices) const
    {
        static_assert(sizeof...(indices) == NP, "Template called with inappropriate number of arguments.");
        return table.at(detail::flatten<NP>({ NS... }, { arg_t(indices)... }));
    }

    /// Get the specialization from the given runtime parameters, for an object instance
    constexpr auto operator()(detail::not_integral auto& obj, detail::convertible_to_arg_t auto... indices) const
    {
        static_assert(sizeof...(indices) == NP, "Template called with inappropriate number of arguments.");

        return [&obj, indices... ](auto&&... args) -> auto
        {
            return (obj.*table.at(detail::flatten<NP>({ NS... }, { arg_t(indices)... })))(
                std::forward<decltype(args)>(args)...);
        };
    }

    /// Get the specialization from the given runtime parameters, for an object instance
    constexpr auto operator()(auto* ptr, detail::convertible_to_arg_t auto... indices) const
    {
        static_assert(sizeof...(indices) == NP, "Template called with inappropriate number of arguments.");

        return [ ptr, indices... ](auto&&... args) -> auto
        {
            return (ptr->*table.at(detail::flatten<NP>({ NS... }, { arg_t(indices)... })))(
                std::forward<decltype(args)>(args)...);
        };
    }
};

// Macros for generating tables of function pointers
#define TABULATE(FnName)                                                                                               \
    []<SpeciaLUT::arg_t... args>() constexpr->auto { return &FnName<args...>; }
#define TABULATE_FUNCTOR(FnName) TABULATE(FnName::template operator())
#define TABULATE_LAMBDA(FnName) TABULATE_FUNCTOR(decltype(FnName))

/// Lambdas are instanced objects so use it to directly choose templated function
template<arg_t... NS>
auto choose_lambda(auto lam, detail::convertible_to_arg_t auto... indices)
{
    static constexpr Chooser<TABULATE_LAMBDA(lam), NS...> chooser;
    return chooser.operator()(lam, indices...);
}

// check if we need to declare CUDA stuff
#if __has_include(<cuda_runtime.h>)

/// Kernel execution parameters
struct CudaKernelExecution
{
    dim3 grid_dim{};
    dim3 block_dim{};
    size_t shmem_bytes = 0;
    cudaStream_t stream = nullptr;
};

/// Simple wrapper around chosen specialized CUDA kernel
template<auto PtrGetter, arg_t... NS>
class CudaKernel
{
    using FnPtr = decltype(PtrGetter.template operator()<(NS * 0)...>());
    FnPtr const fn_ = nullptr;
    CudaKernelExecution exec_{};

public:
    CudaKernel(FnPtr const fn, CudaKernelExecution const& exec)
        : fn_(fn)
        , exec_(exec)
    { }
    ~CudaKernel() = default;

    /// Override inherited execution parameters
    auto prepare(CudaKernelExecution const& exec) -> CudaKernel&
    {
        exec_ = exec;
        return *this;
    }

    /// Override inherited execution parameters
    auto prepare(dim3 grid_dim, dim3 block_dim, size_t shmem_bytes = 0, cudaStream_t stream = nullptr) -> CudaKernel&
    {
        exec_ = { grid_dim, block_dim, shmem_bytes, stream };
        return *this;
    }

    /// Launch the kernel with specified execution parameters and run-time arguments
    template<typename... Args>
    auto launch(Args&&... args) const -> cudaError_t
    {
        if (!fn_) {
            return ::cudaErrorUnknown;
        }
        // convert parameters pack to array to pass all data to the kernel
        auto args_ptrs = std::array<void*, sizeof...(args)>({ &args... });
        // enqueue CUDA kernel with the specific function pointer, execution parameters and forwarded run-time arguments
        return cudaLaunchKernel(reinterpret_cast<const void*>(fn_), exec_.grid_dim, exec_.block_dim, args_ptrs.data(),
                                exec_.shmem_bytes, exec_.stream);
    }

    /// See: launch
    template<typename... Args>
    auto operator()(Args&&... args) const -> cudaError_t
    {
        return launch(std::forward<Args>(args)...);
    }
};

/// Runtime choosing of specialized template CUDA kernels
template<auto PtrGetter, arg_t... NS>
class CudaChooser : public Chooser<PtrGetter, NS...>
{
    CudaKernelExecution exec_{};

public:
    CudaChooser() = default;
    ~CudaChooser() = default;

    /// Prepare execution parameters
    auto prepare(CudaKernelExecution const& exec) -> CudaChooser&
    {
        exec_ = exec;
        return *this;
    }

    /// Prepare execution parameters
    auto prepare(dim3 grid_dim, dim3 block_dim, size_t shmem_bytes = 0, cudaStream_t stream = nullptr) -> CudaChooser&
    {
        exec_ = { grid_dim, block_dim, shmem_bytes, stream };
        return *this;
    }

    /// Get the specialized kernel pointer, deduced from the given runtime parameters
    constexpr auto operator()(detail::convertible_to_arg_t auto... indices) const -> CudaKernel<PtrGetter, NS...>
    {
        return { Chooser<PtrGetter, NS...>::operator()(indices...), exec_ };
    }
};

#endif // end CUDA stuff

} // namespace SpeciaLUT
