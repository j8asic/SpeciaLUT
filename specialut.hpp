#pragma once

#include <array>
#include <utility>
#if __has_include(<cuda_runtime.h>)
#    include <cuda_runtime.h>
#endif

namespace SpeciaLUT {

namespace detail {

    template<auto F>
    struct Signature;
    /// Extract the function signature
    template<typename R, typename... A, R (*F)(A...)>
    struct Signature<F>
    {
        using value = R(A...);
    };

    /// Calculate flattened index in the array from LUT states count and current states
    template<std::size_t NP>
    auto lut_index(std::array<std::size_t, NP> const& n_states, std::array<int, NP> const& state) -> int
    {
        int offset = 0;
        for (int i = NP - 1; i >= 0; --i) {
            offset = state[i] + offset * n_states[i];
        }
        return offset;
    }

    /// Get state index from the flattened index and LUT states count
    template<std::size_t NP>
    constexpr auto unflatten(std::size_t i, std::size_t target_param, std::array<std::size_t, NP> const n_states,
                             std::size_t param = 0, std::size_t product = 1) -> std::size_t
    {
        if (param == target_param) {
            return ((i / product) % n_states[param]);
        }
        return unflatten<NP>(i, target_param, n_states, param + 1, product * n_states[param]);
    }

}

/// Runtime choosing of specialized template functions
template<auto PtrGetter, std::size_t... NS>
class Chooser
{

protected:
    static constexpr std::size_t NP = sizeof...(NS); // number of compile-time parameters
    static constexpr std::size_t NL = (NS * ...); // total number of function pointers

    using FnSignature = typename detail::Signature<PtrGetter.template operator()<(NS * 0)...>()>::value;
    using FnLUT = std::array<FnSignature*, NL>;

    /// Get the function pointer from flattened index
    template<int i, std::size_t... I>
    static constexpr auto fn_ptr(const std::index_sequence<I...> /*unused*/) -> FnSignature*
    {
        return PtrGetter.template operator()<detail::unflatten<NP>(i, I, { NS... })...>();
    }

    /// Generate LUT from
    template<std::size_t... I>
    static constexpr auto make_lut(std::index_sequence<I...> /*unused*/) -> FnLUT
    {
        return { (fn_ptr<I>(std::make_index_sequence<NP>{}))... };
    }

    /// Compile-time-generated table of function pointers for all states combinations
    static constexpr FnLUT lut_ = make_lut(std::make_index_sequence<NL>{});

public:
    Chooser() = default;
    ~Chooser() = default;

    /// Get the specialized function, deduced from the given runtime parameters
    template<typename... Indices>
    auto operator()(Indices... indices) const -> FnSignature const&
    {
        static_assert(sizeof...(indices) == sizeof...(NS), "Template called with inappropriate number of arguments.");
        return *lut_.at(detail::lut_index<sizeof...(NS)>({ NS... }, { indices... }));
    }
};

#define TABULATE(FnName) []<int... args>() constexpr->auto { return &FnName<args...>; }
#define CHOOSER(FnName, ...) SpeciaLUT::Chooser<TABULATE(FnName), __VA_ARGS__>


#if __has_include(<cuda_runtime.h>)

/// Kernel execution parameters
struct CudaKernelExecution
{
    dim3 grid_dim_{};
    dim3 block_dim_{};
    size_t shmem_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
};

/// Simple wrapper around chosen specialized CUDA kernel
template<auto PtrGetter, std::size_t... NS>
class CudaKernel
{
    using FnSignature = typename detail::Signature<PtrGetter.template operator()<(NS * 0)...>()>::value;
    FnSignature* fn_ = nullptr;
    CudaKernelExecution exec_{};

public:
    CudaKernel(FnSignature* fn, CudaKernelExecution const& exec)
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
            return cudaErrorUnknown;
        }
        // convert parameters pack to array to pass all data to the kernel
        auto args_ptrs = std::array<void*, sizeof...(args)>({ &args... });
        // enqueue CUDA kernel with the specific function pointer, execution parameters and forwarded run-time arguments
        return cudaLaunchKernel(reinterpret_cast<const void*>(fn_), exec_.grid_dim_, exec_.block_dim_, args_ptrs.data(),
                                exec_.shmem_bytes_, exec_.stream_);
    }

    /// See: launch
    template<typename... Args>
    auto operator()(Args&&... args) const -> cudaError_t
    {
        return launch(std::forward<Args>(args)...);
    }
};

/// Runtime choosing of specialized template CUDA kernels
template<auto PtrGetter, std::size_t... NS>
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
    template<typename... Indices>
    auto operator()(Indices... indices) const -> CudaKernel<PtrGetter, NS...>
    {
        static_assert(sizeof...(indices) == sizeof...(NS), "Template called with inappropriate number of arguments.");
        return { this->lut_.at(detail::lut_index<sizeof...(NS)>({ NS... }, { indices... })), exec_ };
    }

};

#    define CUDA_CHOOSER(KernelName, ...) SpeciaLUT::CudaChooser<TABULATE(KernelName), __VA_ARGS__>

#endif // end CUDA stuff

}
