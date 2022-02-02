#pragma once

#include <array>
#include <utility>

namespace SpeciaLUT {

namespace detail {

    template<auto F>
    struct Signature;
    template<typename R, typename... A, R (*F)(A...)>
    struct Signature<F>
    {
        using value = R(A...);
    };

    template<std::size_t ND>
    auto flat_offset(std::array<std::size_t, ND> ns, std::array<int, ND> is) -> int
    {
        int off = 0;
        for (int i = ND - 1; i >= 0; --i) {
            off = is[i] + off * ns[i];
        }
        return off;
    }

    template<std::size_t ND>
    constexpr auto unflatten(std::size_t i, std::size_t target_level, std::array<std::size_t, ND> const ns,
                             std::size_t level = 0, std::size_t product = 1) -> std::size_t
    {
        if (level == target_level) {
            return ((i / product) % ns[level]);
        }
        return unflatten<ND>(i, target_level, ns, level + 1, product * ns[level]);
    }

}

/// Runtime choosing of specialized template functions
template<auto PtrGetter, std::size_t... NS>
class Chooser
{

protected:
    static constexpr std::size_t n_dims_ = sizeof...(NS);
    static constexpr std::size_t n_ptrs_ = (NS * ...);

    using FnSignature = typename detail::Signature<PtrGetter.template operator()<(NS * 0)...>()>::value;
    using FnLUT = std::array<FnSignature*, n_ptrs_>;

    template<int i, std::size_t... I>
    static constexpr auto fn_ptr(const std::index_sequence<I...> /*unused*/) -> FnSignature*
    {
        return PtrGetter.template operator()<detail::unflatten<n_dims_>(i, I, { NS... })...>();
    }

    template<std::size_t... I>
    static constexpr auto make_lut(std::index_sequence<I...> /*unused*/) -> FnLUT
    {
        return { (fn_ptr<I>(std::make_index_sequence<n_dims_>{}))... };
    }

    static constexpr FnLUT ptrs_ = make_lut(std::make_index_sequence<n_ptrs_>{});

public:
    Chooser() = default;
    ~Chooser() = default;

    /// Get the specialized function deduced from given runtime parameters
    template<typename... Indices>
    auto operator()(Indices... indices) const -> FnSignature const&
    {
        static_assert(sizeof...(indices) == sizeof...(NS), "Template called with inappropriate number of arguments.");
        return *ptrs_.at(detail::flat_offset<sizeof...(NS)>({ NS... }, { indices... }));
    }
};

#define FUNCTION_CHOOSER(FnName, ...)                                                                                  \
    constexpr auto FnName##PtrGetter = []<int... args>() constexpr->auto { return &FnName<args...>; };                 \
    SpeciaLUT::Chooser<FnName##PtrGetter, __VA_ARGS__>

#if __has_include(<cuda_runtime.h>)

#    include <cuda_runtime.h>

struct CudaKernelExecution
{
    dim3 grid_dim_{};
    dim3 block_dim_{};
    size_t shmem_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
};

template<auto PtrGetter, std::size_t... NS>
class CudaKernel;

template<auto PtrGetter, std::size_t... NS>
class CudaChooser : public Chooser<PtrGetter, NS...>
{
    CudaKernelExecution exec_{};

public:
    CudaChooser() = default;
    ~CudaChooser() = default;

    CudaChooser& prepare(CudaKernelExecution const& exec)
    {
        exec_ = exec;
        return *this;
    }

    CudaChooser& prepare(dim3 grid_dim, dim3 block_dim, size_t shmem_bytes = 0, cudaStream_t stream = nullptr)
    {
        exec_ = { grid_dim, block_dim, shmem_bytes, stream };
        return *this;
    }

    template<typename... Indices>
    CudaKernel<PtrGetter, NS...> operator()(Indices... indices) const
    {
        static_assert(sizeof...(indices) == sizeof...(NS), "Template called with inappropriate number of arguments.");
        return { this->ptrs_.at(detail::flat_offset<sizeof...(NS)>({ NS... }, { indices... })), exec_ };
    }
};

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

    CudaKernel& prepare(CudaKernelExecution const& exec)
    {
        exec_ = exec;
        return *this;
    }

    CudaKernel& prepare(dim3 grid_dim, dim3 block_dim, size_t shmem_bytes = 0, cudaStream_t stream = nullptr)
    {
        exec_ = { grid_dim, block_dim, shmem_bytes, stream };
        return *this;
    }

    template<typename... Args>
    cudaError_t launch(Args&&... args) const
    {
        if (!fn_)
            return cudaErrorUnknown;
        auto args_ptrs = std::array<void*, sizeof...(args)>({ &args... });
        return cudaLaunchKernel(reinterpret_cast<const void*>(fn_), exec_.grid_dim_, exec_.block_dim_, args_ptrs.data(),
                                exec_.shmem_bytes_, exec_.stream_);
    }

    template<typename... Args>
    cudaError_t operator()(Args&&... args) const
    {
        return launch(std::forward<Args>(args)...);
    }
};

#    define CUDA_CHOOSER(KernelName, ...)                                                                              \
        constexpr auto KernelName##PtrGetter = []<int... args>() constexpr->auto { return &KernelName<args...>; };     \
        SpeciaLUT::CudaChooser<KernelName##PtrGetter, __VA_ARGS__>

#endif // end CUDA stuff

}
