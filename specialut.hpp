#pragma once

#include <array>
#include <utility>

namespace SpeciaLUT {


    namespace detail
    {

        template <auto F> struct Signature;
        template <typename R, typename ... A, R (*F)(A...)>
        struct Signature<F> {
            using value = R(A...);
        };

        template<std::size_t ND>
        int flat_offset(std::array<std::size_t, ND> ns, std::array<int, ND> is)
        {
            int off = 0;
            for (int i = ND-1; i >= 0; i--)
                off = is[i] + off * ns[i];
            return off;
        }

        template<std::size_t ND>
        constexpr std::size_t unflatten(std::size_t i, std::size_t target_level, std::array<std::size_t, ND> const ns, std::size_t level = 0, std::size_t product = 1)
        {
            if (level == target_level)
                return ((i / product) % ns[level]);
            return unflatten<ND>(i, target_level, ns, level + 1, product * ns[level]);
        }

    }


    /// Runtime choosing of specialized template functions
    template <typename FnStruct, std::size_t... NS>
    class Chooser {

    private:

        static constexpr std::size_t n_dims_ = sizeof...(NS);
        static constexpr std::size_t n_ptrs_ = (NS * ...);

        using FnSignature = typename detail::Signature<FnStruct::template run<(NS*0)...>>::value;
        using FnLUT = std::array<FnSignature*, n_ptrs_>;

        template<int i, std::size_t... I>
        static constexpr FnSignature* fn_ptr(const std::index_sequence<I...>)
        {
            return &FnStruct::template run< detail::unflatten<n_dims_>(i, I, {NS...})... >;
        }

        template<std::size_t... I>
        static constexpr FnLUT make_lut(std::index_sequence<I...>)
        {
            return {(fn_ptr<I>(std::make_index_sequence<n_dims_>{}))...};
        }

        static constexpr FnLUT ptrs_ = make_lut(std::make_index_sequence<n_ptrs_>{});

    public:

        Chooser() = default;
        ~Chooser() = default;

        /// Get the specialized function deduced from given runtime parameters
        template <typename...Indices>
        FnSignature const& operator()(Indices... indices) const
        {
            static_assert (sizeof...(indices) == sizeof...(NS), "Template called with inappropriate number of arguments.");
            return *ptrs_[detail::flat_offset<sizeof...(NS)>({NS...}, {indices...})];
        }

    };


}
