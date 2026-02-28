#pragma once

#include <tax/expr/base.hpp>

namespace tax::detail {

/**
 * @brief Flattened variadic product expression node.
 * @details Evaluates products using a rolling accumulator to limit temporaries.
 */
template <typename... Es>
class ProductExpr
    : public tax::DAExpr<ProductExpr<Es...>,
                    typename std::tuple_element_t<0, std::tuple<Es...>>::scalar_type,
                    std::tuple_element_t<0, std::tuple<Es...>>::order,
                    std::tuple_element_t<0, std::tuple<Es...>>::nvars>
{
    static_assert(sizeof...(Es) >= 2, "ProductExpr needs at least 2 operands");
    template <typename...> friend class ProductExpr;

public:
    using T = typename std::tuple_element_t<0, std::tuple<Es...>>::scalar_type;
    static constexpr int N = std::tuple_element_t<0, std::tuple<Es...>>::order;
    static constexpr int M = std::tuple_element_t<0, std::tuple<Es...>>::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    /// @brief Construct from operand pack.
    explicit constexpr ProductExpr(stored_t<Es>... es) noexcept : ops_(es...) {}

    template <typename E>
    /// @brief Return a new `ProductExpr` with `e` appended.
    [[nodiscard]] constexpr auto append(stored_t<E> e) const noexcept {
        return std::apply([&](auto const&... x) noexcept {
            return ProductExpr<Es..., E>(x..., e);
        }, ops_);
    }

    /// @brief Evaluate the full product into `out`.
    constexpr void evalTo(coeff_array& out) const noexcept {
        coeff_array a{};
        seedAccumulator(a);
        rollProduct<1>(out, a);
    }

    /// @brief Accumulate this product into `out`.
    constexpr void addTo(coeff_array& out) const noexcept {
        if constexpr (sizeof...(Es) == 2) {
            using L = std::tuple_element_t<0, std::tuple<Es...>>;
            using R = std::tuple_element_t<1, std::tuple<Es...>>;
            const auto& lop = std::get<0>(ops_);
            const auto& rop = std::get<1>(ops_);
            if constexpr (is_leaf_v<L> && is_leaf_v<R>)
                cauchyAccumulate<T,N,M>(out, lop.coeffs(), rop.coeffs());
            else if constexpr (is_leaf_v<R>) {
                coeff_array la{};
                lop.evalTo(la);
                cauchyAccumulate<T,N,M>(out, la, rop.coeffs());
            } else if constexpr (is_leaf_v<L>) {
                coeff_array rb{};
                rop.evalTo(rb);
                cauchyAccumulate<T,N,M>(out, lop.coeffs(), rb);
            } else {
                coeff_array la{}, rb{};
                lop.evalTo(la);
                rop.evalTo(rb);
                cauchyAccumulate<T,N,M>(out, la, rb);
            }
        } else {
            coeff_array tmp{};
            evalTo(tmp);
            addInPlace<T, numMonomials(N,M)>(out, tmp);
        }
    }

    /// @brief Subtract this product from `out`.
    constexpr void subTo(coeff_array& out) const noexcept {
        coeff_array tmp{};
        evalTo(tmp);
        subInPlace<T, numMonomials(N,M)>(out, tmp);
    }

private:
    std::tuple<stored_t<Es>...> ops_;

    constexpr void seedAccumulator(coeff_array& a) const noexcept {
        using E0 = std::tuple_element_t<0, std::tuple<Es...>>;
        if constexpr (is_leaf_v<E0>) a = std::get<0>(ops_).coeffs();
        else                         std::get<0>(ops_).evalTo(a);
    }

    template <std::size_t Start>
    constexpr void rollProduct(coeff_array& out, coeff_array& a) const noexcept {
        [&]<std::size_t... I>(std::index_sequence<I...>) noexcept {
            (productStep<I + Start>(out, a), ...);
        }(std::make_index_sequence<sizeof...(Es) - Start>{});
    }

    template <std::size_t I>
    constexpr void productStep(coeff_array& out, coeff_array& a) const noexcept {
        using Ei = std::tuple_element_t<I, std::tuple<Es...>>;
        if constexpr (is_leaf_v<Ei>) {
            cauchyProduct<T, N, M>(out, a, std::get<I>(ops_).coeffs());
        } else {
            coeff_array b{};
            std::get<I>(ops_).evalTo(b);
            cauchyProduct<T, N, M>(out, a, b);
        }
        a = out;
    }
};

} // namespace tax::detail
