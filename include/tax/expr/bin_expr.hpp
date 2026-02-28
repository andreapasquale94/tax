#pragma once

#include <tax/expr/base.hpp>
#include <tax/expr/arithmetic_ops.hpp>

namespace tax::detail {

/**
 * @brief Binary expression node parameterized by operation tag `Op`.
 * @details Stores leaf operands by reference and composite operands by value.
 */
template <typename L, typename R, typename Op>
class BinExpr
    : public tax::DAExpr<BinExpr<L, R, Op>, typename L::scalar_type, L::order, L::nvars>
{
    static_assert(L::order  == R::order  && L::nvars == R::nvars &&
                  std::is_same_v<typename L::scalar_type, typename R::scalar_type>);
public:
    using T = typename L::scalar_type;
    static constexpr int N = L::order, M = L::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    /// @brief Construct from left/right operands.
    constexpr BinExpr(const L& l, const R& r) noexcept : l_(l), r_(r) {}

    /**
     * @brief Evaluate operation result into `out`.
     * @details Uses leaf fast paths to avoid materializing both operands.
     */
    constexpr void evalTo(coeff_array& out) const noexcept {
        if constexpr (Op::is_additive) {
            l_.evalTo(out);
            if constexpr (Op::negate_right) r_.subTo(out);
            else                             r_.addTo(out);
        } else {
            if constexpr (is_leaf_v<L> && is_leaf_v<R>)
                Op::template apply<T>(out, l_.coeffs(), r_.coeffs());
            else if constexpr (is_leaf_v<R>) {
                coeff_array la{};
                l_.evalTo(la);
                Op::template apply<T>(out, la, r_.coeffs());
            } else if constexpr (is_leaf_v<L>) {
                coeff_array rb{};
                r_.evalTo(rb);
                Op::template apply<T>(out, l_.coeffs(), rb);
            } else {
                coeff_array la{}, rb{};
                l_.evalTo(la);
                r_.evalTo(rb);
                Op::template apply<T>(out, la, rb);
            }
        }
    }

    /**
     * @brief Accumulate operation result into `out`.
     * @details Additive and convolution ops have specialized accumulation paths.
     */
    constexpr void addTo(coeff_array& out) const noexcept {
        if constexpr (Op::is_additive) {
            l_.addTo(out);
            if constexpr (Op::negate_right) r_.subTo(out);
            else                             r_.addTo(out);
        } else if constexpr (Op::is_convolution) {
            if constexpr (is_leaf_v<L> && is_leaf_v<R>)
                cauchyAccumulate<T,N,M>(out, l_.coeffs(), r_.coeffs());
            else if constexpr (is_leaf_v<R>) {
                coeff_array la{};
                l_.evalTo(la);
                cauchyAccumulate<T,N,M>(out, la, r_.coeffs());
            } else if constexpr (is_leaf_v<L>) {
                coeff_array rb{};
                r_.evalTo(rb);
                cauchyAccumulate<T,N,M>(out, l_.coeffs(), rb);
            } else {
                coeff_array la{}, rb{};
                l_.evalTo(la);
                r_.evalTo(rb);
                cauchyAccumulate<T,N,M>(out, la, rb);
            }
        } else {
            coeff_array tmp{};
            evalTo(tmp);
            addInPlace<T, numMonomials(N,M)>(out, tmp);
        }
    }

    /// @brief Subtract operation result from `out`.
    constexpr void subTo(coeff_array& out) const noexcept {
        if constexpr (Op::is_additive) {
            l_.subTo(out);
            if constexpr (Op::negate_right) r_.addTo(out);
            else                             r_.subTo(out);
        } else {
            coeff_array tmp{};
            evalTo(tmp);
            subInPlace<T, numMonomials(N,M)>(out, tmp);
        }
    }

private:
    stored_t<L> l_;
    stored_t<R> r_;
};

} // namespace tax::detail
