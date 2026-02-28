#pragma once

#include <tax/expr/base.hpp>
#include <tax/expr/math_ops.hpp>

namespace tax::detail {
    
/**
 * @brief Expression node applying a series function kernel `Op`.
 * @details For leaf operands, coefficients are passed directly to the kernel;
 * otherwise the operand is materialized once into a temporary buffer.
 */
template <typename E, typename Op>
class FuncExpr
    : public tax::DAExpr<FuncExpr<E, Op>, typename E::scalar_type, E::order, E::nvars>
{
public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    /// @brief Construct from operand.
    explicit constexpr FuncExpr(const E& e) noexcept : e_(e) {}

    /// @brief Evaluate function result into `out`.
    constexpr void evalTo(coeff_array& out) const noexcept {
        if constexpr (is_leaf_v<E>) {
            Op::template apply<T>(out, e_.coeffs());
        } else {
            coeff_array a{};
            e_.evalTo(a);
            Op::template apply<T>(out, a);
        }
    }

private:
    stored_t<E> e_;
};

} // namespace tax::detail
