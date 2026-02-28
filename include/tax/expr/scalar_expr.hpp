#pragma once

#include <tax/expr/base.hpp>
#include <tax/expr/arithmetic_ops.hpp>

namespace tax::detail {

/**
 * @brief Expression node applying scalar operation tag `Op`.
 * @details Evaluates `E` once, then applies scalar modification in place.
 */
template <typename E, typename Op>
class ScalarExpr
    : public tax::DAExpr<ScalarExpr<E, Op>, typename E::scalar_type, E::order, E::nvars>
{
public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    /// @brief Construct from operand and scalar value.
    constexpr ScalarExpr(const E& e, T s) noexcept : e_(e), s_(s) {}

    /// @brief Evaluate operation result into `out`.
    constexpr void evalTo(coeff_array& out) const noexcept {
        e_.evalTo(out);
        Op::template apply<T, numMonomials(N, M)>(out, s_);
    }
private:
    stored_t<E> e_;
    T s_;
};

} // namespace tax::detail
