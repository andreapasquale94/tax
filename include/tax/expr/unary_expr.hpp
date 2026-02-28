#pragma once

#include <tax/expr/base.hpp>
#include <tax/expr/arithmetic_ops.hpp>

namespace da::detail {

// =============================================================================
// UnaryExpr<E, Op> — unary in-place expression node
// =============================================================================

template <typename E, typename Op>
class UnaryExpr
    : public da::DAExpr<UnaryExpr<E, Op>, typename E::scalar_type, E::order, E::nvars>
{
public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    explicit constexpr UnaryExpr(const E& e) noexcept : e_(e) {}

    constexpr void evalTo(coeff_array& out) const noexcept {
        e_.evalTo(out);
        Op::template apply<T, numMonomials(N, M)>(out);
    }

    constexpr void addTo(coeff_array& out) const noexcept {
        if constexpr (std::is_same_v<Op, OpNeg>) e_.subTo(out);
        else { coeff_array tmp{}; evalTo(tmp); addInPlace<T,numMonomials(N,M)>(out,tmp); }
    }
    constexpr void subTo(coeff_array& out) const noexcept {
        if constexpr (std::is_same_v<Op, OpNeg>) e_.addTo(out);
        else { coeff_array tmp{}; evalTo(tmp); subInPlace<T,numMonomials(N,M)>(out,tmp); }
    }

private:
    stored_t<E> e_;
};

} // namespace da::detail
