#pragma once

#include <tax/fwd.hpp>
#include <tax/combinatorics.hpp>
#include <tax/kernels.hpp>
#include <tax/leaf.hpp>

namespace da {

// =============================================================================
// CRTP expression base
// =============================================================================

template <typename Derived, typename T, int N, int M>
struct DAExpr {
    using scalar_type                  = T;
    static constexpr int  order        = N;
    static constexpr int  nvars        = M;
    static constexpr std::size_t ncoef = detail::numMonomials(N, M);
    using coeff_array                  = std::array<T, ncoef>;

    [[nodiscard]] constexpr const Derived& self() const noexcept
    { return static_cast<const Derived&>(*this); }

    constexpr void evalTo(coeff_array& out) const noexcept { self().evalTo(out); }

    // Default addTo/subTo: materialise via evalTo then accumulate.
    // Overridden by DA (direct), BinExpr (recursive), UnaryExpr<OpNeg> (sign flip).
    constexpr void addTo(coeff_array& out) const noexcept {
        coeff_array tmp{};
        self().evalTo(tmp);
        detail::addInPlace<T, ncoef>(out, tmp);
    }
    constexpr void subTo(coeff_array& out) const noexcept {
        coeff_array tmp{};
        self().evalTo(tmp);
        detail::subInPlace<T, ncoef>(out, tmp);
    }

    [[nodiscard]] constexpr coeff_array eval() const noexcept {
        coeff_array out{};
        self().evalTo(out);
        return out;
    }

    [[nodiscard]] constexpr T value() const noexcept { return eval()[0]; }

    [[nodiscard]] constexpr T coeff(const MultiIndex<M>& alpha) const noexcept
    { return eval()[detail::flatIndex<M>(alpha)]; }

    [[nodiscard]] constexpr T derivative(const MultiIndex<M>& alpha) const noexcept {
        std::size_t fac = 1;
        for (int i = 0; i < M; ++i) for (int j = 1; j <= alpha[i]; ++j) fac *= std::size_t(j);
        return eval()[detail::flatIndex<M>(alpha)] * T(fac);
    }
};

} // namespace da
