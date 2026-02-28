#pragma once

#include <cmath>
#include <tax/kernels/ops.hpp>
#include <tax/kernels/cauchy.hpp>

namespace tax::detail {

template <typename T, int N, int M>
/**
 * @brief Reciprocal series solve `a * out = 1`.
 * @details Requires `a[0] != 0`.
 */
constexpr void seriesReciprocal(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    out = {};
    const T inv_a0 = T{1} / a[0];

    if constexpr (M == 1) {
        out[0] = inv_a0;
        for (int d = 1; d <= N; ++d) {
            T rhs = T{0};
            for (int k = 1; k <= d; ++k)
                rhs -= a[k] * out[d - k];
            out[d] = rhs * inv_a0;
        }
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const std::size_t ai = flatIndex<M>(alpha);
                T rhs = (d == 0) ? T{1} : T{0};
                for (int db = 1; db <= d; ++db) {
                    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                        if (bvar == M - 1) {
                            beta[bvar] = brem;
                            if (beta[bvar] > alpha[bvar]) return;
                            tax::MultiIndex<M> gamma{};
                            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                            rhs -= a[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = rhs * inv_a0;
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

template <typename T, int N, int M>
/// @brief Square series `out = a^2`.
constexpr void seriesSquare(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    cauchyProduct<T, N, M>(out, a, a);
}

template <typename T, int N, int M>
/// @brief Cube series `out = a^3`.
constexpr void seriesCube(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    out = {};

    if constexpr (M == 1) {
        for (int d = 0; d <= N; ++d)
            for (int j = 0; j <= d; ++j)
                for (int k = 0; k <= d - j; ++k)
                    out[d] += a[j] * a[k] * a[d - j - k];
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};
        tax::MultiIndex<M> gamma{};

        auto fillGamma = [&](auto& gself, int gvar, int grem) -> void {
            if (gvar == M - 1) {
                gamma[gvar] = grem;
                if (gamma[gvar] > alpha[gvar] - beta[gvar]) return;
                tax::MultiIndex<M> delta{};
                for (int i = 0; i < M; ++i) delta[i] = alpha[i] - beta[i] - gamma[i];
                out[flatIndex<M>(alpha)] +=
                    a[flatIndex<M>(beta)] * a[flatIndex<M>(gamma)] * a[flatIndex<M>(delta)];
                return;
            }
            const int maxg = alpha[gvar] - beta[gvar];
            for (int g = 0; g <= std::min(grem, maxg); ++g) {
                gamma[gvar] = g;
                gself(gself, gvar + 1, grem - g);
            }
        };

        auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
            if (bvar == M - 1) {
                beta[bvar] = brem;
                if (beta[bvar] > alpha[bvar]) return;
                int ab_total = 0;
                for (int i = 0; i < M; ++i) ab_total += alpha[i] - beta[i];
                for (int dg = 0; dg <= ab_total; ++dg)
                    fillGamma(fillGamma, 0, dg);
                return;
            }
            for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                beta[bvar] = b;
                bself(bself, bvar + 1, brem - b);
            }
        };

        auto fillAlpha = [&](auto& aself, int var, int rem) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const int d = totalDegree<M>(alpha);
                for (int db = 0; db <= d; ++db)
                    fillBeta(fillBeta, 0, db);
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; aself(aself, var + 1, rem - k); }
        };

        for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d);
    }
}

template <typename T, int N, int M>
/**
 * @brief Square-root series solve `out * out = a`.
 * @details Uses the principal branch from `sqrt(a[0])`.
 */
constexpr void seriesSqrt(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    using std::sqrt;
    out    = {};
    out[0] = sqrt(a[0]);
    const T inv2g0 = T{1} / (T{2} * out[0]);

    if constexpr (M == 1) {
        for (int d = 1; d <= N; ++d) {
            T rhs = a[d];
            for (int k = 1; k < d; ++k)
                rhs -= out[k] * out[d - k];
            out[d] = rhs * inv2g0;
        }
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const std::size_t ai = flatIndex<M>(alpha);
                T rhs = a[ai];
                for (int db = 1; db < d; ++db) {
                    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                        if (bvar == M - 1) {
                            beta[bvar] = brem;
                            if (beta[bvar] > alpha[bvar]) return;
                            tax::MultiIndex<M> gamma{};
                            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                            rhs -= out[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = rhs * inv2g0;
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

} // namespace tax::detail
