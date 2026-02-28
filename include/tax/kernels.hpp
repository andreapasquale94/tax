#pragma once

#include <cmath>
#include <tax/combinatorics.hpp>

namespace da::detail {

// -- Element-wise arithmetic --------------------------------------------------

template <typename T, std::size_t S>
constexpr void addInPlace(std::array<T, S>& o, const std::array<T, S>& r) noexcept
{ for (std::size_t i = 0; i < S; ++i) o[i] += r[i]; }

template <typename T, std::size_t S>
constexpr void subInPlace(std::array<T, S>& o, const std::array<T, S>& r) noexcept
{ for (std::size_t i = 0; i < S; ++i) o[i] -= r[i]; }

template <typename T, std::size_t S>
constexpr void negateInPlace(std::array<T, S>& o) noexcept
{ for (auto& v : o) v = -v; }

template <typename T, std::size_t S>
constexpr void scaleInPlace(std::array<T, S>& o, T s) noexcept
{ for (auto& v : o) v *= s; }

// -- Multivariate Cauchy product ----------------------------------------------

template <typename T, int N, int M>
constexpr void cauchyProduct(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& f,
    const std::array<T, numMonomials(N, M)>& g) noexcept
{
    out = {};
    da::MultiIndex<M> alpha{};
    da::MultiIndex<M> beta{};

    auto fillBeta = [&](auto& self, int bvar) -> void {
        if (bvar == M) {
            da::MultiIndex<M> gamma{};
            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
            out[flatIndex<M>(alpha)] +=
                f[flatIndex<M>(beta)] * g[flatIndex<M>(gamma)];
            return;
        }
        for (int b = 0; b <= alpha[bvar]; ++b) { beta[bvar] = b; self(self, bvar + 1); }
    };

    auto fillAlpha = [&](auto& self, int var, int rem) -> void {
        if (var == M - 1) { alpha[var] = rem; fillBeta(fillBeta, 0); return; }
        for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k); }
    };

    for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d);
}

// -- Multivariate Cauchy accumulate (out += f*g) ------------------------------

template <typename T, int N, int M>
constexpr void cauchyAccumulate(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& f,
    const std::array<T, numMonomials(N, M)>& g) noexcept
{
    da::MultiIndex<M> alpha{};
    da::MultiIndex<M> beta{};

    auto fillBeta = [&](auto& self, int bvar) -> void {
        if (bvar == M) {
            da::MultiIndex<M> gamma{};
            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
            out[flatIndex<M>(alpha)] +=
                f[flatIndex<M>(beta)] * g[flatIndex<M>(gamma)];
            return;
        }
        for (int b = 0; b <= alpha[bvar]; ++b) { beta[bvar] = b; self(self, bvar + 1); }
    };

    auto fillAlpha = [&](auto& self, int var, int rem) -> void {
        if (var == M - 1) { alpha[var] = rem; fillBeta(fillBeta, 0); return; }
        for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k); }
    };

    for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d);
}

// -- Series reciprocal: solve a * out = 1 -------------------------------------

template <typename T, int N, int M>
constexpr void seriesReciprocal(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    out = {};
    const T inv_a0 = T{1} / a[0];
    da::MultiIndex<M> alpha{};
    da::MultiIndex<M> beta{};

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
                        da::MultiIndex<M> gamma{};
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

// -- Series square: out = a^2 -------------------------------------------------

template <typename T, int N, int M>
constexpr void seriesSquare(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    cauchyProduct<T, N, M>(out, a, a);
}

// -- Series cube: out = a^3 via direct triple convolution ---------------------

template <typename T, int N, int M>
constexpr void seriesCube(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    out = {};
    da::MultiIndex<M> alpha{};
    da::MultiIndex<M> beta{};
    da::MultiIndex<M> gamma{};

    // fillGamma: enumerate gamma with |gamma|=dg, gamma[i] <= (alpha-beta)[i]
    auto fillGamma = [&](auto& gself, int gvar, int grem) -> void {
        if (gvar == M - 1) {
            gamma[gvar] = grem;
            if (gamma[gvar] > alpha[gvar] - beta[gvar]) return;
            da::MultiIndex<M> delta{};
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

    // fillBeta: enumerate beta with |beta|=db, beta[i] <= alpha[i]
    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
        if (bvar == M - 1) {
            beta[bvar] = brem;
            if (beta[bvar] > alpha[bvar]) return;
            // |gamma| ranges from 0 to |alpha - beta|
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

    // fillAlpha: enumerate alpha in grlex order (by total degree, then grlex)
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

// -- Series square root: solve g*g = a ----------------------------------------

template <typename T, int N, int M>
constexpr void seriesSqrt(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    using std::sqrt;
    out    = {};
    out[0] = sqrt(a[0]);
    const T inv2g0 = T{1} / (T{2} * out[0]);

    da::MultiIndex<M> alpha{};
    da::MultiIndex<M> beta{};

    auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
        if (var == M - 1) {
            alpha[var] = rem;
            const std::size_t ai = flatIndex<M>(alpha);
            T rhs = a[ai];
            // sum over beta: 0 < |beta| < d, beta[i] <= alpha[i]
            for (int db = 1; db < d; ++db) {
                auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                    if (bvar == M - 1) {
                        beta[bvar] = brem;
                        if (beta[bvar] > alpha[bvar]) return;
                        da::MultiIndex<M> gamma{};
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

} // namespace da::detail
