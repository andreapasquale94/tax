#pragma once

#include <cmath>
#include <tax/combinatorics.hpp>

namespace tax::detail {

// -- Element-wise arithmetic --------------------------------------------------

template <typename T, std::size_t S>
/// @brief In-place element-wise addition: `o += r`.
constexpr void addInPlace(std::array<T, S>& o, const std::array<T, S>& r) noexcept
{ for (std::size_t i = 0; i < S; ++i) o[i] += r[i]; }

template <typename T, std::size_t S>
/// @brief In-place element-wise subtraction: `o -= r`.
constexpr void subInPlace(std::array<T, S>& o, const std::array<T, S>& r) noexcept
{ for (std::size_t i = 0; i < S; ++i) o[i] -= r[i]; }

template <typename T, std::size_t S>
/// @brief In-place sign flip.
constexpr void negateInPlace(std::array<T, S>& o) noexcept
{ for (auto& v : o) v = -v; }

template <typename T, std::size_t S>
/// @brief In-place scalar multiply.
constexpr void scaleInPlace(std::array<T, S>& o, T s) noexcept
{ for (auto& v : o) v *= s; }

// -- Cauchy product -----------------------------------------------------------

template <typename T, int N, int M>
/**
 * @brief Truncated multivariate Cauchy product `out = f * g`.
 * @details Output is truncated to total degree `N`.
 */
constexpr void cauchyProduct(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& f,
    const std::array<T, numMonomials(N, M)>& g) noexcept
{
    out = {};

    if constexpr (M == 1) {
        for (int d = 0; d <= N; ++d)
            for (int k = 0; k <= d; ++k)
                out[d] += f[k] * g[d - k];
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillBeta = [&](auto& self, int bvar) -> void {
            if (bvar == M) {
                tax::MultiIndex<M> gamma{};
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
}

// -- Cauchy accumulate (out += f*g) -------------------------------------------

template <typename T, int N, int M>
/**
 * @brief Truncated multivariate Cauchy accumulate `out += f * g`.
 * @details Contribution is truncated to total degree `N`.
 */
constexpr void cauchyAccumulate(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& f,
    const std::array<T, numMonomials(N, M)>& g) noexcept
{
    if constexpr (M == 1) {
        for (int d = 0; d <= N; ++d)
            for (int k = 0; k <= d; ++k)
                out[d] += f[k] * g[d - k];
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillBeta = [&](auto& self, int bvar) -> void {
            if (bvar == M) {
                tax::MultiIndex<M> gamma{};
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
}

// -- Series reciprocal: solve a * out = 1 -------------------------------------

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

// -- Series square: out = a^2 -------------------------------------------------

template <typename T, int N, int M>
/// @brief Square series `out = a^2`.
constexpr void seriesSquare(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    cauchyProduct<T, N, M>(out, a, a);
}

// -- Series cube: out = a^3 via direct triple convolution ---------------------

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

// -- Series square root: solve g*g = a ----------------------------------------

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

// -- Series sin & cos: coupled recurrence via Euler homogeneity ---------------
//
//   d·s[α] =  Σ_{β+γ=α, |γ|≥1} |γ|·f[γ]·c[β]
//   d·c[α] = -Σ_{β+γ=α, |γ|≥1} |γ|·f[γ]·s[β]
//
// where d = |α|, processed in graded order so all lower-degree terms are ready.

template <typename T, int N, int M>
/**
 * @brief Coupled trigonometric series expansion of `sin(a)` and `cos(a)`.
 * @details Computes both outputs together to share recurrence work.
 */
constexpr void seriesSinCos(
    std::array<T, numMonomials(N, M)>& s,
    std::array<T, numMonomials(N, M)>& c,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    using std::sin; using std::cos;
    s = {}; c = {};
    s[0] = sin(a[0]);
    c[0] = cos(a[0]);

    if constexpr (M == 1) {
        // 1D: s[d] = (1/d) * sum_{k=0}^{d-1} (d-k) * a[d-k] * c[k]
        //     c[d] = -(1/d) * sum_{k=0}^{d-1} (d-k) * a[d-k] * s[k]
        for (int d = 1; d <= N; ++d) {
            T sr = T{0}, cr = T{0};
            for (int k = 0; k < d; ++k) {
                const T w = T(d - k) * a[d - k];
                sr += w * c[k];
                cr += w * s[k];
            }
            const T inv_d = T{1} / T(d);
            s[d] =  sr * inv_d;
            c[d] = -cr * inv_d;
        }
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const std::size_t ai = flatIndex<M>(alpha);
                T sin_rhs = T{0};
                T cos_rhs = T{0};
                for (int db = 0; db < d; ++db) {
                    const T dg = T(d - db);
                    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                        if (bvar == M - 1) {
                            beta[bvar] = brem;
                            if (beta[bvar] > alpha[bvar]) return;
                            tax::MultiIndex<M> gamma{};
                            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                            const T fg = a[flatIndex<M>(gamma)];
                            sin_rhs += dg * fg * c[flatIndex<M>(beta)];
                            cos_rhs += dg * fg * s[flatIndex<M>(beta)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                const T inv_d = T{1} / T(d);
                s[ai] =  sin_rhs * inv_d;
                c[ai] = -cos_rhs * inv_d;
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

// -- Standalone sin / cos wrappers --------------------------------------------

template <typename T, int N, int M>
/// @brief Sine series wrapper around `seriesSinCos`.
constexpr void seriesSin(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    std::array<T, numMonomials(N, M)> c{};
    seriesSinCos<T, N, M>(out, c, a);
}

template <typename T, int N, int M>
/// @brief Cosine series wrapper around `seriesSinCos`.
constexpr void seriesCos(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    std::array<T, numMonomials(N, M)> s{};
    seriesSinCos<T, N, M>(s, out, a);
}

// -- Series tan: compute sin & cos, then solve c·t = s -----------------------

template <typename T, int N, int M>
/**
 * @brief Tangent series solve from `sin(a)` and `cos(a)`.
 * @details Solves `cos(a) * out = sin(a)` degree by degree.
 */
constexpr void seriesTan(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    constexpr auto S = numMonomials(N, M);
    std::array<T, S> s{}, c{};
    seriesSinCos<T, N, M>(s, c, a);

    // Solve c · out = s  degree by degree (same structure as seriesReciprocal)
    out = {};
    const T inv_c0 = T{1} / c[0];

    if constexpr (M == 1) {
        for (int d = 0; d <= N; ++d) {
            T rhs = s[d];
            for (int k = 1; k <= d; ++k)
                rhs -= c[k] * out[d - k];
            out[d] = rhs * inv_c0;
        }
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const std::size_t ai = flatIndex<M>(alpha);
                T rhs = s[ai];
                for (int db = 1; db <= d; ++db) {
                    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                        if (bvar == M - 1) {
                            beta[bvar] = brem;
                            if (beta[bvar] > alpha[bvar]) return;
                            tax::MultiIndex<M> gamma{};
                            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                            rhs -= c[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = rhs * inv_c0;
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

// -- Series log: g = log(f), from f·g' = f' via Euler homogeneity ------------
//
//   g[0] = log(f[0])
//   g[α] = f[α]/f[0] - (1/(d·f[0])) Σ_{β+γ=α, |β|≥1, |γ|≥1} |γ|·f[β]·g[γ]
//
// where d = |α|.

template <typename T, int N, int M>
/**
 * @brief Natural logarithm series `out = log(a)`.
 * @details Requires `a[0] > 0` for real-valued output.
 */
constexpr void seriesLog(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    using std::log;
    out = {};
    out[0] = log(a[0]);
    const T inv_a0 = T{1} / a[0];

    if constexpr (M == 1) {
        for (int d = 1; d <= N; ++d) {
            T rhs = T{0};
            for (int k = 1; k < d; ++k)
                rhs += T(k) * a[d - k] * out[k];
            out[d] = (a[d] - rhs / T(d)) * inv_a0;
        }
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const std::size_t ai = flatIndex<M>(alpha);
                T rhs = T{0};
                for (int db = 1; db < d; ++db) {
                    const int dg = d - db;
                    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                        if (bvar == M - 1) {
                            beta[bvar] = brem;
                            if (beta[bvar] > alpha[bvar]) return;
                            tax::MultiIndex<M> gamma{};
                            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                            rhs += T(dg) * a[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = (a[ai] - rhs / T(d)) * inv_a0;
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

} // namespace tax::detail
