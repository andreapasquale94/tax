#pragma once

#include <cmath>
#include <tax/utils/combinatorics.hpp>

namespace tax::detail {

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

template <typename T, int N, int M>
constexpr void seriesSinhCosh(
    std::array<T, numMonomials(N, M)>& sh,
    std::array<T, numMonomials(N, M)>& ch,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    using std::sinh; using std::cosh;
    sh = {}; ch = {};
    sh[0] = sinh(a[0]);
    ch[0] = cosh(a[0]);

    if constexpr (M == 1) {
        for (int d = 1; d <= N; ++d) {
            T sr = T{0}, cr = T{0};
            for (int k = 0; k < d; ++k) {
                const T w = T(d - k) * a[d - k];
                sr += w * ch[k];
                cr += w * sh[k];
            }
            const T inv_d = T{1} / T(d);
            sh[d] = sr * inv_d;
            ch[d] = cr * inv_d;
        }
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const std::size_t ai = flatIndex<M>(alpha);
                T sh_rhs = T{0};
                T ch_rhs = T{0};
                for (int db = 0; db < d; ++db) {
                    const T dg = T(d - db);
                    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                        if (bvar == M - 1) {
                            beta[bvar] = brem;
                            if (beta[bvar] > alpha[bvar]) return;
                            tax::MultiIndex<M> gamma{};
                            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                            const T fg = a[flatIndex<M>(gamma)];
                            sh_rhs += dg * fg * ch[flatIndex<M>(beta)];
                            ch_rhs += dg * fg * sh[flatIndex<M>(beta)];
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
                sh[ai] = sh_rhs * inv_d;
                ch[ai] = ch_rhs * inv_d;
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

template <typename T, int N, int M>
constexpr void seriesSinh(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    std::array<T, numMonomials(N, M)> ch{};
    seriesSinhCosh<T, N, M>(out, ch, a);
}

template <typename T, int N, int M>
constexpr void seriesCosh(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    std::array<T, numMonomials(N, M)> sh{};
    seriesSinhCosh<T, N, M>(sh, out, a);
}

template <typename T, int N, int M>
constexpr void seriesTanh(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    constexpr auto S = numMonomials(N, M);
    std::array<T, S> sh{}, ch{};
    seriesSinhCosh<T, N, M>(sh, ch, a);

    out = {};
    const T inv_ch0 = T{1} / ch[0];

    if constexpr (M == 1) {
        for (int d = 0; d <= N; ++d) {
            T rhs = sh[d];
            for (int k = 1; k <= d; ++k)
                rhs -= ch[k] * out[d - k];
            out[d] = rhs * inv_ch0;
        }
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const std::size_t ai = flatIndex<M>(alpha);
                T rhs = sh[ai];
                for (int db = 1; db <= d; ++db) {
                    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                        if (bvar == M - 1) {
                            beta[bvar] = brem;
                            if (beta[bvar] > alpha[bvar]) return;
                            tax::MultiIndex<M> gamma{};
                            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                            rhs -= ch[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = rhs * inv_ch0;
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

} // namespace tax::detail
