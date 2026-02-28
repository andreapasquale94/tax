#pragma once

#include <cmath>
#include <tax/kernels/ops.hpp>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/algebra.hpp>

namespace tax::detail {

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

template <typename T, int N, int M>
constexpr void seriesExp(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    using std::exp;
    out = {};
    out[0] = exp(a[0]);

    if constexpr (M == 1) {
        for (int d = 1; d <= N; ++d) {
            T rhs = T{0};
            for (int k = 0; k < d; ++k)
                rhs += T(d - k) * a[d - k] * out[k];
            out[d] = rhs / T(d);
        }
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const std::size_t ai = flatIndex<M>(alpha);
                T rhs = T{0};
                for (int db = 1; db <= d; ++db) {
                    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                        if (bvar == M - 1) {
                            beta[bvar] = brem;
                            if (beta[bvar] > alpha[bvar]) return;
                            tax::MultiIndex<M> gamma{};
                            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                            rhs += T(db) * a[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = rhs / T(d);
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

template <typename T, int N, int M>
constexpr void seriesPow(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a,
    T c) noexcept
{
    using std::pow;
    out = {};
    out[0] = pow(a[0], c);
    const T inv_a0 = T{1} / a[0];

    if constexpr (M == 1) {
        for (int d = 1; d <= N; ++d) {
            T rhs = T{0};
            for (int k = 0; k < d; ++k)
                rhs += (c * T(d - k) - T(k)) * a[d - k] * out[k];
            out[d] = rhs * inv_a0 / T(d);
        }
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const std::size_t ai = flatIndex<M>(alpha);
                T rhs = T{0};
                for (int db = 1; db <= d; ++db) {
                    const int dg = d - db;
                    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                        if (bvar == M - 1) {
                            beta[bvar] = brem;
                            if (beta[bvar] > alpha[bvar]) return;
                            tax::MultiIndex<M> gamma{};
                            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                            rhs += (c * T(db) - T(dg)) * a[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = rhs * inv_a0 / T(d);
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

template <typename T, int N, int M>
constexpr void seriesIntPow(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a,
    int n) noexcept
{
    constexpr auto S = numMonomials(N, M);

    if (n == 0) { out = {}; out[0] = T{1}; return; }
    if (n == 1) { out = a; return; }
    if (n == -1) { seriesReciprocal<T, N, M>(out, a); return; }
    if (n < 0) {
        std::array<T, S> rec{};
        seriesReciprocal<T, N, M>(rec, a);
        seriesIntPow<T, N, M>(out, rec, -n);
        return;
    }
    // n >= 2: binary exponentiation
    std::array<T, S> base = a;
    out = {};
    out[0] = T{1};
    int e = n;
    while (e > 0) {
        if (e & 1) {
            std::array<T, S> tmp{};
            cauchyProduct<T, N, M>(tmp, out, base);
            out = tmp;
        }
        e >>= 1;
        if (e > 0) {
            std::array<T, S> tmp{};
            cauchyProduct<T, N, M>(tmp, base, base);
            base = tmp;
        }
    }
}

template <typename T, int N, int M>
constexpr void seriesErf(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    using std::erf; using std::exp; using std::sqrt;
    constexpr auto S = numMonomials(N, M);
    constexpr T two_over_sqrtpi = T{2} / sqrt(std::acos(T{-1}));

    // h = (2/√π) · exp(-a²)
    std::array<T, S> asq{}, neg_asq{}, e{}, h{};
    cauchyProduct<T, N, M>(asq, a, a);
    neg_asq = asq;
    negateInPlace<T, S>(neg_asq);
    seriesExp<T, N, M>(e, neg_asq);
    h = e;
    scaleInPlace<T, S>(h, two_over_sqrtpi);

    out = {};
    out[0] = erf(a[0]);

    if constexpr (M == 1) {
        for (int d = 1; d <= N; ++d) {
            T rhs = T{0};
            for (int k = 0; k < d; ++k)
                rhs += T(d - k) * a[d - k] * h[k];
            out[d] = rhs / T(d);
        }
    } else {
        tax::MultiIndex<M> alpha{};
        tax::MultiIndex<M> beta{};

        auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                const std::size_t ai = flatIndex<M>(alpha);
                T rhs = T{0};
                for (int db = 1; db <= d; ++db) {
                    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                        if (bvar == M - 1) {
                            beta[bvar] = brem;
                            if (beta[bvar] > alpha[bvar]) return;
                            tax::MultiIndex<M> gamma{};
                            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                            rhs += T(db) * a[flatIndex<M>(beta)] * h[flatIndex<M>(gamma)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = rhs / T(d);
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

template <typename T, int N, int M>
constexpr void seriesAsin(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    using std::asin;
    constexpr auto S = numMonomials(N, M);

    // h = sqrt(1 - a²)
    std::array<T, S> asq{}, omf{}, h{};
    cauchyProduct<T, N, M>(asq, a, a);
    omf = {};
    omf[0] = T{1};
    subInPlace<T, S>(omf, asq);
    seriesSqrt<T, N, M>(h, omf);

    out = {};
    out[0] = asin(a[0]);
    const T inv_h0 = T{1} / h[0];

    if constexpr (M == 1) {
        for (int d = 1; d <= N; ++d) {
            T rhs = T{0};
            for (int k = 1; k < d; ++k)
                rhs += T(k) * h[d - k] * out[k];
            out[d] = (a[d] - rhs / T(d)) * inv_h0;
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
                            rhs += T(dg) * h[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = (a[ai] - rhs / T(d)) * inv_h0;
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

template <typename T, int N, int M>
constexpr void seriesAcos(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    seriesAsin<T, N, M>(out, a);
    negateInPlace<T, numMonomials(N, M)>(out);
    out[0] += std::acos(T{-1}) / T{2};   // pi/2
}

template <typename T, int N, int M>
constexpr void seriesAtan(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    using std::atan;
    constexpr auto S = numMonomials(N, M);

    // h = 1 + a²
    std::array<T, S> h{};
    cauchyProduct<T, N, M>(h, a, a);
    h[0] += T{1};

    out = {};
    out[0] = atan(a[0]);
    const T inv_h0 = T{1} / h[0];

    if constexpr (M == 1) {
        for (int d = 1; d <= N; ++d) {
            T rhs = T{0};
            for (int k = 1; k < d; ++k)
                rhs += T(k) * h[d - k] * out[k];
            out[d] = (a[d] - rhs / T(d)) * inv_h0;
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
                            rhs += T(dg) * h[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = (a[ai] - rhs / T(d)) * inv_h0;
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

template <typename T, int N, int M>
constexpr void seriesAtan2(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& y,
    const std::array<T, numMonomials(N, M)>& x) noexcept
{
    using std::atan2;
    constexpr auto S = numMonomials(N, M);

    // h = x² + y²
    std::array<T, S> h{};
    cauchyProduct<T, N, M>(h, x, x);
    std::array<T, S> ysq{};
    cauchyProduct<T, N, M>(ysq, y, y);
    addInPlace<T, S>(h, ysq);

    out = {};
    out[0] = atan2(y[0], x[0]);
    const T inv_h0 = T{1} / h[0];

    if constexpr (M == 1) {
        for (int d = 1; d <= N; ++d) {
            T rhs = T(d) * (x[0] * y[d] - y[0] * x[d]);
            for (int k = 1; k < d; ++k)
                rhs += T(k) * (x[d - k] * y[k] - y[d - k] * x[k] - h[d - k] * out[k]);
            out[d] = rhs * inv_h0 / T(d);
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
                            const auto bi = flatIndex<M>(beta);
                            const auto gi = flatIndex<M>(gamma);
                            rhs += T(dg) * (x[bi]*y[gi] - y[bi]*x[gi] - h[bi]*out[gi]);
                            return;
                        }
                        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                            beta[bvar] = b;
                            bself(bself, bvar + 1, brem - b);
                        }
                    };
                    fillBeta(fillBeta, 0, db);
                }
                out[ai] = ((x[0]*y[ai] - y[0]*x[ai]) + rhs / T(d)) * inv_h0;
                return;
            }
            for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
        };

        for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
    }
}

} // namespace tax::detail
