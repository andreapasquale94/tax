#pragma once

#include <tax/utils/combinatorics.hpp>

namespace tax::detail {

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

} // namespace tax::detail
