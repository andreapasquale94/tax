#pragma once

#include <tax/fwd.hpp>

namespace tax::detail {

/**
 * @brief Compute the binomial coefficient `n choose k`.
 * @return `0` when arguments are out of range.
 */
constexpr std::size_t binom(int n, int k) noexcept {
    if (k < 0 || n < 0 || k > n) return 0;
    if (k == 0 || k == n)        return 1;
    if (k > n - k) k = n - k;
    std::size_t r = 1;
    for (int i = 0; i < k; ++i) { r *= std::size_t(n - i); r /= std::size_t(i + 1); }
    return r;
}

/// @brief Number of monomials with total degree `<= N` in `M` variables.
constexpr std::size_t numMonomials(int N, int M) noexcept { return binom(N + M, M); }

template <int M>
/// @brief Total degree `|a| = sum_i a[i]` of a multi-index.
constexpr int totalDegree(const tax::MultiIndex<M>& a) noexcept {
    int d = 0;
    for (int i = 0; i < M; ++i) d += a[i];
    return d;
}

template <int M>
/**
 * @brief Map a multi-index to the internal flat storage index.
 * @details The ordering matches the graded-lex convention used by the kernels.
 */
constexpr std::size_t flatIndex(const tax::MultiIndex<M>& alpha) noexcept {
    static_assert(M >= 1);
    const int d = totalDegree<M>(alpha);
    std::size_t idx = binom(d + M - 1, M);
    int rem = d;
    for (int i = 0; i < M - 1; ++i) {
        idx += binom(rem - alpha[i] + (M - 2 - i), M - 1 - i);
        rem  -= alpha[i];
    }
    return idx;
}

} // namespace tax::detail
