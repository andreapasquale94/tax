#pragma once

#include <tax/fwd.hpp>

namespace da::detail {

constexpr std::size_t binom(int n, int k) noexcept {
    if (k < 0 || n < 0 || k > n) return 0;
    if (k == 0 || k == n)        return 1;
    if (k > n - k) k = n - k;
    std::size_t r = 1;
    for (int i = 0; i < k; ++i) { r *= std::size_t(n - i); r /= std::size_t(i + 1); }
    return r;
}

constexpr std::size_t numMonomials(int N, int M) noexcept { return binom(N + M, M); }

template <int M>
constexpr int totalDegree(const da::MultiIndex<M>& a) noexcept {
    int d = 0;
    for (int i = 0; i < M; ++i) d += a[i];
    return d;
}

template <int M>
constexpr std::size_t flatIndex(const da::MultiIndex<M>& alpha) noexcept {
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

} // namespace da::detail
