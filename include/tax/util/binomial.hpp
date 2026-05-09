// SPDX-License-Identifier: BSD-3-Clause
//
// Binomial coefficients shared by static and dynamic paths.
//
// The same arithmetic is exposed in two flavours:
//   - constexpr `binom<unsigned long long>(n, k)` for compile-time use, and
//   - `binom(std::size_t, std::size_t)` returning std::size_t at runtime.
//
// Both reduce to the iterative `n choose k = product_{i=1..k} (n-k+i)/i`
// formulation, which is exact in integer arithmetic for the modest
// (n, k) we ever encounter (M, N <= a few hundred at most).

#pragma once

#include <cstddef>

namespace tax::util
{

[[nodiscard]] constexpr unsigned long long binomULL( unsigned long long n,
                                                     unsigned long long k ) noexcept
{
    if ( k > n )
    {
        return 0;
    }
    if ( k > n - k )
    {
        k = n - k;
    }
    unsigned long long acc = 1;
    for ( unsigned long long i = 1; i <= k; ++i )
    {
        acc = acc * ( n - k + i ) / i;
    }
    return acc;
}

[[nodiscard]] constexpr std::size_t binom( std::size_t n, std::size_t k ) noexcept
{
    return static_cast< std::size_t >(
        binomULL( static_cast< unsigned long long >( n ), static_cast< unsigned long long >( k ) ) );
}

}  // namespace tax::util
