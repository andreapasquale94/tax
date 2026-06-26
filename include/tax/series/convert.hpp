#pragma once

#include <array>
#include <cstddef>
#include <tax/series/series.hpp>

namespace tax
{

// ===========================================================================
// Exact basis conversion between the Taylor (monomial) and Chebyshev families.
//
// A degree-N polynomial has an exact representation in either basis, so these
// conversions are lossless for the kept modes. They are the bridge that lets a
// function built in one basis (e.g. exp via the Taylor recurrences) be moved
// into the other.
// ===========================================================================

/// Monomial coefficients -> Chebyshev coefficients.
/// Builds the Chebyshev image of each power x^k by repeatedly multiplying by
/// x == T_1 with the Chebyshev product, then accumulates.
template < int N, typename T >
[[nodiscard]] constexpr ChebyshevSeries< N, T > toChebyshev(
    const TaylorSeries< N, T >& f ) noexcept
{
    using Arr = std::array< T, std::size_t( N ) + 1 >;
    Arr out{};
    Arr xpow{};
    xpow[0] = T{ 1 };  // x^0 = T_0
    Arr xcheb{};
    if constexpr ( N >= 1 ) xcheb[1] = T{ 1 };  // x = T_1

    for ( int k = 0; k <= N; ++k )
    {
        const T ck = f[std::size_t( k )];
        if ( ck != T{ 0 } )
            for ( std::size_t i = 0; i < out.size(); ++i ) out[i] += ck * xpow[i];
        if ( k < N )
        {
            Arr next{};
            ChebyshevBasis::template product< T, N >( next, xpow, xcheb );
            xpow = next;
        }
    }
    return ChebyshevSeries< N, T >{ out };
}

/// Chebyshev coefficients -> monomial coefficients.
/// Generates the monomial image of each T_k via the three-term recurrence
/// T_{k+1} = 2x T_k - T_{k-1}, then accumulates.
template < int N, typename T >
[[nodiscard]] constexpr TaylorSeries< N, T > toTaylor( const ChebyshevSeries< N, T >& f ) noexcept
{
    using Arr = std::array< T, std::size_t( N ) + 1 >;
    Arr out{};
    Arr prev{};  // T_0 = 1
    prev[0] = T{ 1 };
    for ( std::size_t i = 0; i < out.size(); ++i ) out[i] += f[0] * prev[i];
    if constexpr ( N >= 1 )
    {
        Arr cur{};  // T_1 = x
        cur[1] = T{ 1 };
        for ( std::size_t i = 0; i < out.size(); ++i ) out[i] += f[1] * cur[i];

        for ( int k = 2; k <= N; ++k )
        {
            Arr next{};
            // next = 2 * x * cur - prev   (x* == shift coefficients up by one)
            for ( int i = N; i >= 1; --i )
                next[std::size_t( i )] = T{ 2 } * cur[std::size_t( i - 1 )];
            for ( std::size_t i = 0; i < next.size(); ++i ) next[i] -= prev[i];
            for ( std::size_t i = 0; i < out.size(); ++i ) out[i] += f[std::size_t( k )] * next[i];
            prev = cur;
            cur = next;
        }
    }
    return TaylorSeries< N, T >{ out };
}

}  // namespace tax
