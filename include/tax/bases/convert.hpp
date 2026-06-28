#pragma once

#include <array>
#include <cstddef>
#include <tax/expansion/scheme/isotropic.hpp>
#include <tax/bases/aliases.hpp>

namespace tax
{

// ===========================================================================
// Exact basis conversion between the Taylor (monomial) and Chebyshev families
// (univariate). A degree-N polynomial has an exact image in either basis, so
// these are lossless for the kept modes — the bridge that lets a function built
// in one basis be moved into the other.
// ===========================================================================

/// Monomial coefficients -> Chebyshev coefficients (canonical [-1,1]).
template < int N, typename T = double >
[[nodiscard]] constexpr ChebyshevSeries< N, 1, T > toChebyshev(
    const TaylorSeries< N, 1, T >& f ) noexcept
{
    using Iso = IsotropicScheme< N, 1 >;
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
            ChebyshevBasis::template product< T, Iso >( next, xpow, xcheb );
            xpow = next;
        }
    }
    return ChebyshevSeries< N, 1, T >{ out };
}

/// Chebyshev coefficients -> monomial coefficients (canonical [-1,1]).
template < int N, typename T = double >
[[nodiscard]] constexpr TaylorSeries< N, 1, T > toTaylor(
    const ChebyshevSeries< N, 1, T >& f ) noexcept
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
            // next = 2 x cur - prev   (x* == shift coefficients up by one)
            for ( int i = N; i >= 1; --i )
                next[std::size_t( i )] = T{ 2 } * cur[std::size_t( i - 1 )];
            for ( std::size_t i = 0; i < next.size(); ++i ) next[i] -= prev[i];
            for ( std::size_t i = 0; i < out.size(); ++i ) out[i] += f[std::size_t( k )] * next[i];
            prev = cur;
            cur = next;
        }
    }
    return TaylorSeries< N, 1, T >{ out };
}

}  // namespace tax
