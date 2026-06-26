#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <tax/series/basis.hpp>

namespace tax
{

// ===========================================================================
// ChebyshevBasis — Chebyshev polynomials of the first kind  P_k(x) = T_k(x)
// ===========================================================================
//
// Stores  f = sum_{k=0}^{N} c_k T_k(x)  with the *plain* (un-normalised)
// convention — no factor of 1/2 on the constant term. Every coefficient-space
// routine below is derived for that convention and unit-tested against closed
// forms.
//
// Identities used:
//   T_0 = 1, T_1 = x, T_{k+1} = 2x T_k - T_{k-1}            (recurrence)
//   T_i T_j = (T_{i+j} + T_{|i-j|}) / 2                     (product)
//   evaluation by Clenshaw recurrence
// ===========================================================================

struct ChebyshevBasis
{
    static constexpr bool is_tax_basis = true;

    [[nodiscard]] static constexpr std::string_view name() noexcept { return "chebyshev"; }

    [[nodiscard]] static std::string term( int k )
    {
        if ( k == 0 ) return "1";
        return "T_" + std::to_string( k );
    }

    /// Truncated Chebyshev product via  T_i T_j = (T_{i+j} + T_{|i-j|})/2.
    /// Modes with i+j > N fold only their |i-j| part back into range; the
    /// out-of-range sum part is dropped (the inherent truncation).
    template < typename T, int N >
    static constexpr void product( std::array< T, std::size_t( N ) + 1 >& out,
                                   const std::array< T, std::size_t( N ) + 1 >& a,
                                   const std::array< T, std::size_t( N ) + 1 >& b ) noexcept
    {
        out = {};
        for ( int i = 0; i <= N; ++i )
        {
            if ( a[std::size_t( i )] == T{ 0 } ) continue;
            for ( int j = 0; j <= N; ++j )
            {
                const T p = T( 0.5 ) * a[std::size_t( i )] * b[std::size_t( j )];
                if ( p == T{ 0 } ) continue;
                const int s = i + j;
                if ( s <= N ) out[std::size_t( s )] += p;
                out[std::size_t( i < j ? j - i : i - j )] += p;
            }
        }
    }

    /// Clenshaw evaluation of  f(x) = sum_k c_k T_k(x).
    template < typename T, int N >
    [[nodiscard]] static constexpr T eval( const std::array< T, std::size_t( N ) + 1 >& c,
                                           T x ) noexcept
    {
        T b1 = T{ 0 };
        T b2 = T{ 0 };
        const T two_x = T( 2 ) * x;
        for ( int k = N; k >= 1; --k )
        {
            const T bk = c[std::size_t( k )] + two_x * b1 - b2;
            b2 = b1;
            b1 = bk;
        }
        return c[0] + x * b1 - b2;
    }

    /// Coefficient-space derivative (Chebyshev "chder" recurrence, plain
    /// convention): given c (degree N) produce the degree-(N-1) coefficients of
    /// f'(x), stored back into an order-N array with the top term zero.
    template < typename T, int N >
    static constexpr void derivative( std::array< T, std::size_t( N ) + 1 >& out,
                                      const std::array< T, std::size_t( N ) + 1 >& c ) noexcept
    {
        out = {};
        if constexpr ( N >= 1 )
        {
            // out[k] = (k+2 <= N ? out[k+2] : 0) + 2(k+1) c[k+1], walked downwards.
            for ( int k = N - 1; k >= 0; --k )
            {
                T v = T( 2 * ( k + 1 ) ) * c[std::size_t( k + 1 )];
                if ( k + 2 <= N ) v += out[std::size_t( k + 2 )];
                out[std::size_t( k )] = v;
            }
            out[0] *= T( 0.5 );
        }
    }

    /// Coefficient-space indefinite integral (constant of integration 0), plain
    /// convention. Inverse of `derivative` up to the integration constant:
    ///   B_1 = c_0 - c_2/2,   B_k = (c_{k-1} - c_{k+1}) / (2k)  for k >= 2.
    template < typename T, int N >
    static constexpr void integral( std::array< T, std::size_t( N ) + 1 >& out,
                                    const std::array< T, std::size_t( N ) + 1 >& c ) noexcept
    {
        out = {};
        if constexpr ( N >= 1 )
        {
            out[1] = c[0];
            if constexpr ( N >= 2 ) out[1] -= T( 0.5 ) * c[2];
            for ( int k = 2; k <= N; ++k )
            {
                T v = c[std::size_t( k - 1 )];
                if ( k + 1 <= N ) v -= c[std::size_t( k + 1 )];
                out[std::size_t( k )] = v / T( 2 * k );
            }
        }
    }
};

static_assert( Basis< ChebyshevBasis > );

}  // namespace tax
