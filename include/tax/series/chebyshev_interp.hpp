#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <tax/series/series.hpp>

namespace tax
{

// ===========================================================================
// Chebyshev interpolation of an arbitrary callable on [-1, 1].
//
// This is the operation that has no Taylor analogue and is the whole point of
// a Chebyshev representation: build the order-N polynomial that interpolates a
// black-box function f at the N+1 Chebyshev-Gauss-Lobatto nodes
//   x_j = cos(pi j / N),   j = 0..N
// recovering the (plain-convention) coefficients by a discrete cosine sum.
//
// Unlike a truncated Taylor series, the result is a near-best uniform
// approximation of f over the whole interval, not just near a single point.
// ===========================================================================

template < int N, typename T = double, typename F >
[[nodiscard]] ChebyshevSeries< N, T > chebyshevInterpolate( F&& f )
{
    using std::cos;
    std::array< T, std::size_t( N ) + 1 > c{};

    if constexpr ( N == 0 )
    {
        c[0] = T( f( T{ 1 } ) );
        return ChebyshevSeries< N, T >{ c };
    } else
    {
        const T pi = std::numbers::pi_v< T >;

        // Sample f at the Gauss-Lobatto nodes.
        std::array< T, std::size_t( N ) + 1 > fx{};
        for ( int j = 0; j <= N; ++j ) fx[std::size_t( j )] = T( f( cos( pi * T( j ) / T( N ) ) ) );

        // c_k = (2/N) w_k sum_j pp_j f_j cos(pi j k / N),
        //   pp_j = 1/2 at the two endpoints (else 1), w_k = 1/2 at k = 0, N (else 1).
        for ( int k = 0; k <= N; ++k )
        {
            T sum = T{ 0 };
            for ( int j = 0; j <= N; ++j )
            {
                const T pp = ( j == 0 || j == N ) ? T( 0.5 ) : T{ 1 };
                sum += pp * fx[std::size_t( j )] * cos( pi * T( j ) * T( k ) / T( N ) );
            }
            const T wk = ( k == 0 || k == N ) ? T( 0.5 ) : T{ 1 };
            c[std::size_t( k )] = T( 2 ) / T( N ) * wk * sum;
        }
        return ChebyshevSeries< N, T >{ c };
    }
}

}  // namespace tax
