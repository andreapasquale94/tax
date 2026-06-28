#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <tax/expansion/scheme/isotropic.hpp>
#include <tax/bases/chebyshev_basis.hpp>
#include <tax/bases/aliases.hpp>

namespace tax
{

// ===========================================================================
// Chebyshev interpolation of an arbitrary callable on the basis's interval.
//
// Builds the order-N polynomial interpolating f at the N+1 Chebyshev-Gauss-
// Lobatto nodes of [Lo, Hi] (Lo/Hi taken from the target basis), recovering the
// canonical-variable coefficients by a discrete cosine sum. Unlike a truncated
// Taylor series, the result is a near-best uniform approximation of f over the
// whole interval. Pass a domain-mapped basis to interpolate off [-1, 1]:
//   chebyshevInterpolate< N, ChebyshevBasisOn< 0.0, 3.0 > >( f );
// ===========================================================================

template < int N, typename Basis = ChebyshevBasis, typename T = double, typename F >
[[nodiscard]] Expansion< T, Basis, IsotropicScheme< N, 1 > > chebyshevInterpolate( F&& f )
{
    using std::cos;
    using Result = Expansion< T, Basis, IsotropicScheme< N, 1 > >;
    std::array< T, std::size_t( N ) + 1 > c{};

    constexpr T lo = T( Basis::domainLo );
    constexpr T hi = T( Basis::domainHi );
    // Canonical node u -> physical sample point in [lo, hi].
    const auto physical = [&]( T u ) -> T { return ( ( hi - lo ) * u + ( hi + lo ) ) / T( 2 ); };

    if constexpr ( N == 0 )
    {
        c[0] = T( f( physical( T{ 1 } ) ) );
        return Result{ c };
    } else
    {
        const T pi = std::numbers::pi_v< T >;

        std::array< T, std::size_t( N ) + 1 > fx{};
        for ( int j = 0; j <= N; ++j )
            fx[std::size_t( j )] = T( f( physical( cos( pi * T( j ) / T( N ) ) ) ) );

        // c_k = (2/N) w_k Σ_j pp_j f_j cos(pi j k / N),
        //   pp_j = 1/2 at the endpoints (else 1), w_k = 1/2 at k = 0, N (else 1).
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
        return Result{ c };
    }
}

}  // namespace tax
