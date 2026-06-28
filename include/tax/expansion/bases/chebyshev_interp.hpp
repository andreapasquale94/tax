#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <tax/expansion/bases/aliases.hpp>
#include <tax/expansion/bases/chebyshev_basis.hpp>
#include <tax/expansion/scheme/isotropic.hpp>

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

namespace detail
{
/// Compile-time integer power `base^exp` (used for the (N+1)^M tensor-grid size).
[[nodiscard]] constexpr std::size_t ipow( std::size_t base, int exp ) noexcept
{
    std::size_t r = 1;
    for ( int i = 0; i < exp; ++i ) r *= base;
    return r;
}
}  // namespace detail

// ===========================================================================
// Multivariate (M >= 1) Chebyshev interpolation of a callable on the box.
//
// Samples f on the (N+1)^M tensor grid of Chebyshev-Gauss-Lobatto nodes, then
// recovers the tensor-product Chebyshev coefficients by a SEPARABLE (row-column)
// 1-D analysis DCT applied along each axis, and projects onto the total-degree-
// <= N IsotropicScheme. The result is the genuine spectral (near-best uniform)
// approximation of f over [Lo, Hi]^M. Cost is M*(N+1)^(M+1) and the grid buffer
// is (N+1)^M wide — exponential in M (intrinsic to multivariate Chebyshev
// composition; there is no triangular recurrence as in the monomial basis).
//
//   f : std::array<T, M> (physical point) -> T
// ===========================================================================
template < int N, int M, typename Basis = ChebyshevBasis, typename T = double, typename F >
[[nodiscard]] Expansion< T, Basis, IsotropicScheme< N, M > > chebyshevInterpolate( F&& f )
{
    using std::cos;
    using Scheme = IsotropicScheme< N, M >;
    using Result = Expansion< T, Basis, Scheme >;

    constexpr std::size_t L = std::size_t( N ) + 1;     // nodes per axis
    constexpr std::size_t Full = detail::ipow( L, M );  // (N+1)^M tensor grid
    constexpr T lo = T( Basis::domainLo );
    constexpr T hi = T( Basis::domainHi );
    const auto physical = [&]( T u ) -> T { return ( ( hi - lo ) * u + ( hi + lo ) ) / T( 2 ); };
    const T pi = std::numbers::pi_v< T >;

    // Per-axis physical node coordinates: u_j = cos(pi j / N) mapped to [lo, hi].
    std::array< T, L > node{};
    for ( std::size_t j = 0; j < L; ++j )
        node[j] = physical( ( N == 0 ) ? T{ 1 } : cos( pi * T( j ) / T( N ) ) );

    // Sample f on the full tensor grid (flat index = sum_i j_i * L^i).
    std::array< T, Full > g{};
    for ( std::size_t idx = 0; idx < Full; ++idx )
    {
        std::array< T, std::size_t( M ) > pt{};
        std::size_t t = idx;
        for ( int i = 0; i < M; ++i )
        {
            pt[std::size_t( i )] = node[t % L];
            t /= L;
        }
        g[idx] = T( f( pt ) );
    }

    // Separable analysis: apply the 1-D DCT in place along each axis. After all
    // M sweeps, g holds the tensor-product Chebyshev coefficients C_alpha.
    if constexpr ( N >= 1 )
    {
        for ( int axis = 0; axis < M; ++axis )
        {
            const std::size_t stride = detail::ipow( L, axis );
            for ( std::size_t base = 0; base < Full; ++base )
            {
                if ( ( base / stride ) % L != 0 ) continue;  // one origin per fibre
                std::array< T, L > in{};
                for ( std::size_t m = 0; m < L; ++m ) in[m] = g[base + m * stride];
                for ( std::size_t k = 0; k < L; ++k )
                {
                    T sum = T{ 0 };
                    for ( std::size_t j = 0; j < L; ++j )
                    {
                        const T pp = ( j == 0 || j == std::size_t( N ) ) ? T( 0.5 ) : T{ 1 };
                        sum += pp * in[j] * cos( pi * T( j ) * T( k ) / T( N ) );
                    }
                    const T wk = ( k == 0 || k == std::size_t( N ) ) ? T( 0.5 ) : T{ 1 };
                    g[base + k * stride] = T( 2 ) / T( N ) * wk * sum;
                }
            }
        }
    }

    // Project the tensor coefficients onto total-degree <= N (drop |alpha| > N).
    std::array< T, Scheme::nCoeff > c{};
    for ( std::size_t k = 0; k < Scheme::nCoeff; ++k )
    {
        const auto a = Scheme::multiOf( k );
        std::size_t fidx = 0;
        std::size_t mul = 1;
        for ( int i = 0; i < M; ++i )
        {
            fidx += std::size_t( a[std::size_t( i )] ) * mul;
            mul *= L;
        }
        c[k] = g[fidx];
    }
    return Result{ c };
}

}  // namespace tax
