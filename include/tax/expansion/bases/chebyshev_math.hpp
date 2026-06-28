#pragma once

#include <array>
#include <cmath>
#include <tax/expansion/bases/aliases.hpp>
#include <tax/expansion/bases/chebyshev_basis.hpp>
#include <tax/expansion/bases/chebyshev_interp.hpp>
#include <tax/expansion/bases/operators.hpp>
#include <tax/expansion/scheme/isotropic.hpp>
#include <type_traits>

namespace tax
{

// ===========================================================================
// Elementary math on a Chebyshev series (any number of variables M) —
// composition by interpolation.
//
// The Chebyshev basis has no triangular composition recurrence, so g(f) is
// formed the way a spectral library does: sample g(f(x)) at the Gauss-Lobatto
// nodes and re-interpolate, giving a near-best uniform approximation of g(f)
// over the basis's box. Runtime (not constexpr); the pure-algebraic square/cube
// stay exact via the truncated product.
//
// For M > 1 the sampling/re-interpolation run on the (N+1)^M tensor grid via a
// separable row-column DCT (see chebyshevInterpolate<N,M,...>): a genuine
// spectral approximation over [Lo,Hi]^M, projected to total degree <= N. Cost
// is exponential in M (the tensor grid is intrinsic to multivariate Chebyshev
// composition) — practical for modest M. M == 1 reduces to the plain 1-D DCT.
//
// Works for any domain-mapped Chebyshev basis (ChebyshevBasisOn<Lo,Hi>); the
// sampling and re-interpolation are carried out on that same box.
// ===========================================================================

/// A univariate Chebyshev-family basis (canonical or domain-mapped).
template < typename B >
concept ChebyshevLike = BasisPolicy< B > && requires {
    B::domainLo;
    B::domainHi;
};

namespace detail
{
template < typename T, typename Cheb, int N, int M, typename G >
[[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, M > > chebCompose(
    const Expansion< T, Cheb, IsotropicScheme< N, M > >& f, G&& g )
{
    return chebyshevInterpolate< N, M, Cheb, T >(
        [&]( const std::array< T, std::size_t( M ) >& p ) -> T { return T( g( f.eval( p ) ) ); } );
}
}  // namespace detail

#define TAX_CHEB_UNARY( NAME, EXPR )                                      \
    template < typename T, typename Cheb, int N, int M >                  \
        requires ChebyshevLike< Cheb >                                    \
    [[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, M > > NAME(     \
        const Expansion< T, Cheb, IsotropicScheme< N, M > >& f )          \
    {                                                                     \
        using std::NAME;                                                  \
        return detail::chebCompose( f, []( T v ) -> T { return EXPR; } ); \
    }

TAX_CHEB_UNARY( sqrt, sqrt( v ) )
TAX_CHEB_UNARY( cbrt, cbrt( v ) )
TAX_CHEB_UNARY( exp, exp( v ) )
TAX_CHEB_UNARY( log, log( v ) )
TAX_CHEB_UNARY( sin, sin( v ) )
TAX_CHEB_UNARY( cos, cos( v ) )
TAX_CHEB_UNARY( tan, tan( v ) )
TAX_CHEB_UNARY( asin, asin( v ) )
TAX_CHEB_UNARY( acos, acos( v ) )
TAX_CHEB_UNARY( atan, atan( v ) )
TAX_CHEB_UNARY( sinh, sinh( v ) )
TAX_CHEB_UNARY( cosh, cosh( v ) )
TAX_CHEB_UNARY( tanh, tanh( v ) )
TAX_CHEB_UNARY( asinh, asinh( v ) )
TAX_CHEB_UNARY( acosh, acosh( v ) )
TAX_CHEB_UNARY( atanh, atanh( v ) )
TAX_CHEB_UNARY( erf, erf( v ) )

#undef TAX_CHEB_UNARY

/// Exact square f^2 via the truncated Chebyshev product (constexpr, no sampling).
template < typename T, typename Cheb, int N, int M >
    requires ChebyshevLike< Cheb >
[[nodiscard]] constexpr Expansion< T, Cheb, IsotropicScheme< N, M > > square(
    const Expansion< T, Cheb, IsotropicScheme< N, M > >& f ) noexcept
{
    return f * f;
}

/// Exact cube f^3 via two truncated Chebyshev products (constexpr, no sampling).
template < typename T, typename Cheb, int N, int M >
    requires ChebyshevLike< Cheb >
[[nodiscard]] constexpr Expansion< T, Cheb, IsotropicScheme< N, M > > cube(
    const Expansion< T, Cheb, IsotropicScheme< N, M > >& f ) noexcept
{
    return f * f * f;
}

/// Reciprocal 1/f (requires f != 0 on the interval).
template < typename T, typename Cheb, int N, int M >
    requires ChebyshevLike< Cheb >
[[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, M > > reciprocal(
    const Expansion< T, Cheb, IsotropicScheme< N, M > >& f )
{
    return detail::chebCompose( f, []( T v ) -> T { return T{ 1 } / v; } );
}

/// Real power f^p (requires f > 0 on the interval for non-integer p).
template < typename T, typename Cheb, int N, int M >
    requires ChebyshevLike< Cheb >
[[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, M > > pow(
    const Expansion< T, Cheb, IsotropicScheme< N, M > >& f, std::type_identity_t< T > p )
{
    using std::pow;
    return detail::chebCompose( f, [p]( T v ) -> T { return pow( v, p ); } );
}

// ---------------------------------------------------------------------------
// Division (composed — no exact truncated form in the Chebyshev basis).
// ---------------------------------------------------------------------------

template < typename T, typename Cheb, int N, int M >
    requires ChebyshevLike< Cheb >
[[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, M > > operator/(
    const Expansion< T, Cheb, IsotropicScheme< N, M > >& a,
    const Expansion< T, Cheb, IsotropicScheme< N, M > >& b )
{
    return chebyshevInterpolate< N, M, Cheb, T >(
        [&]( const std::array< T, std::size_t( M ) >& p ) -> T {
            return a.eval( p ) / b.eval( p );
        } );
}

template < typename T, typename Cheb, int N, int M >
    requires ChebyshevLike< Cheb >
[[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, M > > operator/(
    std::type_identity_t< T > s, const Expansion< T, Cheb, IsotropicScheme< N, M > >& b )
{
    return detail::chebCompose( b, [s]( T v ) -> T { return s / v; } );
}

template < typename T, typename Cheb, int N, int M >
    requires ChebyshevLike< Cheb >
Expansion< T, Cheb, IsotropicScheme< N, M > >& operator/=(
    Expansion< T, Cheb, IsotropicScheme< N, M > >& a,
    const Expansion< T, Cheb, IsotropicScheme< N, M > >& b )
{
    a = a / b;
    return a;
}

}  // namespace tax
