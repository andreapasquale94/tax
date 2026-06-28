#pragma once

#include <cmath>
#include <tax/expansion/scheme/isotropic.hpp>
#include <tax/bases/chebyshev_basis.hpp>
#include <tax/bases/chebyshev_interp.hpp>
#include <tax/bases/operators.hpp>
#include <tax/bases/aliases.hpp>
#include <type_traits>

namespace tax
{

// ===========================================================================
// Elementary math on a (univariate) Chebyshev series — composition by
// interpolation.
//
// The Chebyshev basis has no triangular composition recurrence, so g(f) is
// formed the way a spectral library does: sample g(f(x)) at the Gauss-Lobatto
// nodes and re-interpolate, giving a near-best uniform approximation of g(f)
// over the basis's interval. Runtime (not constexpr); the pure-algebraic
// square/cube stay exact via the truncated product.
//
// Works for any domain-mapped Chebyshev basis (ChebyshevBasisOn<Lo,Hi>); the
// sampling and re-interpolation are carried out on that same interval.
// ===========================================================================

/// A univariate Chebyshev-family basis (canonical or domain-mapped).
template < typename B >
concept ChebyshevLike = BasisPolicy< B > && requires {
    B::domainLo;
    B::domainHi;
};

namespace detail
{
template < typename T, typename Cheb, int N, typename G >
[[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, 1 > > chebCompose(
    const Expansion< T, Cheb, IsotropicScheme< N, 1 > >& f, G&& g )
{
    return chebyshevInterpolate< N, Cheb, T >( [&]( T x ) -> T { return T( g( f.eval( x ) ) ); } );
}
}  // namespace detail

#define TAX_CHEB_UNARY( NAME, EXPR )                                      \
    template < typename T, typename Cheb, int N >                         \
        requires ChebyshevLike< Cheb >                                    \
    [[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, 1 > > NAME(     \
        const Expansion< T, Cheb, IsotropicScheme< N, 1 > >& f )          \
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
template < typename T, typename Cheb, int N >
    requires ChebyshevLike< Cheb >
[[nodiscard]] constexpr Expansion< T, Cheb, IsotropicScheme< N, 1 > > square(
    const Expansion< T, Cheb, IsotropicScheme< N, 1 > >& f ) noexcept
{
    return f * f;
}

/// Exact cube f^3 via two truncated Chebyshev products (constexpr, no sampling).
template < typename T, typename Cheb, int N >
    requires ChebyshevLike< Cheb >
[[nodiscard]] constexpr Expansion< T, Cheb, IsotropicScheme< N, 1 > > cube(
    const Expansion< T, Cheb, IsotropicScheme< N, 1 > >& f ) noexcept
{
    return f * f * f;
}

/// Reciprocal 1/f (requires f != 0 on the interval).
template < typename T, typename Cheb, int N >
    requires ChebyshevLike< Cheb >
[[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, 1 > > reciprocal(
    const Expansion< T, Cheb, IsotropicScheme< N, 1 > >& f )
{
    return detail::chebCompose( f, []( T v ) -> T { return T{ 1 } / v; } );
}

/// Real power f^p (requires f > 0 on the interval for non-integer p).
template < typename T, typename Cheb, int N >
    requires ChebyshevLike< Cheb >
[[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, 1 > > pow(
    const Expansion< T, Cheb, IsotropicScheme< N, 1 > >& f, std::type_identity_t< T > p )
{
    using std::pow;
    return detail::chebCompose( f, [p]( T v ) -> T { return pow( v, p ); } );
}

// ---------------------------------------------------------------------------
// Division (composed — no exact truncated form in the Chebyshev basis).
// ---------------------------------------------------------------------------

template < typename T, typename Cheb, int N >
    requires ChebyshevLike< Cheb >
[[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, 1 > > operator/(
    const Expansion< T, Cheb, IsotropicScheme< N, 1 > >& a,
    const Expansion< T, Cheb, IsotropicScheme< N, 1 > >& b )
{
    return chebyshevInterpolate< N, Cheb, T >(
        [&]( T x ) -> T { return a.eval( x ) / b.eval( x ); } );
}

template < typename T, typename Cheb, int N >
    requires ChebyshevLike< Cheb >
[[nodiscard]] Expansion< T, Cheb, IsotropicScheme< N, 1 > > operator/(
    std::type_identity_t< T > s, const Expansion< T, Cheb, IsotropicScheme< N, 1 > >& b )
{
    return detail::chebCompose( b, [s]( T v ) -> T { return s / v; } );
}

template < typename T, typename Cheb, int N >
    requires ChebyshevLike< Cheb >
Expansion< T, Cheb, IsotropicScheme< N, 1 > >& operator/=(
    Expansion< T, Cheb, IsotropicScheme< N, 1 > >& a,
    const Expansion< T, Cheb, IsotropicScheme< N, 1 > >& b )
{
    a = a / b;
    return a;
}

}  // namespace tax
