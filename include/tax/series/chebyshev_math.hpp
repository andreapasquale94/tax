#pragma once

#include <cmath>
#include <tax/series/chebyshev_basis.hpp>
#include <tax/series/chebyshev_interp.hpp>
#include <tax/series/operators.hpp>
#include <tax/series/series.hpp>
#include <type_traits>

namespace tax
{

// ===========================================================================
// Elementary math on a Chebyshev series — composition by interpolation
// ===========================================================================
//
// The Taylor basis composes g(f) through the classical ODE recurrences. The
// Chebyshev basis has no such triangular recurrence, so we compose the way a
// spectral library (Chebfun) does: sample the polynomial g(f(x)) at the
// Chebyshev-Gauss-Lobatto nodes and re-interpolate. For analytic g and f this
// is spectrally accurate and yields a *near-best uniform* approximation of g(f)
// over [-1, 1] — not a single-point jet re-expressed in the Chebyshev basis.
//
// These are runtime (not constexpr): they sample the host math functions and
// run a discrete cosine sum. Pure-algebraic operations (square, cube, the
// bilinear product) stay exact and constexpr via operators.hpp.
//
// Domain notes (values fed to the host function are f(x) for x in [-1, 1]):
//   sqrt/log: require f(x) > 0          asin/acos/atanh: require |f(x)| < 1
//   acosh:    require f(x) > 1          reciprocal/division: require f(x) != 0
// A violation propagates the host function's nan/inf into the samples.
// ===========================================================================

namespace detail
{

/// Order-N Chebyshev interpolant of  x |-> g( f(x) ).
template < int N, typename T, typename G >
[[nodiscard]] ChebyshevSeries< N, T > chebCompose( const ChebyshevSeries< N, T >& f, G&& g )
{
    return chebyshevInterpolate< N, T >( [&]( T x ) -> T { return T( g( f.eval( x ) ) ); } );
}

}  // namespace detail

#define TAX_CHEB_UNARY( NAME, EXPR )                                               \
    template < int N, typename T >                                                 \
    [[nodiscard]] ChebyshevSeries< N, T > NAME( const ChebyshevSeries< N, T >& f ) \
    {                                                                              \
        using std::NAME;                                                           \
        return detail::chebCompose( f, []( T v ) -> T { return EXPR; } );          \
    }

// Roots and reciprocal.
TAX_CHEB_UNARY( sqrt, sqrt( v ) )
TAX_CHEB_UNARY( cbrt, cbrt( v ) )

// Exponential / logarithm family.
TAX_CHEB_UNARY( exp, exp( v ) )
TAX_CHEB_UNARY( log, log( v ) )

// Trigonometric and inverse-trigonometric.
TAX_CHEB_UNARY( sin, sin( v ) )
TAX_CHEB_UNARY( cos, cos( v ) )
TAX_CHEB_UNARY( tan, tan( v ) )
TAX_CHEB_UNARY( asin, asin( v ) )
TAX_CHEB_UNARY( acos, acos( v ) )
TAX_CHEB_UNARY( atan, atan( v ) )

// Hyperbolic and inverse-hyperbolic.
TAX_CHEB_UNARY( sinh, sinh( v ) )
TAX_CHEB_UNARY( cosh, cosh( v ) )
TAX_CHEB_UNARY( tanh, tanh( v ) )
TAX_CHEB_UNARY( asinh, asinh( v ) )
TAX_CHEB_UNARY( acosh, acosh( v ) )
TAX_CHEB_UNARY( atanh, atanh( v ) )

// Error function.
TAX_CHEB_UNARY( erf, erf( v ) )

#undef TAX_CHEB_UNARY

/// Exact square f^2 via the truncated Chebyshev product (constexpr, no sampling).
template < int N, typename T >
[[nodiscard]] constexpr ChebyshevSeries< N, T > square( const ChebyshevSeries< N, T >& f ) noexcept
{
    return f * f;
}

/// Exact cube f^3 via two truncated Chebyshev products (constexpr, no sampling).
template < int N, typename T >
[[nodiscard]] constexpr ChebyshevSeries< N, T > cube( const ChebyshevSeries< N, T >& f ) noexcept
{
    return f * f * f;
}

/// Reciprocal 1/f (requires f(x) != 0 on [-1, 1]).
template < int N, typename T >
[[nodiscard]] ChebyshevSeries< N, T > reciprocal( const ChebyshevSeries< N, T >& f )
{
    return detail::chebCompose( f, []( T v ) -> T { return T{ 1 } / v; } );
}

/// Real power f^p (requires f(x) > 0 on [-1, 1] for non-integer p).
template < int N, typename T >
[[nodiscard]] ChebyshevSeries< N, T > pow( const ChebyshevSeries< N, T >& f,
                                           std::type_identity_t< T > p )
{
    using std::pow;
    return detail::chebCompose( f, [p]( T v ) -> T { return pow( v, p ); } );
}

// ---------------------------------------------------------------------------
// Division (no exact truncated form in the Chebyshev basis — composed).
// ---------------------------------------------------------------------------

/// Series / Series:  (a/b)(x) interpolated from a.eval(x)/b.eval(x).
template < int N, typename T >
[[nodiscard]] ChebyshevSeries< N, T > operator/( const ChebyshevSeries< N, T >& a,
                                                 const ChebyshevSeries< N, T >& b )
{
    return chebyshevInterpolate< N, T >( [&]( T x ) -> T { return a.eval( x ) / b.eval( x ); } );
}

/// scalar / Series.
template < int N, typename T >
[[nodiscard]] ChebyshevSeries< N, T > operator/( std::type_identity_t< T > s,
                                                 const ChebyshevSeries< N, T >& b )
{
    return detail::chebCompose( b, [s]( T v ) -> T { return s / v; } );
}

/// Series /= Series.
template < int N, typename T >
ChebyshevSeries< N, T >& operator/=( ChebyshevSeries< N, T >& a, const ChebyshevSeries< N, T >& b )
{
    a = a / b;
    return a;
}

}  // namespace tax
