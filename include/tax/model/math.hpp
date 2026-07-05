#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <tax/model/arithmetic.hpp>
#include <tax/model/taylor_model.hpp>
#include <tax/operators/math_unary.hpp>

namespace tax::model
{

// ===========================================================================
// Intrinsic functions on Taylor models (§4.3.2 / §5.3.3)
//
// Every intrinsic follows the same two-step recipe. Split off the constant
// part, f = c + fbar, so fbar has zero constant part and Taylor model
// (P - c, I). With A = B(P - c) + I enclosing fbar over the domain and
// Theta = hull(0, A) enclosing theta * fbar (0 < theta < 1):
//
//   1. evaluate the degree-N Taylor polynomial of g around c at fbar with
//      Horner's scheme in Taylor-model arithmetic — its remainder interval
//      accumulates the propagated I^R_{N,poly} of (5.5) automatically;
//   2. add the Lagrange remainder enclosure
//      g^(N+1)(c + Theta) / (N+1)! * A^(N+1)
//      evaluated in outward-rounded interval arithmetic.
//
// Domain conditions (thesis §4.3.2) are checked on W = c + Theta, which
// encloses every point c + theta*fbar(x) the Lagrange form can sample; a
// violation throws std::domain_error.
//
// Deviation from the thesis: asin/acos/atan use the direct Taylor expansion
// around c (with the same derivative recursions the thesis uses for the
// remainder term) instead of the arcsin/arctan addition formulas. The
// addition formulas are sharper for large c but only valid under branch
// conditions the thesis leaves implicit; the direct form is unconditionally
// correct on the checked domain.
// ===========================================================================

namespace detail
{

/// Enclosure of 1/k!.
template < std::floating_point T >
[[nodiscard]] inline Interval< T > invFactorial( int k )
{
    Interval< T > r{ T{ 1 } };
    for ( int i = 2; i <= k; ++i ) r = r / T( i );
    return r;
}

/// Constant-part split of §4.3.2: f = c + fbar with the derived enclosures.
template < std::floating_point T, int N, int M >
struct Split
{
    T c;                          ///< constant part of the polynomial
    TaylorModel< T, N, M > fbar;  ///< (P - c, I)
    Interval< T > a;              ///< A = B(P - c) + I, encloses fbar over the domain
    Interval< T > theta;          ///< hull(0, A), encloses theta * fbar
    Interval< T > w;              ///< c + Theta, where the Lagrange derivative lives
};

template < std::floating_point T, int N, int M >
[[nodiscard]] Split< T, N, M > split( const TaylorModel< T, N, M >& f )
{
    Split< T, N, M > s{ .c = f.value(), .fbar = f, .a = {}, .theta = {}, .w = {} };
    s.fbar.polynomial()[0] = T{ 0 };
    s.a = s.fbar.bound();
    s.theta = hull( Interval< T >{}, s.a );
    s.w = Interval< T >{ s.c } + s.theta;
    return s;
}

/// Horner evaluation of sum_k a_k * fbar^k in Taylor-model arithmetic
/// (cf. (4.8) and the Step-1 polynomial (5.5)).
template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > hornerSeries( const TaylorModel< T, N, M >& fbar,
                                                   const std::array< T, std::size_t( N ) + 1 >& a )
{
    auto r = TaylorModel< T, N, M >::constant( a[std::size_t( N )], fbar.expansionPoint(),
                                               fbar.domain() );
    for ( int k = N - 1; k >= 0; --k ) r = r * fbar + a[std::size_t( k )];
    return r;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// square — dedicated so the remainder uses the sharp interval square (5.4)
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > square( const TaylorModel< T, N, M >& f )
{
    const detail::DomainPowers< T, M, 2 * N > pows{ f.displacementDomain() };
    const Interval< T > excess = detail::excessProductBound( f.polynomial(), f.polynomial(), pows );
    const Interval< T > bound_p = detail::polyRangeBound( f.polynomial(), pows );
    const Interval< T > rem = excess + T{ 2 } * bound_p * f.remainder() + sqr( f.remainder() );
    return { tax::square( f.polynomial() ), rem, f.expansionPoint(), f.domain() };
}

// ---------------------------------------------------------------------------
// exp — thesis (4.9)/(4.10)
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > exp( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );

    std::array< T, std::size_t( N ) + 1 > a{};
    a[0] = std::exp( s.c );
    for ( int k = 1; k <= N; ++k ) a[std::size_t( k )] = a[std::size_t( k - 1 )] / T( k );

    auto r = detail::hornerSeries( s.fbar, a );
    r.remainder() += exp( Interval< T >{ s.c } ) * detail::invFactorial< T >( N + 1 ) *
                     pow( s.a, N + 1 ) * exp( s.theta );
    return r;
}

// ---------------------------------------------------------------------------
// log — requires the enclosure to stay positive
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > log( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );
    if ( !( s.w.lower() > T{ 0 } ) )
        throw std::domain_error( "tax::model::log: enclosure of the argument is not positive" );

    const T u = T{ 1 } / s.c;
    std::array< T, std::size_t( N ) + 1 > a{};
    a[0] = std::log( s.c );
    T p = u;  // p_k = (-1)^(k+1) u^k
    for ( int k = 1; k <= N; ++k )
    {
        a[std::size_t( k )] = p / T( k );
        p *= -u;
    }

    auto r = detail::hornerSeries( s.fbar, a );

    const Interval< T > ui = Interval< T >{ T{ 1 } } / Interval< T >{ s.c };
    const Interval< T > v = T{ 1 } + s.theta * ui;
    Interval< T > lag = pow( s.a * ui, N + 1 ) / pow( v, N + 1 ) / T( N + 1 );
    if ( N % 2 != 0 ) lag = -lag;  // sign (-1)^(N+2)
    r.remainder() += lag;
    return r;
}

// ---------------------------------------------------------------------------
// reciprocal — thesis (4.11); requires the enclosure to avoid 0
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > reciprocal( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );
    if ( s.w.contains( T{ 0 } ) )
        throw std::domain_error( "tax::model::reciprocal: enclosure of the argument contains 0" );

    const T u = T{ 1 } / s.c;
    std::array< T, std::size_t( N ) + 1 > a{};
    a[0] = u;  // a_k = (-1)^k u^(k+1)
    for ( int k = 1; k <= N; ++k ) a[std::size_t( k )] = -a[std::size_t( k - 1 )] * u;

    auto r = detail::hornerSeries( s.fbar, a );

    const Interval< T > ui = Interval< T >{ T{ 1 } } / Interval< T >{ s.c };
    const Interval< T > v = T{ 1 } + s.theta * ui;  // = W / c, positive by the check above
    Interval< T > lag = pow( s.a * ui, N + 1 ) * ui / pow( v, N + 2 );
    if ( N % 2 == 0 ) lag = -lag;  // sign (-1)^(N+1)
    r.remainder() += lag;
    return r;
}

// ---------------------------------------------------------------------------
// sqrt and reciprocal square root — binomial series, positive enclosure
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > sqrt( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );
    if ( !( s.w.lower() > T{ 0 } ) )
        throw std::domain_error( "tax::model::sqrt: enclosure of the argument is not positive" );

    const T u = T{ 1 } / s.c;
    std::array< T, std::size_t( N ) + 1 > a{};
    a[0] = std::sqrt( s.c );
    if constexpr ( N >= 1 ) a[1] = a[0] * u / T{ 2 };
    for ( int k = 2; k <= N; ++k )
        a[std::size_t( k )] = a[std::size_t( k - 1 )] * ( -u ) * T( 2 * k - 3 ) / T( 2 * k );

    auto r = detail::hornerSeries( s.fbar, a );

    // Continue the coefficient recursion one step in interval arithmetic:
    // aI = (-1)^N sqrt(c) (2N-1)!! / ((N+1)! 2^(N+1)) * u^(N+1).
    const Interval< T > ui = Interval< T >{ T{ 1 } } / Interval< T >{ s.c };
    Interval< T > ai = sqrt( Interval< T >{ s.c } ) * ui / T{ 2 };
    for ( int k = 2; k <= N + 1; ++k ) ai = ai * ( -ui ) * T( 2 * k - 3 ) / T( 2 * k );

    const Interval< T > v = T{ 1 } + s.theta * ui;
    r.remainder() += ai * pow( s.a, N + 1 ) / ( pow( v, N ) * sqrt( v ) );
    return r;
}

/// Reciprocal square root 1/sqrt(f) (COSY's ISRT).
template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > isqrt( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );
    if ( !( s.w.lower() > T{ 0 } ) )
        throw std::domain_error( "tax::model::isqrt: enclosure of the argument is not positive" );

    const T u = T{ 1 } / s.c;
    std::array< T, std::size_t( N ) + 1 > a{};
    a[0] = T{ 1 } / std::sqrt( s.c );
    for ( int k = 1; k <= N; ++k )
        a[std::size_t( k )] = a[std::size_t( k - 1 )] * ( -u ) * T( 2 * k - 1 ) / T( 2 * k );

    auto r = detail::hornerSeries( s.fbar, a );

    // aI = (-1)^(N+1) (2N+1)!! / ((N+1)! 2^(N+1)) / (sqrt(c) c^(N+1)).
    const Interval< T > ui = Interval< T >{ T{ 1 } } / Interval< T >{ s.c };
    Interval< T > ai = Interval< T >{ T{ 1 } } / sqrt( Interval< T >{ s.c } );
    for ( int k = 1; k <= N + 1; ++k ) ai = ai * ( -ui ) * T( 2 * k - 1 ) / T( 2 * k );

    const Interval< T > v = T{ 1 } + s.theta * ui;
    r.remainder() += ai * pow( s.a, N + 1 ) / ( pow( v, N + 1 ) * sqrt( v ) );
    return r;
}

// ---------------------------------------------------------------------------
// Trigonometric — derivative cycle, |J| bounded over c + Theta
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > sin( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );
    const T sc = std::sin( s.c );
    const T cc = std::cos( s.c );
    const std::array< T, 4 > cycle{ sc, cc, -sc, -cc };  // d^k sin at c, times k!

    std::array< T, std::size_t( N ) + 1 > a{};
    T invf = T{ 1 };
    for ( int k = 0; k <= N; ++k )
    {
        if ( k > 0 ) invf /= T( k );
        a[std::size_t( k )] = cycle[std::size_t( k % 4 )] * invf;
    }

    auto r = detail::hornerSeries( s.fbar, a );

    // d^(N+1) sin over W = c + Theta.
    Interval< T > j;
    switch ( ( N + 1 ) % 4 )
    {
        case 0:
            j = sin( s.w );
            break;
        case 1:
            j = cos( s.w );
            break;
        case 2:
            j = -sin( s.w );
            break;
        default:
            j = -cos( s.w );
            break;
    }
    r.remainder() += pow( s.a, N + 1 ) * detail::invFactorial< T >( N + 1 ) * j;
    return r;
}

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > cos( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );
    const T sc = std::sin( s.c );
    const T cc = std::cos( s.c );
    const std::array< T, 4 > cycle{ cc, -sc, -cc, sc };  // d^k cos at c, times k!

    std::array< T, std::size_t( N ) + 1 > a{};
    T invf = T{ 1 };
    for ( int k = 0; k <= N; ++k )
    {
        if ( k > 0 ) invf /= T( k );
        a[std::size_t( k )] = cycle[std::size_t( k % 4 )] * invf;
    }

    auto r = detail::hornerSeries( s.fbar, a );

    // d^(N+1) cos over W = c + Theta.
    Interval< T > j;
    switch ( ( N + 1 ) % 4 )
    {
        case 0:
            j = cos( s.w );
            break;
        case 1:
            j = -sin( s.w );
            break;
        case 2:
            j = -cos( s.w );
            break;
        default:
            j = sin( s.w );
            break;
    }
    r.remainder() += pow( s.a, N + 1 ) * detail::invFactorial< T >( N + 1 ) * j;
    return r;
}

// ---------------------------------------------------------------------------
// Hyperbolic
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > sinh( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );
    const T sh = std::sinh( s.c );
    const T ch = std::cosh( s.c );

    std::array< T, std::size_t( N ) + 1 > a{};
    T invf = T{ 1 };
    for ( int k = 0; k <= N; ++k )
    {
        if ( k > 0 ) invf /= T( k );
        a[std::size_t( k )] = ( k % 2 == 0 ? sh : ch ) * invf;
    }

    auto r = detail::hornerSeries( s.fbar, a );
    const Interval< T > j = ( N + 1 ) % 2 == 0 ? sinh( s.w ) : cosh( s.w );
    r.remainder() += pow( s.a, N + 1 ) * detail::invFactorial< T >( N + 1 ) * j;
    return r;
}

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > cosh( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );
    const T sh = std::sinh( s.c );
    const T ch = std::cosh( s.c );

    std::array< T, std::size_t( N ) + 1 > a{};
    T invf = T{ 1 };
    for ( int k = 0; k <= N; ++k )
    {
        if ( k > 0 ) invf /= T( k );
        a[std::size_t( k )] = ( k % 2 == 0 ? ch : sh ) * invf;
    }

    auto r = detail::hornerSeries( s.fbar, a );
    const Interval< T > j = ( N + 1 ) % 2 == 0 ? cosh( s.w ) : sinh( s.w );
    r.remainder() += pow( s.a, N + 1 ) * detail::invFactorial< T >( N + 1 ) * j;
    return r;
}

// ---------------------------------------------------------------------------
// Inverse trigonometric
// ---------------------------------------------------------------------------

/// asin via the direct Taylor expansion at c; both the coefficients and the
/// Lagrange derivative bound use the thesis recursion
///   asin^(k+2)(a) = ((2k+1) a asin^(k+1)(a) + k^2 asin^(k)(a)) / (1 - a^2).
template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > asin( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );
    const Interval< T > om = T{ 1 } - sqr( s.w );
    if ( !( om.lower() > T{ 0 } ) )
        throw std::domain_error( "tax::model::asin: enclosure of the argument leaves (-1, 1)" );

    // Scalar derivative recursion at c for the polynomial coefficients.
    const T omc = T{ 1 } - s.c * s.c;
    std::array< T, std::size_t( N ) + 1 > a{};
    a[0] = std::asin( s.c );
    if constexpr ( N >= 1 )
    {
        T d_prev = T{ 1 } / std::sqrt( omc );      // asin'(c)
        T d_cur = s.c * d_prev * d_prev * d_prev;  // asin''(c)
        T fact = T{ 1 };
        a[1] = d_prev;
        if constexpr ( N >= 2 ) a[2] = d_cur / T{ 2 };
        for ( int k = 1; k + 2 <= N; ++k )
        {
            const T d_next = ( T( 2 * k + 1 ) * s.c * d_cur + T( k ) * T( k ) * d_prev ) / omc;
            d_prev = d_cur;
            d_cur = d_next;
            fact *= T( k + 2 );  // running (k+2)!/2 factor base
            a[std::size_t( k + 2 )] = d_next / ( T{ 2 } * fact );
        }
    }

    auto r = detail::hornerSeries( s.fbar, a );

    // Interval derivative recursion over W for the Lagrange bound.
    Interval< T > d_prev = Interval< T >{ T{ 1 } } / sqrt( om );  // asin' over W
    Interval< T > j = d_prev;
    if ( N + 1 >= 2 )
    {
        Interval< T > d_cur = s.w * pow( d_prev, 3 );  // asin'' over W
        for ( int k = 1; k + 2 <= N + 1; ++k )
        {
            const Interval< T > d_next =
                ( T( 2 * k + 1 ) * s.w * d_cur + T( k ) * T( k ) * d_prev ) / om;
            d_prev = d_cur;
            d_cur = d_next;
        }
        j = d_cur;
    }
    r.remainder() += pow( s.a, N + 1 ) * detail::invFactorial< T >( N + 1 ) * j;
    return r;
}

/// acos(f) = pi/2 - asin(f).
template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > acos( const TaylorModel< T, N, M >& f )
{
    return std::numbers::pi_v< T > / T{ 2 } - asin( f );
}

/// atan via the direct Taylor expansion at c; the closed form
///   atan^(k)(x) = (k-1)! cos^k(atan x) sin(k (atan x + pi/2))
/// gives the coefficients, and |atan^(N+1)| <= N! bounds the remainder.
template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > atan( const TaylorModel< T, N, M >& f )
{
    const auto s = detail::split( f );
    const T phi = std::atan( s.c );
    const T cphi = std::cos( phi );

    std::array< T, std::size_t( N ) + 1 > a{};
    a[0] = phi;
    T cpow = T{ 1 };
    for ( int k = 1; k <= N; ++k )
    {
        cpow *= cphi;
        a[std::size_t( k )] =
            cpow * std::sin( T( k ) * ( phi + std::numbers::pi_v< T > / T{ 2 } ) ) / T( k );
    }

    auto r = detail::hornerSeries( s.fbar, a );
    r.remainder() += pow( s.a, N + 1 ) / T( N + 1 ) * Interval< T >{ T{ -1 }, T{ 1 } };
    return r;
}

// ---------------------------------------------------------------------------
// Quotients and integer powers
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > tan( const TaylorModel< T, N, M >& f )
{
    return sin( f ) * reciprocal( cos( f ) );
}

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > tanh( const TaylorModel< T, N, M >& f )
{
    return sinh( f ) * reciprocal( cosh( f ) );
}

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > operator/( const TaylorModel< T, N, M >& a,
                                                const TaylorModel< T, N, M >& b )
{
    detail::checkCompatible( a, b );
    return a * reciprocal( b );
}

template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > operator/( std::type_identity_t< T > s,
                                                const TaylorModel< T, N, M >& a )
{
    return reciprocal( a ) * s;
}

template < std::floating_point T, int N, int M >
constexpr TaylorModel< T, N, M >& operator/=( TaylorModel< T, N, M >& a,
                                              const TaylorModel< T, N, M >& b )
{
    return a = a / b;
}

/// Integer power by binary exponentiation; negative exponents reciprocate.
template < std::floating_point T, int N, int M >
[[nodiscard]] TaylorModel< T, N, M > pow( const TaylorModel< T, N, M >& f, int n )
{
    if ( n < 0 ) return reciprocal( pow( f, -n ) );
    auto r = TaylorModel< T, N, M >::constant( T{ 1 }, f.expansionPoint(), f.domain() );
    if ( n == 0 ) return r;
    auto base = f;
    while ( true )
    {
        if ( n & 1 ) r = r * base;
        n >>= 1;
        if ( n == 0 ) break;
        base = square( base );
    }
    return r;
}

}  // namespace tax::model
