#include <gtest/gtest.h>

#include <cmath>
#include <tax/tax.hpp>

using tax::ChebyshevExpansion;

// A modest non-trivial argument whose range stays comfortably inside the
// domains of the functions under test:  f(x) = 0.3 + 0.4 x  maps [-1,1] to
// [-0.1, 0.7].
template < int N >
static ChebyshevExpansion< N > arg()
{
    return 0.3 + 0.4 * ChebyshevExpansion< N >::variable();
}

// Check g(f) against the host function over the whole interval.
template < int N, typename Series, typename G >
static void expectComposition( const Series& gf, const ChebyshevExpansion< N >& f, G g, double tol )
{
    for ( double x = -1.0; x <= 1.0; x += 0.05 )
        EXPECT_NEAR( gf.eval( x ), g( f.eval( x ) ), tol ) << "at x=" << x;
}

TEST( ChebyshevMath, ExpLogSqrt )
{
    constexpr int N = 18;
    auto f = arg< N >();
    expectComposition( exp( f ), f, []( double v ) { return std::exp( v ); }, 1e-10 );
    // log/sqrt need a positive argument: shift to [0.9, 1.7].
    auto g = 1.3 + 0.4 * ChebyshevExpansion< N >::variable();
    expectComposition( log( g ), g, []( double v ) { return std::log( v ); }, 1e-10 );
    expectComposition( sqrt( g ), g, []( double v ) { return std::sqrt( v ); }, 1e-10 );
    expectComposition( cbrt( g ), g, []( double v ) { return std::cbrt( v ); }, 1e-10 );
}

TEST( ChebyshevMath, Trig )
{
    constexpr int N = 18;
    auto f = arg< N >();
    expectComposition( sin( f ), f, []( double v ) { return std::sin( v ); }, 1e-10 );
    expectComposition( cos( f ), f, []( double v ) { return std::cos( v ); }, 1e-10 );
    expectComposition( tan( f ), f, []( double v ) { return std::tan( v ); }, 1e-10 );
    expectComposition( atan( f ), f, []( double v ) { return std::atan( v ); }, 1e-10 );
    // asin/acos need |arg| < 1, satisfied by f in [-0.1, 0.7].
    expectComposition( asin( f ), f, []( double v ) { return std::asin( v ); }, 1e-10 );
    expectComposition( acos( f ), f, []( double v ) { return std::acos( v ); }, 1e-10 );
}

TEST( ChebyshevMath, Hyperbolic )
{
    constexpr int N = 18;
    auto f = arg< N >();
    expectComposition( sinh( f ), f, []( double v ) { return std::sinh( v ); }, 1e-10 );
    expectComposition( cosh( f ), f, []( double v ) { return std::cosh( v ); }, 1e-10 );
    expectComposition( tanh( f ), f, []( double v ) { return std::tanh( v ); }, 1e-10 );
    expectComposition( asinh( f ), f, []( double v ) { return std::asinh( v ); }, 1e-10 );
    expectComposition( atanh( f ), f, []( double v ) { return std::atanh( v ); }, 1e-10 );
    expectComposition( erf( f ), f, []( double v ) { return std::erf( v ); }, 1e-10 );
}

TEST( ChebyshevMath, ReciprocalAndDivision )
{
    constexpr int N = 18;
    auto f = arg< N >();                                       // [-0.1, 0.7]
    auto d = 1.5 + 0.3 * ChebyshevExpansion< N >::variable();  // [1.2, 1.8], no zero
    expectComposition( reciprocal( d ), d, []( double v ) { return 1.0 / v; }, 1e-10 );

    auto q = f / d;
    for ( double x = -1.0; x <= 1.0; x += 0.05 )
        EXPECT_NEAR( q.eval( x ), f.eval( x ) / d.eval( x ), 1e-10 ) << "at x=" << x;

    auto r = 2.0 / d;
    expectComposition( r, d, []( double v ) { return 2.0 / v; }, 1e-10 );
}

TEST( ChebyshevMath, PowReal )
{
    constexpr int N = 18;
    auto g = 1.3 + 0.4 * ChebyshevExpansion< N >::variable();  // positive
    auto p = pow( g, 2.5 );
    expectComposition( p, g, []( double v ) { return std::pow( v, 2.5 ); }, 1e-10 );
}

TEST( ChebyshevMath, SquareCubeAreExact )
{
    // square/cube use the exact truncated product, so they match the algebraic
    // identity to machine precision (and are constexpr).
    constexpr int N = 8;
    auto x = ChebyshevExpansion< N >::variable();
    auto f = 1.0 + 2.0 * x;
    auto s = square( f );
    auto c = cube( f );
    for ( double p : { -0.9, -0.2, 0.5, 0.8 } )
    {
        const double fv = f.eval( p );
        EXPECT_NEAR( s.eval( p ), fv * fv, 1e-12 );
        EXPECT_NEAR( c.eval( p ), fv * fv * fv, 1e-12 );
    }
}

TEST( ChebyshevMath, ComposeRoundTripExpLog )
{
    // log(exp(f)) ~ f over the interval.
    constexpr int N = 20;
    auto f = arg< N >();
    auto rt = log( exp( f ) );
    for ( double x = -1.0; x <= 1.0; x += 0.05 ) EXPECT_NEAR( rt.eval( x ), f.eval( x ), 1e-10 );
}

// ---------------------------------------------------------------------------
// Multivariate (M > 1) transcendentals + division via tensor-grid Gauss-Lobatto
// interpolation (Route B): a genuine spectral approximation over [-1,1]^M,
// projected to total degree <= N. The univariate tests above are the M == 1
// case of the same code path (which must stay bit-identical).
// ---------------------------------------------------------------------------

TEST( ChebyshevMath, MultivariateTranscendentalsAndDivision )
{
    constexpr int N = 10;
    using Cheb = ChebyshevExpansion< N, 2 >;
    const auto x = Cheb::variable< 0 >();
    const auto y = Cheb::variable< 1 >();
    const auto f = 0.3 * x + 0.2 * y;        // range [-0.5, 0.5] on [-1,1]^2
    const auto d = 1.5 + 0.3 * x + 0.2 * y;  // range [1.0, 2.0], no zero

    const auto ef = exp( f );
    const auto sf = sin( f );
    const auto lg = log( d );
    const auto q = f / d;
    const auto ri = 2.0 / d;
    const auto rc = reciprocal( d );

    for ( auto pt : { std::array< double, 2 >{ 0.4, -0.6 }, std::array< double, 2 >{ -0.8, 0.5 },
                      std::array< double, 2 >{ 0.1, 0.9 }, std::array< double, 2 >{ -0.3, -0.7 } } )
    {
        const double fv = f.eval( pt );
        const double dv = d.eval( pt );
        EXPECT_NEAR( ef.eval( pt ), std::exp( fv ), 1e-6 );
        EXPECT_NEAR( sf.eval( pt ), std::sin( fv ), 1e-6 );
        EXPECT_NEAR( lg.eval( pt ), std::log( dv ), 1e-6 );
        EXPECT_NEAR( q.eval( pt ), fv / dv, 1e-6 );
        EXPECT_NEAR( ri.eval( pt ), 2.0 / dv, 1e-6 );
        EXPECT_NEAR( rc.eval( pt ), 1.0 / dv, 1e-6 );
    }
}

TEST( ChebyshevMath, TrivariateExp )
{
    // M = 3 exercises the axis>=2 fibre sweep (stride (N+1)^2) of the separable DCT.
    constexpr int N = 8;
    using Cheb = ChebyshevExpansion< N, 3 >;
    const auto f =
        0.2 * Cheb::variable< 0 >() + 0.15 * Cheb::variable< 1 >() - 0.1 * Cheb::variable< 2 >();
    const auto ef = exp( f );
    for ( auto pt : { std::array< double, 3 >{ 0.5, -0.4, 0.8 },
                      std::array< double, 3 >{ -0.9, 0.6, -0.2 } } )
        EXPECT_NEAR( ef.eval( pt ), std::exp( f.eval( pt ) ), 1e-6 );
}
