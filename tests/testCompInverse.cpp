#include "testUtils.hpp"

// =============================================================================
// Compositional inverse (series reversion): f(g(y)) = y
// =============================================================================

TEST( CompInverse, Identity )
{
    // f(x) = x  ->  g(y) = y
    DA< 6 > f;
    f[1] = 1.0;
    DA< 6 > g = inv( f );
    EXPECT_NEAR( g[0], 0.0, kTol );
    EXPECT_NEAR( g[1], 1.0, kTol );
    for ( std::size_t k = 2; k < DA< 6 >::ncoef; ++k )
        EXPECT_NEAR( g[k], 0.0, kTol ) << "  coeff k=" << k;
}

TEST( CompInverse, LinearScaling )
{
    // f(x) = 3x  ->  g(y) = y/3
    DA< 6 > f;
    f[1] = 3.0;
    DA< 6 > g = inv( f );
    EXPECT_NEAR( g[0], 0.0, kTol );
    EXPECT_NEAR( g[1], 1.0 / 3.0, kTol );
    for ( std::size_t k = 2; k < DA< 6 >::ncoef; ++k )
        EXPECT_NEAR( g[k], 0.0, kTol ) << "  coeff k=" << k;
}

TEST( CompInverse, QuadraticCatalan )
{
    // f(x) = x + x^2  ->  g(y) = y - y^2 + 2y^3 - 5y^4 + 14y^5 - 42y^6
    // (Catalan numbers with alternating signs)
    DA< 6 > f;
    f[1] = 1.0;
    f[2] = 1.0;
    DA< 6 > g = inv( f );
    EXPECT_NEAR( g[0], 0.0, kTol );
    EXPECT_NEAR( g[1], 1.0, kTol );
    EXPECT_NEAR( g[2], -1.0, kTol );
    EXPECT_NEAR( g[3], 2.0, kTol );
    EXPECT_NEAR( g[4], -5.0, kTol );
    EXPECT_NEAR( g[5], 14.0, kTol );
    EXPECT_NEAR( g[6], -42.0, kTol );
}

TEST( CompInverse, SinReversion )
{
    // f(x) = sin(x) = x - x^3/6 + x^5/120 - ...
    // g(y) = arcsin(y) = y + y^3/6 + 3y^5/40 + ...
    auto x = DA< 7 >::variable( 0.0 );
    DA< 7 > f = sin( x );
    DA< 7 > g = inv( f );

    // arcsin coefficients (odd terms only, even are 0):
    // [1] = 1
    // [3] = 1/6
    // [5] = 3/40
    // [7] = 15/336
    EXPECT_NEAR( g[0], 0.0, kTol );
    EXPECT_NEAR( g[1], 1.0, kTol );
    EXPECT_NEAR( g[2], 0.0, kTol );
    EXPECT_NEAR( g[3], 1.0 / 6.0, kTol );
    EXPECT_NEAR( g[4], 0.0, kTol );
    EXPECT_NEAR( g[5], 3.0 / 40.0, kTol );
    EXPECT_NEAR( g[6], 0.0, kTol );
    EXPECT_NEAR( g[7], 15.0 / 336.0, kTol );
}

TEST( CompInverse, TanReversion )
{
    // f(x) = tan(x) = x + x^3/3 + 2x^5/15 + ...
    // g(y) = arctan(y) = y - y^3/3 + y^5/5 - ...
    auto x = DA< 7 >::variable( 0.0 );
    DA< 7 > f = tan( x );
    DA< 7 > g = inv( f );

    EXPECT_NEAR( g[0], 0.0, kTol );
    EXPECT_NEAR( g[1], 1.0, kTol );
    EXPECT_NEAR( g[2], 0.0, kTol );
    EXPECT_NEAR( g[3], -1.0 / 3.0, kTol );
    EXPECT_NEAR( g[4], 0.0, kTol );
    EXPECT_NEAR( g[5], 1.0 / 5.0, kTol );
    EXPECT_NEAR( g[6], 0.0, kTol );
    EXPECT_NEAR( g[7], -1.0 / 7.0, kTol );
}

TEST( CompInverse, ExpMinusOneReversion )
{
    // f(x) = exp(x) - 1 = x + x^2/2 + x^3/6 + ...
    // g(y) = log(1+y) = y - y^2/2 + y^3/3 - y^4/4 + ...
    auto x = DA< 6 >::variable( 0.0 );
    DA< 6 > f = exp( x ) - 1.0;
    DA< 6 > g = inv( f );

    EXPECT_NEAR( g[0], 0.0, kTol );
    EXPECT_NEAR( g[1], 1.0, kTol );
    EXPECT_NEAR( g[2], -1.0 / 2.0, kTol );
    EXPECT_NEAR( g[3], 1.0 / 3.0, kTol );
    EXPECT_NEAR( g[4], -1.0 / 4.0, kTol );
    EXPECT_NEAR( g[5], 1.0 / 5.0, kTol );
    EXPECT_NEAR( g[6], -1.0 / 6.0, kTol );
}

TEST( CompInverse, RoundTrip )
{
    // Verify f(g(y)) ≈ y by evaluating numerically
    auto x = DA< 8 >::variable( 0.0 );
    DA< 8 > f = sin( x );
    DA< 8 > g = inv( f );

    // Evaluate f(g(y)) at several y values near 0
    for ( double y : { -0.3, -0.1, 0.0, 0.1, 0.3 } )
    {
        double gy = g.eval( y );
        double fgy = f.eval( gy );
        EXPECT_NEAR( fgy, y, 1e-6 ) << "  y=" << y;
    }
}

TEST( CompInverse, RoundTripExp )
{
    // f(x) = exp(x) - 1, g = inv(f) ≈ log(1+y)
    // Verify f(g(y)) ≈ y
    auto x = DA< 10 >::variable( 0.0 );
    DA< 10 > f = exp( x ) - 1.0;
    DA< 10 > g = inv( f );

    for ( double y : { -0.3, -0.1, 0.0, 0.1, 0.3 } )
    {
        double gy = g.eval( y );
        double fgy = f.eval( gy );
        EXPECT_NEAR( fgy, y, 1e-6 ) << "  y=" << y;
    }
}

TEST( CompInverse, FromExpression )
{
    // inv should work on expression nodes, not just materialized TDA
    auto x = DA< 5 >::variable( 0.0 );
    DA< 5 > g = inv( x + x * x );  // f(x) = x + x^2
    EXPECT_NEAR( g[1], 1.0, kTol );
    EXPECT_NEAR( g[2], -1.0, kTol );
    EXPECT_NEAR( g[3], 2.0, kTol );
}
