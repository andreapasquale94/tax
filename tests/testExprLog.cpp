#include "testUtils.hpp"

// =============================================================================
// Log (natural logarithm) — Taylor series of log(f)
// =============================================================================

TEST( Log, ConstantLog )
{
    DA< 3 > a{ 2.0 };
    DA< 3 > r = log( a );
    EXPECT_NEAR( r.value(), std::log( 2.0 ), kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Log, LogOfVariable )
{
    // log(1+x) at x0=1: coeffs are 0, 1, -1/2, 1/3, -1/4, 1/5, ...
    auto x = DA< 5 >::variable< 0 >( { 1.0 } );
    DA< 5 > r = log( x );
    EXPECT_NEAR( r[0], 0.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
    EXPECT_NEAR( r[2], -1.0 / 2.0, kTol );
    EXPECT_NEAR( r[3], 1.0 / 3.0, kTol );
    EXPECT_NEAR( r[4], -1.0 / 4.0, kTol );
    EXPECT_NEAR( r[5], 1.0 / 5.0, kTol );
}

TEST( Log, DerivativeCheck )
{
    // d/dx log(x) at x0=3: value = log(3), deriv = 1/3
    constexpr double x0 = 3.0;
    auto x = DA< 3 >::variable< 0 >( { x0 } );
    DA< 3 > r = log( x );
    EXPECT_NEAR( r.value(), std::log( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), 1.0 / x0, kTol );
}

TEST( Log, ExpLogIsIdentity )
{
    // exp(log(f)) should recover f for any f with f[0] > 0
    // We test this indirectly: log(exp(x)) ~ x via the derivative chain
    // More directly: log(a^2) = 2*log(a) for a > 0
    auto x = DA< 5 >::variable< 0 >( { 2.0 } );
    DA< 5 > r1 = log( square( x ) );
    DA< 5 > r2 = log( x ) * 2.0;
    ExpectCoeffsNear( r1, r2 );
}

TEST( Log, LogOfProduct )
{
    // log(a*b) = log(a) + log(b)
    auto x = DA< 4 >::variable< 0 >( { 2.0 } );
    DA< 4 > a{ 3.0 };
    DA< 4 > r1 = log( x * a );
    DA< 4 > r2 = log( x ) + log( a );
    ExpectCoeffsNear( r1, r2 );
}

TEST( Log, LogOfReciprocal )
{
    // log(1/x) = -log(x)
    auto x = DA< 5 >::variable< 0 >( { 2.5 } );
    DA< 5 > r1 = log( 1.0 / x );
    DA< 5 > r2 = -log( x );
    ExpectCoeffsNear( r1, r2 );
}

TEST( Log, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 2.0, 1.0 } );
    DAn< 3, 2 > r = log( x );
    // log(x) should not depend on y
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::log( 2.0 ), kTol );
}

TEST( Log, OfExpression )
{
    DA< 4 > a{ 1.0 }, b{ 1.0 };
    DA< 4 > r = log( a + b );  // log(2)
    EXPECT_NEAR( r.value(), std::log( 2.0 ), kTol );
}

TEST( Log, BivariateSumIdentity )
{
    // log(x*y) = log(x) + log(y)
    auto [x, y] = DAn< 3, 2 >::variables( { 2.0, 3.0 } );
    DAn< 3, 2 > r1 = log( x * y );
    DAn< 3, 2 > r2 = log( x ) + log( y );
    ExpectCoeffsNear( r1, r2 );
}

// =============================================================================
// Log10 — base-10 logarithm
// =============================================================================

TEST( Log10, ConstantLog10 )
{
    DA< 3 > a{ 100.0 };
    DA< 3 > r = log10( a );
    EXPECT_NEAR( r.value(), 2.0, kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Log10, MatchesLogScaled )
{
    auto x = DA< 5 >::variable< 0 >( { 3.0 } );
    DA< 5 > r1 = log10( x );
    DA< 5 > r2 = log( x ) * ( 1.0 / std::log( 10.0 ) );
    ExpectCoeffsNear( r1, r2 );
}

TEST( Log10, DerivativeCheck )
{
    // d/dx log10(x) at x0 = 1/(x0 * log(10))
    constexpr double x0 = 5.0;
    auto x = DA< 3 >::variable< 0 >( { x0 } );
    DA< 3 > r = log10( x );
    EXPECT_NEAR( r.value(), std::log10( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), 1.0 / ( x0 * std::log( 10.0 ) ), kTol );
}

TEST( Log10, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 10.0, 2.0 } );
    DAn< 3, 2 > r1 = log10( x * y );
    DAn< 3, 2 > r2 = log10( x ) + log10( y );
    ExpectCoeffsNear( r1, r2 );
}

TEST( Log10, OfExpression )
{
    auto x = DA< 4 >::variable< 0 >( { 5.0 } );
    DA< 4 > r1 = log10( x * 2.0 );
    DA< 4 > r2 = log( x * 2.0 ) * ( 1.0 / std::log( 10.0 ) );
    ExpectCoeffsNear( r1, r2 );
}
