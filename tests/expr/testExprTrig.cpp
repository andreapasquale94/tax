#include "testUtils.hpp"

// =============================================================================
// Sin — Taylor series of sin(f)
// =============================================================================

TEST( Sin, ConstantSin )
{
    DA< 3 > a{ 1.0 };
    DA< 3 > r = sin( a );
    EXPECT_NEAR( r.value(), std::sin( 1.0 ), kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Sin, SinOfVariableZero )
{
    auto x = DA< 5 >::variable< 0 >( { 0.0 } );
    DA< 5 > r = sin( x );
    EXPECT_NEAR( r[0], 0.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
    EXPECT_NEAR( r[2], 0.0, kTol );
    EXPECT_NEAR( r[3], -1.0 / 6.0, kTol );
    EXPECT_NEAR( r[4], 0.0, kTol );
    EXPECT_NEAR( r[5], 1.0 / 120.0, kTol );
}

TEST( Sin, SinOfVariable )
{
    auto x = DA< 5 >::variable( 1.0 );
    DA< 5 > r = sin( x );
    EXPECT_NEAR( r[0], 0.8414709848078965, kTol );
    EXPECT_NEAR( r[1], 0.5403023058681398, kTol );
    EXPECT_NEAR( r[2], -0.42073549240394825, kTol );
    EXPECT_NEAR( r[3], -0.09005038431135663, kTol );
    EXPECT_NEAR( r[4], 0.03506129103366235, kTol );
    EXPECT_NEAR( r[5], 0.004502519215567832, kTol );
}

TEST( Sin, DerivativeCheck )
{
    // d/dx sin(x) at x0=pi/4: value = sin(pi/4), deriv = cos(pi/4)
    constexpr double x0 = M_PI / 4.0;
    auto x = DA< 3 >::variable< 0 >( { x0 } );
    DA< 3 > r = sin( x );
    EXPECT_NEAR( r.value(), std::sin( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), std::cos( x0 ), kTol );
}

TEST( Sin, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 2.0 } );
    DAn< 3, 2 > r = sin( x );
    // sin(x) should not depend on y
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::sin( 1.0 ), kTol );
}

TEST( Sin, OfExpression )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 1.0 } );
    DAn< 3, 2 > r = sin( x + y );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), 0.9092974268256817, kTol );
    EXPECT_NEAR( r.coeff( { 1, 0 } ), -0.4161468365471424, kTol );
    EXPECT_NEAR( r.coeff( { 0, 1 } ), -0.4161468365471424, kTol );
    EXPECT_NEAR( r.coeff( { 2, 0 } ), -0.45464871341284085, kTol );
    EXPECT_NEAR( r.coeff( { 1, 1 } ), -0.9092974268256817, kTol );
    EXPECT_NEAR( r.coeff( { 0, 2 } ), -0.45464871341284085, kTol );
    EXPECT_NEAR( r.coeff( { 3, 0 } ), 0.0693578060911904, kTol );
    EXPECT_NEAR( r.coeff( { 2, 1 } ), 0.2080734182735712, kTol );
    EXPECT_NEAR( r.coeff( { 1, 2 } ), 0.2080734182735712, kTol );
    EXPECT_NEAR( r.coeff( { 0, 3 } ), 0.0693578060911904, kTol );
}

// =============================================================================
// Cos — Taylor series of cos(f)
// =============================================================================

TEST( Cos, ConstantCos )
{
    DA< 3 > a{ 1.0 };
    DA< 3 > r = cos( a );
    EXPECT_NEAR( r.value(), std::cos( 1.0 ), kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Cos, CosOfVariableZero )
{
    // cos(x) at x0=0: coeffs are 1, 0, -1/2, 0, 1/24, ...
    auto x = DA< 5 >::variable< 0 >( { 0.0 } );
    DA< 5 > r = cos( x );
    EXPECT_NEAR( r[0], 1.0, kTol );
    EXPECT_NEAR( r[1], 0.0, kTol );
    EXPECT_NEAR( r[2], -0.5, kTol );
    EXPECT_NEAR( r[3], 0.0, kTol );
    EXPECT_NEAR( r[4], 1.0 / 24.0, kTol );
    EXPECT_NEAR( r[5], 0.0, kTol );
}

TEST( Cos, CosOfVariable )
{
    // cos(x) at x0=0: coeffs are 1, 0, -1/2, 0, 1/24, ...
    auto x = DA< 5 >::variable( 1.0 );
    DA< 5 > r = cos( x );
    EXPECT_NEAR( r[0], 0.5403023058681398, kTol );
    EXPECT_NEAR( r[1], -0.841470984807896, kTol );
    EXPECT_NEAR( r[2], -0.270151152934069, kTol );
    EXPECT_NEAR( r[3], 0.1402451641346494, kTol );
    EXPECT_NEAR( r[4], 0.0225125960778391, kTol );
    EXPECT_NEAR( r[5], -0.007012258206732, kTol );
}

TEST( Cos, DerivativeCheck )
{
    // d/dx cos(x) at x0=pi/4: value = cos(pi/4), deriv = -sin(pi/4)
    constexpr double x0 = M_PI / 4.0;
    auto x = DA< 3 >::variable< 0 >( { x0 } );
    DA< 3 > r = cos( x );
    EXPECT_NEAR( r.value(), std::cos( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), -std::sin( x0 ), kTol );
}

TEST( Cos, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 2.0 } );
    DAn< 3, 2 > r = cos( x );
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::cos( 1.0 ), kTol );
}

TEST( Cos, OfExpression )
{
    DA< 4 > a{ M_PI / 6.0 };
    DA< 4 > r = cos( a + a );  // cos(pi/3)
    EXPECT_NEAR( r.value(), std::cos( M_PI / 3.0 ), kTol );
}

// =============================================================================
// Pythagorean identity: sin^2 + cos^2 = 1
// =============================================================================

TEST( SinCosIdentity, PythagoreanUnivariate )
{
    auto x = DA< 5 >::variable< 0 >( { 1.5 } );
    DA< 5 > s = sin( x );
    DA< 5 > c = cos( x );
    DA< 5 > r = square( s ) + square( c );
    // Should be identically 1
    EXPECT_NEAR( r[0], 1.0, kTol );
    for ( std::size_t k = 1; k < DA< 5 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( SinCosIdentity, PythagoreanBivariate )
{
    auto [x, y] = DAn< 4, 2 >::variables( { 0.7, -1.3 } );
    DAn< 4, 2 > s = sin( x + y );
    DAn< 4, 2 > c = cos( x + y );
    DAn< 4, 2 > r = square( s ) + square( c );
    EXPECT_NEAR( r[0], 1.0, kTol );
    for ( std::size_t k = 1; k < DAn< 4, 2 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

// =============================================================================
// SinCos — returns both sin and cos in one call
// =============================================================================

TEST( SinCos, MatchesSeparate )
{
    auto x = DA< 5 >::variable< 0 >( { 1.5 } );
    auto [s, c] = sincos( x );
    DA< 5 > s2 = sin( x );
    DA< 5 > c2 = cos( x );
    ExpectCoeffsNear( s, s2 );
    ExpectCoeffsNear( c, c2 );
}

TEST( SinCos, MatchesSeparateBivariate )
{
    auto [x, y] = DAn< 4, 2 >::variables( { 0.7, -1.3 } );
    auto expr = x * y + x;
    auto [s, c] = sincos( expr );
    DAn< 4, 2 > s2 = sin( expr );
    DAn< 4, 2 > c2 = cos( expr );
    ExpectCoeffsNear( s, s2 );
    ExpectCoeffsNear( c, c2 );
}

TEST( SinCos, Pythagorean )
{
    auto x = DA< 5 >::variable< 0 >( { 2.3 } );
    auto [s, c] = sincos( x );
    DA< 5 > r = square( s ) + square( c );
    EXPECT_NEAR( r[0], 1.0, kTol );
    for ( std::size_t k = 1; k < DA< 5 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

// =============================================================================
// Tan — Taylor series of tan(f)
// =============================================================================

TEST( Tan, ConstantTan )
{
    DA< 3 > a{ 0.5 };
    DA< 3 > r = tan( a );
    EXPECT_NEAR( r.value(), std::tan( 0.5 ), kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Tan, TanOfVariableZero )
{
    auto x = DA< 5 >::variable< 0 >( { 0.0 } );
    DA< 5 > r = tan( x );
    EXPECT_NEAR( r[0], 0.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
    EXPECT_NEAR( r[2], 0.0, kTol );
    EXPECT_NEAR( r[3], 1.0 / 3.0, kTol );
    EXPECT_NEAR( r[4], 0.0, kTol );
    EXPECT_NEAR( r[5], 2.0 / 15.0, kTol );
}

TEST( Tan, TanOfVariable )
{
    auto x = DA< 5 >::variable( 1.0 );
    DA< 5 > r = tan( x );
    EXPECT_NEAR( r[0], 1.5574077246549023, kTol );
    EXPECT_NEAR( r[1], 3.42551882081476, kTol );
    EXPECT_NEAR( r[2], 5.3349294724876595, kTol );
    EXPECT_NEAR( r[3], 9.450499977879637, kTol );
    EXPECT_NEAR( r[4], 16.496591491563287, kTol );
    EXPECT_NEAR( r[5], 28.918208319192765, kTol );
}

TEST( Tan, MatchesSinOverCos )
{
    auto x = DA< 5 >::variable< 0 >( { 0.8 } );
    DA< 5 > r1 = tan( x );
    DA< 5 > r2 = sin( x ) / cos( x );
    ExpectCoeffsNear( r1, r2 );
}

TEST( Tan, DerivativeCheck )
{
    // d/dx tan(x) at x0 = 1/sec^2(x0) = 1 + tan^2(x0)
    constexpr double x0 = 0.7;
    auto x = DA< 3 >::variable< 0 >( { x0 } );
    DA< 3 > r = tan( x );
    EXPECT_NEAR( r.value(), std::tan( x0 ), kTol );
    double expected_deriv = 1.0 + std::tan( x0 ) * std::tan( x0 );
    EXPECT_NEAR( r.derivative( { 1 } ), expected_deriv, kTol );
}

TEST( Tan, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 0.5, 1.0 } );
    DAn< 3, 2 > r1 = tan( x + y );
    DAn< 3, 2 > r2 = sin( x + y ) / cos( x + y );
    ExpectCoeffsNear( r1, r2 );
}

TEST( Tan, OfExpression )
{
    auto x = DA< 4 >::variable< 0 >( { 0.3 } );
    DA< 4 > r1 = tan( x * 2.0 );
    DA< 4 > r2 = sin( x * 2.0 ) / cos( x * 2.0 );
    ExpectCoeffsNear( r1, r2 );
}
