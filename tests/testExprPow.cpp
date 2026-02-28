#include "testUtils.hpp"

// =============================================================================
// Exp — Taylor series of exp(f)
// =============================================================================

TEST( Exp, Constant )
{
    DA< 3 > a{ 2.0 };
    DA< 3 > r = exp( a );
    EXPECT_NEAR( r.value(), std::exp( 2.0 ), kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Exp, KnownSeries )
{
    // exp(x) at x0=0: coeffs are 1/k!
    auto x = DA< 5 >::variable( 0.0 );
    DA< 5 > r = exp( x );
    EXPECT_NEAR( r[0], 1.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
    EXPECT_NEAR( r[2], 1.0 / 2.0, kTol );
    EXPECT_NEAR( r[3], 1.0 / 6.0, kTol );
    EXPECT_NEAR( r[4], 1.0 / 24.0, kTol );
    EXPECT_NEAR( r[5], 1.0 / 120.0, kTol );
}

TEST( Exp, DerivativeCheck )
{
    constexpr double x0 = 1.5;
    auto x = DA< 3 >::variable( x0 );
    DA< 3 > r = exp( x );
    EXPECT_NEAR( r.value(), std::exp( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), std::exp( x0 ), kTol );
}

TEST( Exp, ExpLogIdentity )
{
    // exp(log(x)) = x
    auto x = DA< 5 >::variable( 2.0 );
    DA< 5 > r = exp( log( x ) );
    DA< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Exp, LogExpIdentity )
{
    // log(exp(x)) = x
    auto x = DA< 5 >::variable( 1.0 );
    DA< 5 > r = log( exp( x ) );
    DA< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Exp, SumRule )
{
    // exp(a+b) = exp(a)*exp(b)
    auto x = DA< 4 >::variable( 1.0 );
    DA< 4 > a = x * 2.0;
    DA< 4 > b = x * 0.5 + 1.0;
    DA< 4 > r1 = exp( a + b );
    DA< 4 > r2 = exp( a ) * exp( b );
    ExpectCoeffsNear( r1, r2 );
}

TEST( Exp, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 2.0 } );
    DAn< 3, 2 > r = exp( x );
    // exp(x) should not depend on y
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::exp( 1.0 ), kTol );
}

// =============================================================================
// IPow — integer exponent via binary exponentiation
// =============================================================================

TEST( IPow, ZeroPower )
{
    auto x = DA< 4 >::variable( 3.0 );
    DA< 4 > r = ipow( x, 0 );
    EXPECT_NEAR( r.value(), 1.0, kTol );
    for ( std::size_t k = 1; k < DA< 4 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( IPow, FirstPower )
{
    auto x = DA< 4 >::variable( 3.0 );
    DA< 4 > r = ipow( x, 1 );
    ExpectCoeffsNear( r, x );
}

TEST( IPow, SquareMatchesSquare )
{
    auto x = DA< 5 >::variable( 2.0 );
    DA< 5 > r1 = ipow( x, 2 );
    DA< 5 > r2 = square( x );
    ExpectCoeffsNear( r1, r2 );
}

TEST( IPow, CubeMatchesCube )
{
    auto x = DA< 5 >::variable( 2.0 );
    DA< 5 > r1 = ipow( x, 3 );
    DA< 5 > r2 = cube( x );
    ExpectCoeffsNear( r1, r2 );
}

TEST( IPow, FourthPower )
{
    auto x = DA< 5 >::variable( 2.0 );
    DA< 5 > r1 = ipow( x, 4 );
    DA< 5 > r2 = square( square( x ) );
    ExpectCoeffsNear( r1, r2 );
}

TEST( IPow, NegativeOne )
{
    auto x = DA< 4 >::variable( 3.0 );
    DA< 4 > r1 = ipow( x, -1 );
    DA< 4 > r2 = 1.0 / x;
    ExpectCoeffsNear( r1, r2 );
}

TEST( IPow, NegativeTwo )
{
    auto x = DA< 4 >::variable( 2.0 );
    DA< 4 > r1 = ipow( x, -2 );
    DA< 4 > r2 = 1.0 / square( x );
    ExpectCoeffsNear( r1, r2 );
}

TEST( IPow, ViaPow )
{
    auto x = DA< 5 >::variable( 2.0 );
    DA< 5 > r1 = pow( x, 3 );
    DA< 5 > r2 = ipow( x, 3 );
    ExpectCoeffsNear( r1, r2 );
}

// =============================================================================
// DPow — real exponent via recurrence
// =============================================================================

TEST( DPow, HalfMatchesSqrt )
{
    auto x = DA< 5 >::variable( 4.0 );
    DA< 5 > r1 = dpow( x, 0.5 );
    DA< 5 > r2 = sqrt( x );
    ExpectCoeffsNear( r1, r2 );
}

TEST( DPow, IntegerExponentMatchesIPow )
{
    auto x = DA< 5 >::variable( 2.0 );
    DA< 5 > r1 = dpow( x, 3.0 );
    DA< 5 > r2 = ipow( x, 3 );
    ExpectCoeffsNear( r1, r2 );
}

TEST( DPow, FractionalExponent )
{
    // x^1.5 = x * sqrt(x)
    auto x = DA< 5 >::variable( 4.0 );
    DA< 5 > r1 = dpow( x, 1.5 );
    DA< 5 > r2 = x * sqrt( x );
    ExpectCoeffsNear( r1, r2 );
}

TEST( DPow, DerivativeCheck )
{
    // d/dx x^c = c * x^(c-1)
    constexpr double x0 = 3.0, c = 2.7;
    auto x = DA< 3 >::variable( x0 );
    DA< 3 > r = dpow( x, c );
    EXPECT_NEAR( r.value(), std::pow( x0, c ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), c * std::pow( x0, c - 1.0 ), kTol );
}

TEST( DPow, PowerRule )
{
    // (x^a)^b = x^(a*b)
    constexpr double x0 = 2.0;
    auto x = DA< 5 >::variable( x0 );
    DA< 5 > r1 = dpow( dpow( x, 1.5 ), 2.0 );
    DA< 5 > r2 = dpow( x, 3.0 );
    ExpectCoeffsNear( r1, r2 );
}

TEST( DPow, ViaPow )
{
    auto x = DA< 5 >::variable( 2.0 );
    DA< 5 > r1 = pow( x, 2.5 );
    DA< 5 > r2 = dpow( x, 2.5 );
    ExpectCoeffsNear( r1, r2 );
}

TEST( DPow, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 2.0, 3.0 } );
    DAn< 3, 2 > r1 = dpow( x * y, 0.5 );
    DAn< 3, 2 > r2 = sqrt( x * y );
    ExpectCoeffsNear( r1, r2 );
}

// =============================================================================
// TPow — DA exponent via exp(g * log(f))
// =============================================================================

TEST( TPow, ConstantExponent )
{
    // x^2.0 (constant DA)
    auto x = DA< 5 >::variable( 3.0 );
    DA< 5 > c{ 2.0 };
    DA< 5 > r1 = tpow( x, c );
    DA< 5 > r2 = dpow( x, 2.0 );
    ExpectCoeffsNear( r1, r2 );
}

TEST( TPow, IdentityExponent )
{
    // x^x at x0 = 2 => value = 4, deriv = x^x*(1+log(x))
    constexpr double x0 = 2.0;
    auto x = DA< 3 >::variable( x0 );
    DA< 3 > r = tpow( x, x );
    EXPECT_NEAR( r.value(), std::pow( x0, x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), std::pow( x0, x0 ) * ( 1.0 + std::log( x0 ) ), kTol );
}

TEST( TPow, ViaPow )
{
    auto x = DA< 4 >::variable( 2.0 );
    auto y = DA< 4 >::variable( 1.5 );
    DA< 4 > r1 = pow( x, y );
    DA< 4 > r2 = tpow( x, y );
    ExpectCoeffsNear( r1, r2 );
}

TEST( TPow, VsExpLogManual )
{
    // tpow(f, g) should equal exp(g * log(f))
    auto x = DA< 4 >::variable( 2.0 );
    DA< 4 > f = x * 1.5 + 0.5;
    DA< 4 > g = x * 0.3 + 1.0;
    DA< 4 > r1 = tpow( f, g );
    DA< 4 > r2 = exp( g * log( f ) );
    ExpectCoeffsNear( r1, r2 );
}
