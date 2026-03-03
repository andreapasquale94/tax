#include "testUtils.hpp"

// =============================================================================
// SquareExpr — f^2 via Cauchy self-convolution
// =============================================================================

TEST( Square, ConstantSquare )
{
    DA< 3 > a{ 3.0 };
    DA< 3 > r = square( a );
    EXPECT_NEAR( r.value(), 9.0, kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Square, LinearSquare )
{
    // (1+x)^2 = 1 + 2x + x^2
    auto x = DA< 4 >::variable< 0 >( { 1.0 } );
    DA< 4 > r = square( x );
    EXPECT_NEAR( r[0], 1.0, kTol );
    EXPECT_NEAR( r[1], 2.0, kTol );
    EXPECT_NEAR( r[2], 1.0, kTol );
    for ( std::size_t k = 3; k < DA< 4 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Square, MatchesSelfMultiply )
{
    auto x = DA< 5 >::variable< 0 >( { 2.0 } );
    DA< 5 > r1 = square( x );
    DA< 5 > r2 = x * x;
    ExpectCoeffsNear( r1, r2 );
}

TEST( Square, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 2.0 } );
    DAn< 3, 2 > r1 = square( x + y );
    DAn< 3, 2 > r2 = ( x + y ) * ( x + y );
    ExpectCoeffsNear( r1, r2 );
}

TEST( Square, OfExpression )
{
    // square(a + b) should work when operand is a non-leaf
    auto x = DA< 4 >::variable< 0 >( { 0.0 } );
    DA< 4 > one{ 1.0 };
    DA< 4 > r = square( x + one );  // (1+x)^2
    EXPECT_NEAR( r[0], 1.0, kTol );
    EXPECT_NEAR( r[1], 2.0, kTol );
    EXPECT_NEAR( r[2], 1.0, kTol );
}

// =============================================================================
// CubeExpr — f^3 via direct triple convolution
// =============================================================================

TEST( Cube, ConstantCube )
{
    DA< 3 > a{ 2.0 };
    DA< 3 > r = cube( a );
    EXPECT_NEAR( r.value(), 8.0, kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Cube, LinearCube )
{
    // (1+x)^3 = 1 + 3x + 3x^2 + x^3
    auto x = DA< 4 >::variable< 0 >( { 1.0 } );
    DA< 4 > r = cube( x );
    EXPECT_NEAR( r[0], 1.0, kTol );
    EXPECT_NEAR( r[1], 3.0, kTol );
    EXPECT_NEAR( r[2], 3.0, kTol );
    EXPECT_NEAR( r[3], 1.0, kTol );
    EXPECT_NEAR( r[4], 0.0, kTol );
}

TEST( Cube, MatchesTripleMultiply )
{
    auto x = DA< 5 >::variable< 0 >( { 2.0 } );
    DA< 5 > r1 = cube( x );
    DA< 5 > r2 = x * x * x;
    ExpectCoeffsNear( r1, r2 );
}

TEST( Cube, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 1.0 } );
    DAn< 3, 2 > r1 = cube( x );
    DAn< 3, 2 > r2 = x * x * x;
    ExpectCoeffsNear( r1, r2 );
}

TEST( Cube, OfExpression )
{
    auto x = DA< 4 >::variable< 0 >( { 0.0 } );
    DA< 4 > one{ 1.0 };
    DA< 4 > r = cube( x + one );  // (1+x)^3
    EXPECT_NEAR( r[0], 1.0, kTol );
    EXPECT_NEAR( r[1], 3.0, kTol );
    EXPECT_NEAR( r[2], 3.0, kTol );
    EXPECT_NEAR( r[3], 1.0, kTol );
}

// =============================================================================
// SqrtExpr — Taylor series of sqrt(f)
// =============================================================================

TEST( Sqrt, ConstantSqrt )
{
    DA< 3 > a{ 9.0 };
    DA< 3 > r = sqrt( a );
    EXPECT_NEAR( r.value(), 3.0, kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Sqrt, Sqrt1PlusX )
{
    // sqrt(1+x) Taylor coefficients:
    //   c[0]=1, c[1]=1/2, c[2]=-1/8, c[3]=1/16, c[4]=-5/128
    auto x = DA< 4 >::variable< 0 >( { 1.0 } );
    DA< 4 > r = sqrt( x );
    EXPECT_NEAR( r[0], 1.0, kTol );
    EXPECT_NEAR( r[1], 0.5, kTol );
    EXPECT_NEAR( r[2], -0.125, kTol );
    EXPECT_NEAR( r[3], 0.0625, kTol );
    EXPECT_NEAR( r[4], -5.0 / 128.0, kTol );
}

TEST( Sqrt, SqrtSquaredIsIdentity )
{
    // sqrt(x)^2 should recover x
    auto x = DA< 5 >::variable< 0 >( { 4.0 } );
    DA< 5 > r = square( sqrt( x ) );
    ExpectCoeffsNear( r, x );
}

TEST( Sqrt, SquareThenSqrt )
{
    // sqrt(x^2) when x>0 should give |x| = x
    auto x = DA< 4 >::variable< 0 >( { 3.0 } );
    DA< 4 > r = sqrt( square( x ) );
    ExpectCoeffsNear( r, x );
}

TEST( Sqrt, DerivativeCheck )
{
    // d/dx sqrt(x) at x0=4: value = 2, deriv = 1/(2*sqrt(4)) = 0.25
    auto x = DA< 3 >::variable< 0 >( { 4.0 } );
    DA< 3 > r = sqrt( x );
    EXPECT_NEAR( r.value(), 2.0, kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), 0.25, kTol );
}

TEST( Sqrt, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 4.0, 1.0 } );
    DAn< 3, 2 > r1 = sqrt( x );
    // sqrt(x) should not depend on y
    EXPECT_NEAR( r1.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r1.coeff( { 0, 0 } ), 2.0, kTol );
}

TEST( Sqrt, OfExpression )
{
    // sqrt(a+b) where a+b is a non-leaf expression
    DA< 4 > a{ 3.0 }, b{ 1.0 };
    DA< 4 > r = sqrt( a + b );  // sqrt(4) = 2
    EXPECT_NEAR( r.value(), 2.0, kTol );
}

// =============================================================================
// CbrtExpr — Taylor series of cbrt(f)
// =============================================================================

TEST( Cbrt, ConstantCbrt )
{
    DA< 3 > a{ 8.0 };
    DA< 3 > r = cbrt( a );
    EXPECT_NEAR( r.value(), 2.0, kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Cbrt, Cbrt8PlusX )
{
    // cbrt(8+x) = 2 + x/12 - x^2/288 + 5 x^3/20736 + ...
    auto x = DA< 4 >::variable< 0 >( { 8.0 } );
    DA< 4 > r = cbrt( x );
    EXPECT_NEAR( r[0], 2.0, kTol );
    EXPECT_NEAR( r[1], 1.0 / 12.0, kTol );
    EXPECT_NEAR( r[2], -1.0 / 288.0, kTol );
    EXPECT_NEAR( r[3], 5.0 / 20736.0, kTol );
}

TEST( Cbrt, CbrtCubedIsIdentity )
{
    auto x = DA< 5 >::variable< 0 >( { 8.0 } );
    DA< 5 > r = cube( cbrt( x ) );
    ExpectCoeffsNear( r, x );
}

TEST( Cbrt, CubeThenCbrt )
{
    auto x = DA< 4 >::variable< 0 >( { 3.0 } );
    DA< 4 > r = cbrt( cube( x ) );
    ExpectCoeffsNear( r, x );
}

TEST( Cbrt, DerivativeCheck )
{
    // d/dx cbrt(x) at x0=8: value = 2, deriv = 1/(3*cbrt(8)^2) = 1/12
    auto x = DA< 3 >::variable< 0 >( { 8.0 } );
    DA< 3 > r = cbrt( x );
    EXPECT_NEAR( r.value(), 2.0, kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), 1.0 / 12.0, kTol );
}

TEST( Cbrt, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 8.0, 1.0 } );
    DAn< 3, 2 > r1 = cbrt( x );
    // cbrt(x) should not depend on y
    EXPECT_NEAR( r1.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r1.coeff( { 0, 0 } ), 2.0, kTol );
}

TEST( Cbrt, OfExpression )
{
    DA< 4 > a{ 7.0 }, b{ 1.0 };
    DA< 4 > r = cbrt( a + b );  // cbrt(8) = 2
    EXPECT_NEAR( r.value(), 2.0, kTol );
}
