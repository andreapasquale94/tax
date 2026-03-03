#include "testUtils.hpp"

// =============================================================================
// Erf
// =============================================================================

TEST( Erf, Constant )
{
    DA< 3 > a{ 1.0 };
    DA< 3 > r = erf( a );
    EXPECT_NEAR( r.value(), std::erf( 1.0 ), kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Erf, AtZero )
{
    // erf(x) at x0=0: coeffs are 0, 2/sqrt(pi), 0, -2/(3*sqrt(pi)), ...
    auto x = DA< 5 >::variable( 0.0 );
    DA< 5 > r = erf( x );
    const double c = 2.0 / std::sqrt( M_PI );
    EXPECT_NEAR( r[0], 0.0, kTol );
    EXPECT_NEAR( r[1], c, kTol );
    EXPECT_NEAR( r[2], 0.0, kTol );
    EXPECT_NEAR( r[3], -c / 3.0, kTol );
    EXPECT_NEAR( r[4], 0.0, kTol );
    EXPECT_NEAR( r[5], c / 10.0, kTol );
}

TEST( Erf, DerivativeCheck )
{
    // d/dx erf(x) = (2/sqrt(pi)) * exp(-x^2)
    constexpr double x0 = 0.5;
    auto x = DA< 3 >::variable( x0 );
    DA< 3 > r = erf( x );
    double expected_deriv = ( 2.0 / std::sqrt( M_PI ) ) * std::exp( -x0 * x0 );
    EXPECT_NEAR( r.value(), std::erf( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), expected_deriv, kTol );
}

TEST( Erf, SymmetryOdd )
{
    // erf(-x) = -erf(x)
    auto x = DA< 5 >::variable( 0.7 );
    DA< 5 > r1 = erf( -x );
    DA< 5 > r2 = -erf( x );
    ExpectCoeffsNear( r1, r2 );
}

TEST( Erf, LargeArgument )
{
    // erf(3) is very close to 1
    DA< 3 > a{ 3.0 };
    DA< 3 > r = erf( a );
    EXPECT_NEAR( r.value(), std::erf( 3.0 ), kTol );
}

TEST( Erf, OfExpression )
{
    DA< 4 > a{ 0.3 }, b{ 0.2 };
    DA< 4 > r = erf( a + b );
    EXPECT_NEAR( r.value(), std::erf( 0.5 ), kTol );
}

TEST( Erf, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 0.5, 1.0 } );
    DAn< 3, 2 > r = erf( x );
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::erf( 0.5 ), kTol );
}
