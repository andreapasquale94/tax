#include "testUtils.hpp"

// =============================================================================
// Asin
// =============================================================================

TEST( Asin, Constant )
{
    TE< 3 > a{ 0.5 };
    TE< 3 > r = asin( a );
    EXPECT_NEAR( r.value(), std::asin( 0.5 ), kTol );
    for ( std::size_t k = 1; k < TE< 3 >::nCoefficients; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Asin, KnownSeries )
{
    // asin(x) at x0=0: coeffs are 0, 1, 0, 1/6, 0, 3/40
    auto x = TE< 5 >::variable( 0.0 );
    TE< 5 > r = asin( x );
    EXPECT_NEAR( r[0], 0.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
    EXPECT_NEAR( r[2], 0.0, kTol );
    EXPECT_NEAR( r[3], 1.0 / 6.0, kTol );
    EXPECT_NEAR( r[4], 0.0, kTol );
    EXPECT_NEAR( r[5], 3.0 / 40.0, kTol );
}

TEST( Asin, DerivativeCheck )
{
    constexpr double x0 = 0.3;
    auto x = TE< 3 >::variable( x0 );
    TE< 3 > r = asin( x );
    EXPECT_NEAR( r.value(), std::asin( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), 1.0 / std::sqrt( 1.0 - x0 * x0 ), kTol );
}

TEST( Asin, RoundTrip )
{
    // sin(asin(x)) = x
    auto x = TE< 5 >::variable( 0.4 );
    TE< 5 > r = sin( asin( x ) );
    TE< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Asin, RoundTripReverse )
{
    // asin(sin(x)) = x  (for |x| < pi/2)
    auto x = TE< 5 >::variable( 0.3 );
    TE< 5 > r = asin( sin( x ) );
    TE< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Asin, Bivariate )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 0.3, 0.5 } );
    TEn< 3, 2 > r = asin( x );
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::asin( 0.3 ), kTol );
}

// =============================================================================
// Acos
// =============================================================================

TEST( Acos, Constant )
{
    TE< 3 > a{ 0.5 };
    TE< 3 > r = acos( a );
    EXPECT_NEAR( r.value(), std::acos( 0.5 ), kTol );
    for ( std::size_t k = 1; k < TE< 3 >::nCoefficients; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Acos, DerivativeCheck )
{
    constexpr double x0 = 0.3;
    auto x = TE< 3 >::variable( x0 );
    TE< 3 > r = acos( x );
    EXPECT_NEAR( r.value(), std::acos( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), -1.0 / std::sqrt( 1.0 - x0 * x0 ), kTol );
}

TEST( Acos, AsinPlusAcosIsPiOver2 )
{
    auto x = TE< 5 >::variable( 0.4 );
    TE< 5 > r = asin( x ) + acos( x );
    EXPECT_NEAR( r.value(), std::acos( -1.0 ) / 2.0, kTol );
    for ( std::size_t k = 1; k < TE< 5 >::nCoefficients; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Acos, RoundTrip )
{
    // cos(acos(x)) = x
    auto x = TE< 5 >::variable( 0.4 );
    TE< 5 > r = cos( acos( x ) );
    TE< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

// =============================================================================
// Atan
// =============================================================================

TEST( Atan, Constant )
{
    TE< 3 > a{ 1.0 };
    TE< 3 > r = atan( a );
    EXPECT_NEAR( r.value(), std::atan( 1.0 ), kTol );
    for ( std::size_t k = 1; k < TE< 3 >::nCoefficients; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Atan, KnownSeries )
{
    // atan(x) at x0=0: coeffs are 0, 1, 0, -1/3, 0, 1/5
    auto x = TE< 5 >::variable( 0.0 );
    TE< 5 > r = atan( x );
    EXPECT_NEAR( r[0], 0.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
    EXPECT_NEAR( r[2], 0.0, kTol );
    EXPECT_NEAR( r[3], -1.0 / 3.0, kTol );
    EXPECT_NEAR( r[4], 0.0, kTol );
    EXPECT_NEAR( r[5], 1.0 / 5.0, kTol );
}

TEST( Atan, DerivativeCheck )
{
    constexpr double x0 = 2.0;
    auto x = TE< 3 >::variable( x0 );
    TE< 3 > r = atan( x );
    EXPECT_NEAR( r.value(), std::atan( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), 1.0 / ( 1.0 + x0 * x0 ), kTol );
}

TEST( Atan, RoundTrip )
{
    // tan(atan(x)) = x
    auto x = TE< 5 >::variable( 0.7 );
    TE< 5 > r = tan( atan( x ) );
    TE< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Atan, RoundTripReverse )
{
    // atan(tan(x)) = x  (for |x| < pi/2)
    auto x = TE< 5 >::variable( 0.5 );
    TE< 5 > r = atan( tan( x ) );
    TE< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Atan, Bivariate )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    TEn< 3, 2 > r = atan( x );
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::atan( 1.0 ), kTol );
}

// =============================================================================
// Atan2
// =============================================================================

TEST( Atan2, ConstantArgs )
{
    TE< 3 > y{ 1.0 }, x{ 1.0 };
    TE< 3 > r = atan2( y, x );
    EXPECT_NEAR( r.value(), std::atan2( 1.0, 1.0 ), kTol );
    for ( std::size_t k = 1; k < TE< 3 >::nCoefficients; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Atan2, MatchesAtan )
{
    // atan2(y, 1) = atan(y) when x=1 (constant)
    auto y = TE< 5 >::variable( 0.7 );
    TE< 5 > one{ 1.0 };
    TE< 5 > r1 = atan2( y, one );
    TE< 5 > r2 = atan( y );
    ExpectCoeffsNear( r1, r2 );
}

TEST( Atan2, DerivativeCheckY )
{
    // d/dy atan2(y, x0) = x0 / (x0² + y²)
    constexpr double x0 = 2.0, y0 = 1.0;
    TE< 3 > xc{ x0 };
    auto y = TE< 3 >::variable( y0 );
    TE< 3 > r = atan2( y, xc );
    EXPECT_NEAR( r.value(), std::atan2( y0, x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), x0 / ( x0 * x0 + y0 * y0 ), kTol );
}

TEST( Atan2, AngleFromSinCos )
{
    // atan2(sin(t), cos(t)) = t  (for |t| < pi)
    auto t = TE< 5 >::variable( 0.7 );
    TE< 5 > r = atan2( sin( t ), cos( t ) );
    TE< 5 > expected = t;
    ExpectCoeffsNear( r, expected );
}

TEST( Atan2, Bivariate )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 2.0, 1.0 } );
    TEn< 3, 2 > r = atan2( y, x );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::atan2( 1.0, 2.0 ), kTol );
}
