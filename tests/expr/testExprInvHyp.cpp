#include "testUtils.hpp"

// =============================================================================
// Asinh
// =============================================================================

TEST( Asinh, Constant )
{
    DA< 3 > a{ 1.5 };
    DA< 3 > r = asinh( a );
    EXPECT_NEAR( r.value(), std::asinh( 1.5 ), kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Asinh, KnownSeries )
{
    // asinh(x) at x0=0: coeffs are 0, 1, 0, -1/6, 0, 3/40
    auto x = DA< 5 >::variable( 0.0 );
    DA< 5 > r = asinh( x );
    EXPECT_NEAR( r[0], 0.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
    EXPECT_NEAR( r[2], 0.0, kTol );
    EXPECT_NEAR( r[3], -1.0 / 6.0, kTol );
    EXPECT_NEAR( r[4], 0.0, kTol );
    EXPECT_NEAR( r[5], 3.0 / 40.0, kTol );
}

TEST( Asinh, DerivativeCheck )
{
    constexpr double x0 = 0.6;
    auto x = DA< 3 >::variable( x0 );
    DA< 3 > r = asinh( x );
    EXPECT_NEAR( r.value(), std::asinh( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), 1.0 / std::sqrt( 1.0 + x0 * x0 ), kTol );
}

TEST( Asinh, RoundTrip )
{
    // sinh(asinh(x)) = x
    auto x = DA< 5 >::variable( 0.9 );
    DA< 5 > r = sinh( asinh( x ) );
    DA< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Asinh, RoundTripReverse )
{
    // asinh(sinh(x)) = x
    auto x = DA< 5 >::variable( 0.7 );
    DA< 5 > r = asinh( sinh( x ) );
    DA< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Asinh, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 0.6, 0.4 } );
    DAn< 3, 2 > r = asinh( x );
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::asinh( 0.6 ), kTol );
}

// =============================================================================
// Acosh
// =============================================================================

TEST( Acosh, Constant )
{
    DA< 3 > a{ 1.5 };
    DA< 3 > r = acosh( a );
    EXPECT_NEAR( r.value(), std::acosh( 1.5 ), kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Acosh, DerivativeCheck )
{
    constexpr double x0 = 1.5;
    auto x = DA< 3 >::variable( x0 );
    DA< 3 > r = acosh( x );
    EXPECT_NEAR( r.value(), std::acosh( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), 1.0 / std::sqrt( x0 * x0 - 1.0 ), kTol );
}

TEST( Acosh, RoundTrip )
{
    // cosh(acosh(x)) = x  (for x >= 1)
    auto x = DA< 5 >::variable( 1.3 );
    DA< 5 > r = cosh( acosh( x ) );
    DA< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Acosh, RoundTripReverse )
{
    // acosh(cosh(x)) = x  (for x > 0)
    auto x = DA< 5 >::variable( 0.8 );
    DA< 5 > r = acosh( cosh( x ) );
    DA< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Acosh, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.8, 0.2 } );
    DAn< 3, 2 > r = acosh( x );
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::acosh( 1.8 ), kTol );
}

// =============================================================================
// Atanh
// =============================================================================

TEST( Atanh, Constant )
{
    DA< 3 > a{ 0.5 };
    DA< 3 > r = atanh( a );
    EXPECT_NEAR( r.value(), std::atanh( 0.5 ), kTol );
    for ( std::size_t k = 1; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Atanh, KnownSeries )
{
    // atanh(x) at x0=0: coeffs are 0, 1, 0, 1/3, 0, 1/5
    auto x = DA< 5 >::variable( 0.0 );
    DA< 5 > r = atanh( x );
    EXPECT_NEAR( r[0], 0.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
    EXPECT_NEAR( r[2], 0.0, kTol );
    EXPECT_NEAR( r[3], 1.0 / 3.0, kTol );
    EXPECT_NEAR( r[4], 0.0, kTol );
    EXPECT_NEAR( r[5], 1.0 / 5.0, kTol );
}

TEST( Atanh, DerivativeCheck )
{
    constexpr double x0 = 0.4;
    auto x = DA< 3 >::variable( x0 );
    DA< 3 > r = atanh( x );
    EXPECT_NEAR( r.value(), std::atanh( x0 ), kTol );
    EXPECT_NEAR( r.derivative( { 1 } ), 1.0 / ( 1.0 - x0 * x0 ), kTol );
}

TEST( Atanh, RoundTrip )
{
    // tanh(atanh(x)) = x
    auto x = DA< 5 >::variable( 0.6 );
    DA< 5 > r = tanh( atanh( x ) );
    DA< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Atanh, RoundTripReverse )
{
    // atanh(tanh(x)) = x
    auto x = DA< 5 >::variable( 0.7 );
    DA< 5 > r = atanh( tanh( x ) );
    DA< 5 > expected = x;
    ExpectCoeffsNear( r, expected );
}

TEST( Atanh, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 0.4, 0.6 } );
    DAn< 3, 2 > r = atanh( x );
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0 } ), std::atanh( 0.4 ), kTol );
}
