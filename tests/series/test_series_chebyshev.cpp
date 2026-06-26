#include <gtest/gtest.h>

#include <cmath>
#include <tax/tax.hpp>

using tax::ChebyshevSeries;

// Helper: evaluate T_k at x via the recurrence (reference).
static double Tk( int k, double x )
{
    double t0 = 1.0, t1 = x;
    if ( k == 0 ) return t0;
    if ( k == 1 ) return t1;
    for ( int i = 2; i <= k; ++i )
    {
        double t = 2.0 * x * t1 - t0;
        t0 = t1;
        t1 = t;
    }
    return t1;
}

TEST( SeriesChebyshev, VariableEvalIsIdentity )
{
    auto x = ChebyshevSeries< 5 >::variable();
    EXPECT_DOUBLE_EQ( x[1], 1.0 );
    for ( double p : { -0.9, -0.3, 0.0, 0.4, 0.85 } ) EXPECT_NEAR( x.eval( p ), p, 1e-13 );
}

TEST( SeriesChebyshev, ProductFoldsModes )
{
    // x * x = (T_0 + T_2)/2 == x^2.
    auto x = ChebyshevSeries< 4 >::variable();
    auto sq = x * x;
    EXPECT_NEAR( sq[0], 0.5, 1e-13 );
    EXPECT_NEAR( sq[1], 0.0, 1e-13 );
    EXPECT_NEAR( sq[2], 0.5, 1e-13 );
    for ( double p : { -0.7, 0.1, 0.6 } ) EXPECT_NEAR( sq.eval( p ), p * p, 1e-13 );
}

TEST( SeriesChebyshev, ClenshawMatchesReference )
{
    // f = 1 + 2 T_1 + 3 T_2 - T_3
    std::array< double, 5 > c{ 1.0, 2.0, 3.0, -1.0, 0.0 };
    ChebyshevSeries< 4 > f{ c };
    for ( double x : { -0.8, -0.2, 0.3, 0.9 } )
    {
        double ref = 0.0;
        for ( int k = 0; k <= 4; ++k ) ref += c[std::size_t( k )] * Tk( k, x );
        EXPECT_NEAR( f.eval( x ), ref, 1e-12 );
    }
}

TEST( SeriesChebyshev, DerivativeClosedForms )
{
    // d/dx T_1 = 1 = T_0
    {
        auto f = ChebyshevSeries< 3 >::variable();  // T_1
        auto d = f.deriv();
        EXPECT_NEAR( d[0], 1.0, 1e-13 );
        EXPECT_NEAR( d[1], 0.0, 1e-13 );
    }
    // d/dx T_2 = 4 T_1   (T_2 = 2x^2-1, derivative 4x)
    {
        std::array< double, 4 > c{ 0.0, 0.0, 1.0, 0.0 };
        ChebyshevSeries< 3 > f{ c };
        auto d = f.deriv();
        EXPECT_NEAR( d[0], 0.0, 1e-13 );
        EXPECT_NEAR( d[1], 4.0, 1e-13 );
    }
    // d/dx T_3 = 3 T_0 + 6 T_2
    {
        std::array< double, 4 > c{ 0.0, 0.0, 0.0, 1.0 };
        ChebyshevSeries< 3 > f{ c };
        auto d = f.deriv();
        EXPECT_NEAR( d[0], 3.0, 1e-13 );
        EXPECT_NEAR( d[1], 0.0, 1e-13 );
        EXPECT_NEAR( d[2], 6.0, 1e-13 );
    }
}

TEST( SeriesChebyshev, IntegralIsDerivativeInverse )
{
    // Integrate then differentiate recovers the original (constant term aside).
    std::array< double, 6 > c{ 0.0, 0.5, -1.0, 2.0, 0.3, 0.0 };
    ChebyshevSeries< 5 > f{ c };
    auto roundtrip = f.integ().deriv();
    for ( int k = 0; k <= 5; ++k )
        EXPECT_NEAR( roundtrip[std::size_t( k )], c[std::size_t( k )], 1e-12 );
}

TEST( SeriesChebyshev, DerivativeMatchesNumeric )
{
    std::array< double, 6 > c{ 0.2, -0.4, 1.1, 0.5, -0.7, 0.3 };
    ChebyshevSeries< 5 > f{ c };
    auto d = f.deriv();
    const double h = 1e-6;
    for ( double x : { -0.6, 0.1, 0.5 } )
    {
        double num = ( f.eval( x + h ) - f.eval( x - h ) ) / ( 2 * h );
        EXPECT_NEAR( d.eval( x ), num, 1e-5 );
    }
}
