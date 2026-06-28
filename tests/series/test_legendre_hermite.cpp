#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/tax.hpp>

using tax::HermiteSeries;
using tax::LegendreExpansion;

// ---------------------------------------------------------------------------
// Reference evaluations via the three-term recurrences.
// ---------------------------------------------------------------------------
static double legendreP( int n, double x )
{
    if ( n == 0 ) return 1.0;
    double p0 = 1.0, p1 = x;
    for ( int k = 1; k < n; ++k )
    {
        double p2 = ( ( 2 * k + 1 ) * x * p1 - k * p0 ) / ( k + 1 );
        p0 = p1;
        p1 = p2;
    }
    return p1;
}
static double hermiteHe( int n, double x )
{
    if ( n == 0 ) return 1.0;
    double h0 = 1.0, h1 = x;
    for ( int k = 1; k < n; ++k )
    {
        double h2 = x * h1 - k * h0;
        h0 = h1;
        h1 = h2;
    }
    return h1;
}

// ===========================================================================
// Legendre
// ===========================================================================

TEST( Legendre, BasisElementsEvalMatchReference )
{
    constexpr int N = 8;
    // Single-mode series c = e_k evaluates to P_k.
    for ( int k = 0; k <= N; ++k )
    {
        std::array< double, N + 1 > c{};
        c[std::size_t( k )] = 1.0;
        LegendreExpansion< N > f{ c };
        for ( double x : { -0.95, -0.5, -0.1, 0.2, 0.6, 0.99 } )
            EXPECT_NEAR( f.eval( x ), legendreP( k, x ), 1e-11 ) << "k=" << k << " x=" << x;
    }
}

TEST( Legendre, VariableIsIdentity )
{
    auto x = LegendreExpansion< 6 >::variable();
    EXPECT_DOUBLE_EQ( x[1], 1.0 );
    for ( double p : { -0.8, 0.0, 0.3, 0.9 } ) EXPECT_NEAR( x.eval( p ), p, 1e-13 );
}

TEST( Legendre, ProductClosedForm )
{
    // x*x = (1/3) P_0 + (2/3) P_2.
    auto x = LegendreExpansion< 6 >::variable();
    auto sq = x * x;
    EXPECT_NEAR( sq[0], 1.0 / 3.0, 1e-13 );
    EXPECT_NEAR( sq[1], 0.0, 1e-13 );
    EXPECT_NEAR( sq[2], 2.0 / 3.0, 1e-13 );
}

TEST( Legendre, ProductMatchesPointwiseWhenInBox )
{
    // Degrees chosen so the exact product stays within the kept order.
    constexpr int N = 8;
    std::array< double, N + 1 > ca{}, cb{};
    ca[0] = 0.5;
    ca[1] = -1.0;
    ca[2] = 0.3;
    ca[3] = 0.2;  // degree 3
    cb[0] = -0.4;
    cb[1] = 0.7;
    cb[2] = -0.6;  // degree 2 -> product degree 5 <= 8
    LegendreExpansion< N > a{ ca }, b{ cb };
    auto p = a * b;
    for ( double x : { -0.9, -0.3, 0.1, 0.5, 0.85 } )
        EXPECT_NEAR( p.eval( x ), a.eval( x ) * b.eval( x ), 1e-11 ) << "x=" << x;
}

TEST( Legendre, DerivativeClosedFormAndNumeric )
{
    // d/dx P_2 = 3 P_1.
    {
        std::array< double, 5 > c{ 0, 0, 1, 0, 0 };
        LegendreExpansion< 4 > f{ c };
        auto d = f.deriv();
        EXPECT_NEAR( d[1], 3.0, 1e-12 );
        EXPECT_NEAR( d[0], 0.0, 1e-12 );
        EXPECT_NEAR( d[2], 0.0, 1e-12 );
    }
    // General: compare against a central difference.
    std::array< double, 7 > c{ 0.2, -0.5, 0.9, 0.1, -0.3, 0.4, 0.05 };
    LegendreExpansion< 6 > f{ c };
    auto d = f.deriv();
    const double h = 1e-6;
    for ( double x : { -0.7, -0.2, 0.3, 0.6 } )
    {
        double num = ( f.eval( x + h ) - f.eval( x - h ) ) / ( 2 * h );
        EXPECT_NEAR( d.eval( x ), num, 1e-5 ) << "x=" << x;
    }
}

TEST( Legendre, IntegralInvertsDerivative )
{
    std::array< double, 7 > c{ 0.0, 0.5,  -1.0, 2.0,
                               0.3, -0.2, 0.0 };  // top mode 0: integ stays in box
    LegendreExpansion< 6 > f{ c };
    auto rt = f.integ().deriv();
    for ( int k = 0; k <= 6; ++k ) EXPECT_NEAR( rt[std::size_t( k )], c[std::size_t( k )], 1e-11 );
}

TEST( Legendre, LinearOpsAndPow )
{
    auto x = LegendreExpansion< 8 >::variable();
    auto f = 2.0 + 3.0 * x;
    EXPECT_NEAR( f.eval( 0.4 ), 2.0 + 3.0 * 0.4, 1e-13 );
    auto g = pow( x, 3 );  // x^3
    for ( double p : { -0.8, 0.2, 0.7 } ) EXPECT_NEAR( g.eval( p ), p * p * p, 1e-11 );
}

TEST( Legendre, Multivariate )
{
    constexpr int N = 6;
    auto x = LegendreExpansion< N, 2 >::variable< 0 >();
    auto y = LegendreExpansion< N, 2 >::variable< 1 >();
    auto f = 1.0 + 2.0 * x + y - 0.5 * ( x * y );  // degree 2
    auto g = 0.5 - x + 0.3 * y;                    // degree 1 -> product degree 3 <= N
    auto h = f * g;
    for ( auto pt : { std::array< double, 2 >{ 0.3, -0.5 }, std::array< double, 2 >{ -0.7, 0.6 },
                      std::array< double, 2 >{ 0.9, 0.2 } } )
        EXPECT_NEAR( h.eval( pt ), f.eval( pt ) * g.eval( pt ), 1e-11 );

    auto fx = f.deriv< 0 >();
    auto fy = f.deriv< 1 >();
    const double hh = 1e-6;
    std::array< double, 2 > pt{ 0.2, -0.3 };
    double nx = ( f.eval( { pt[0] + hh, pt[1] } ) - f.eval( { pt[0] - hh, pt[1] } ) ) / ( 2 * hh );
    double ny = ( f.eval( { pt[0], pt[1] + hh } ) - f.eval( { pt[0], pt[1] - hh } ) ) / ( 2 * hh );
    EXPECT_NEAR( fx.eval( pt ), nx, 1e-6 );
    EXPECT_NEAR( fy.eval( pt ), ny, 1e-6 );
}

// ===========================================================================
// Hermite (probabilists')
// ===========================================================================

TEST( Hermite, BasisElementsEvalMatchReference )
{
    constexpr int N = 8;
    for ( int k = 0; k <= N; ++k )
    {
        std::array< double, N + 1 > c{};
        c[std::size_t( k )] = 1.0;
        HermiteSeries< N > f{ c };
        for ( double x : { -2.0, -0.7, 0.0, 0.5, 1.3, 2.4 } )
            EXPECT_NEAR( f.eval( x ), hermiteHe( k, x ), 1e-9 ) << "k=" << k << " x=" << x;
    }
}

TEST( Hermite, ProductClosedForm )
{
    // He_1 * He_1 = x*x = He_2 + He_0   (He_2 = x^2 - 1).
    auto x = HermiteSeries< 6 >::variable();
    auto sq = x * x;
    EXPECT_NEAR( sq[0], 1.0, 1e-13 );
    EXPECT_NEAR( sq[1], 0.0, 1e-13 );
    EXPECT_NEAR( sq[2], 1.0, 1e-13 );
}

TEST( Hermite, ProductMatchesPointwiseWhenInBox )
{
    constexpr int N = 8;
    std::array< double, N + 1 > ca{}, cb{};
    ca[0] = 0.3;
    ca[1] = 1.0;
    ca[2] = -0.5;
    ca[3] = 0.2;
    cb[0] = -0.6;
    cb[1] = 0.4;
    cb[2] = 0.7;
    HermiteSeries< N > a{ ca }, b{ cb };
    auto p = a * b;
    for ( double x : { -1.5, -0.4, 0.3, 1.1 } )
        EXPECT_NEAR( p.eval( x ), a.eval( x ) * b.eval( x ), 1e-9 ) << "x=" << x;
}

TEST( Hermite, DerivativeClosedFormAndNumeric )
{
    // He_n' = n He_{n-1}: d/dx He_3 = 3 He_2.
    {
        std::array< double, 5 > c{ 0, 0, 0, 1, 0 };
        HermiteSeries< 4 > f{ c };
        auto d = f.deriv();
        EXPECT_NEAR( d[2], 3.0, 1e-12 );
        EXPECT_NEAR( d[0], 0.0, 1e-12 );
    }
    std::array< double, 7 > c{ 0.2, -0.5, 0.9, 0.1, -0.3, 0.4, 0.05 };
    HermiteSeries< 6 > f{ c };
    auto d = f.deriv();
    const double h = 1e-6;
    for ( double x : { -1.2, -0.2, 0.6, 1.4 } )
    {
        double num = ( f.eval( x + h ) - f.eval( x - h ) ) / ( 2 * h );
        EXPECT_NEAR( d.eval( x ), num, 1e-4 ) << "x=" << x;
    }
}

TEST( Hermite, IntegralInvertsDerivative )
{
    std::array< double, 7 > c{ 0.0, 0.5,  -1.0, 2.0,
                               0.3, -0.2, 0.0 };  // top mode 0: integ stays in box
    HermiteSeries< 6 > f{ c };
    auto rt = f.integ().deriv();
    for ( int k = 0; k <= 6; ++k ) EXPECT_NEAR( rt[std::size_t( k )], c[std::size_t( k )], 1e-11 );
}

TEST( Hermite, Multivariate )
{
    constexpr int N = 6;
    auto x = HermiteSeries< N, 2 >::variable< 0 >();
    auto y = HermiteSeries< N, 2 >::variable< 1 >();
    auto f = 1.0 + x + 2.0 * y + 0.5 * ( x * y );  // degree 2
    auto g = -0.3 + 0.8 * x - y;                   // degree 1
    auto h = f * g;
    for ( auto pt : { std::array< double, 2 >{ 0.5, -1.0 }, std::array< double, 2 >{ -1.2, 0.7 },
                      std::array< double, 2 >{ 1.1, 1.3 } } )
        EXPECT_NEAR( h.eval( pt ), f.eval( pt ) * g.eval( pt ), 1e-9 );
}

TEST( Hermite, OrthogonalProductIsConstexpr )
{
    constexpr double v = [] {
        auto x = HermiteSeries< 4 >::variable();
        auto sq = x * x;        // He_2 + He_0
        return sq.eval( 2.0 );  // 2^2 = 4
    }();
    static_assert( v == 4.0 );
    EXPECT_DOUBLE_EQ( v, 4.0 );
}
