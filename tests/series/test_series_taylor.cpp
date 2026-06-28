#include <gtest/gtest.h>

#include <cmath>
#include <tax/tax.hpp>

using tax::TE;

TEST( SeriesTaylor, VariableAndEval )
{
    auto x = TE< 5 >::variable();
    EXPECT_DOUBLE_EQ( x[0], 0.0 );
    EXPECT_DOUBLE_EQ( x[1], 1.0 );
    EXPECT_DOUBLE_EQ( x.eval( 0.7 ), 0.7 );
}

TEST( SeriesTaylor, ConstantAndArithmetic )
{
    auto x = TE< 4 >::variable();
    auto f = 2.0 + 3.0 * x;  // 2 + 3x
    EXPECT_DOUBLE_EQ( f[0], 2.0 );
    EXPECT_DOUBLE_EQ( f[1], 3.0 );
    EXPECT_DOUBLE_EQ( f.eval( 2.0 ), 8.0 );

    auto g = f - 2.0;
    EXPECT_DOUBLE_EQ( g[0], 0.0 );
}

TEST( SeriesTaylor, ProductIsCauchy )
{
    auto x = TE< 4 >::variable();
    auto f = ( 1.0 + x ) * ( 1.0 + x );  // 1 + 2x + x^2
    EXPECT_DOUBLE_EQ( f[0], 1.0 );
    EXPECT_DOUBLE_EQ( f[1], 2.0 );
    EXPECT_DOUBLE_EQ( f[2], 1.0 );
    EXPECT_DOUBLE_EQ( f[3], 0.0 );
}

TEST( SeriesTaylor, Division )
{
    auto x = TE< 5 >::variable();
    auto f = 1.0 / ( 1.0 - x );  // geometric series 1 + x + x^2 + ...
    for ( std::size_t k = 0; k <= 5; ++k ) EXPECT_NEAR( f[k], 1.0, 1e-12 );
}

TEST( SeriesTaylor, DerivIntegRoundTrip )
{
    auto x = TE< 5 >::variable();
    auto f = 1.0 + 2.0 * x + 3.0 * ( x * x );  // 1 + 2x + 3x^2
    auto df = f.deriv();                       // 2 + 6x
    EXPECT_DOUBLE_EQ( df[0], 2.0 );
    EXPECT_DOUBLE_EQ( df[1], 6.0 );
    EXPECT_DOUBLE_EQ( df[2], 0.0 );

    auto F = df.integ();  // back to 2x + 3x^2 (constant of integration 0)
    EXPECT_DOUBLE_EQ( F[0], 0.0 );
    EXPECT_DOUBLE_EQ( F[1], 2.0 );
    EXPECT_DOUBLE_EQ( F[2], 3.0 );
}

TEST( SeriesTaylor, TranscendentalsMatchClosedForm )
{
    auto x = TE< 6 >::variable();
    auto e = exp( x );
    // exp(x) = sum x^k / k!
    double fact = 1.0;
    for ( int k = 0; k <= 6; ++k )
    {
        if ( k > 0 ) fact *= k;
        EXPECT_NEAR( e[std::size_t( k )], 1.0 / fact, 1e-12 );
    }

    auto s = sin( x );
    EXPECT_NEAR( s[1], 1.0, 1e-12 );
    EXPECT_NEAR( s[3], -1.0 / 6.0, 1e-12 );
    EXPECT_NEAR( s[5], 1.0 / 120.0, 1e-12 );
}

TEST( SeriesTaylor, ReusesExistingKernelEval )
{
    // sin(x)*exp(x) evaluated against the host math at a small displacement.
    auto x = TE< 10 >::variable();
    auto f = sin( x ) * exp( x );
    const double dx = 0.2;
    EXPECT_NEAR( f.eval( dx ), std::sin( dx ) * std::exp( dx ), 1e-9 );
}
