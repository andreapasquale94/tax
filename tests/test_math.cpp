// SPDX-License-Identifier: BSD-3-Clause
//
// Math kernel tests: sin, cos, exp, log, sqrt, square, cube.

#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "tax/tax.hpp"

using tax::TE;
using tax::TEn;

static constexpr double kTol = 1e-11;

TEST( Math, ExpAtZero )
{
    // exp(x) = 1 + x + x^2/2 + x^3/6 + x^4/24
    auto x = TE< 4 >::variable( 0.0 );
    TE< 4 > r;
    r <<= tax::exp( x );
    EXPECT_NEAR( r.coeffs()( 0 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), 0.5, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), 1.0 / 6.0, kTol );
    EXPECT_NEAR( r.coeffs()( 4 ), 1.0 / 24.0, kTol );
}

TEST( Math, LogAtOne )
{
    // log(1 + x) = x - x^2/2 + x^3/3 - x^4/4
    // i.e. log(u) at u0=1: coeffs of degree 1..4 are 1, -1/2, 1/3, -1/4.
    auto x = TE< 4 >::variable( 1.0 );
    TE< 4 > r;
    r <<= tax::log( x );
    EXPECT_NEAR( r.coeffs()( 0 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), -0.5, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), 1.0 / 3.0, kTol );
    EXPECT_NEAR( r.coeffs()( 4 ), -0.25, kTol );
}

TEST( Math, SinAtZero )
{
    // sin(x) = x - x^3/6 + x^5/120
    auto x = TE< 5 >::variable( 0.0 );
    TE< 5 > r;
    r <<= tax::sin( x );
    EXPECT_NEAR( r.coeffs()( 0 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), -1.0 / 6.0, kTol );
    EXPECT_NEAR( r.coeffs()( 4 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 5 ), 1.0 / 120.0, kTol );
}

TEST( Math, CosAtZero )
{
    // cos(x) = 1 - x^2/2 + x^4/24
    auto x = TE< 4 >::variable( 0.0 );
    TE< 4 > r;
    r <<= tax::cos( x );
    EXPECT_NEAR( r.coeffs()( 0 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), -0.5, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 4 ), 1.0 / 24.0, kTol );
}

TEST( Math, SinhCoshAtZero )
{
    auto x = TE< 4 >::variable( 0.0 );
    TE< 4 > sh, ch;
    sh <<= tax::sinh( x );
    ch <<= tax::cosh( x );
    // sinh: x + x^3/6 + ...
    EXPECT_NEAR( sh.coeffs()( 0 ), 0.0, kTol );
    EXPECT_NEAR( sh.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( sh.coeffs()( 3 ), 1.0 / 6.0, kTol );
    // cosh: 1 + x^2/2 + x^4/24
    EXPECT_NEAR( ch.coeffs()( 0 ), 1.0, kTol );
    EXPECT_NEAR( ch.coeffs()( 2 ), 0.5, kTol );
    EXPECT_NEAR( ch.coeffs()( 4 ), 1.0 / 24.0, kTol );
}

TEST( Math, SqrtAroundOne )
{
    // sqrt(1 + x) = 1 + x/2 - x^2/8 + x^3/16 - 5 x^4/128
    auto x = TE< 4 >::variable( 1.0 );
    TE< 4 > r;
    r <<= tax::sqrt( x );
    EXPECT_NEAR( r.coeffs()( 0 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 0.5, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), -1.0 / 8.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), 1.0 / 16.0, kTol );
    EXPECT_NEAR( r.coeffs()( 4 ), -5.0 / 128.0, kTol );
}

TEST( Math, SquareIsSelfMultiply )
{
    auto x = TE< 4 >::variable( 1.5 );
    TE< 4 > a, b;
    a <<= tax::square( x );
    b <<= x * x;
    for ( Eigen::Index i = 0; i < a.coeffs().size(); ++i )
    {
        EXPECT_NEAR( a.coeffs()( i ), b.coeffs()( i ), kTol );
    }
}

TEST( Math, CubeAgreesWithMul )
{
    auto x = TE< 5 >::variable( 0.5 );
    TE< 5 > a, b;
    a <<= tax::cube( x );
    b <<= x * x * x;
    for ( Eigen::Index i = 0; i < a.coeffs().size(); ++i )
    {
        EXPECT_NEAR( a.coeffs()( i ), b.coeffs()( i ), kTol );
    }
}

TEST( Math, SinSquaredPlusCosSquared )
{
    // sin^2 + cos^2 = 1 (constant), all higher coeffs zero.
    auto x = TE< 5 >::variable( 0.7 );
    TE< 5 > r;
    auto s = tax::sin( x );
    auto c = tax::cos( x );
    r <<= tax::square( s ) + tax::square( c );
    EXPECT_NEAR( r.coeffs()( 0 ), 1.0, kTol );
    for ( Eigen::Index i = 1; i < r.coeffs().size(); ++i )
    {
        EXPECT_NEAR( r.coeffs()( i ), 0.0, kTol ) << "coeff " << i;
    }
}

TEST( Math, ExpLogIdentity )
{
    auto x = TE< 4 >::variable( 0.5 );
    TE< 4 > r;
    r <<= tax::log( tax::exp( x ) );
    EXPECT_NEAR( r.coeffs()( 0 ), 0.5, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0, kTol );
    for ( Eigen::Index i = 2; i < r.coeffs().size(); ++i )
    {
        EXPECT_NEAR( r.coeffs()( i ), 0.0, kTol );
    }
}

TEST( Math, MultivariateSinCosCircle )
{
    // f(x, y) = sin(x*y); evaluate around (0, 0) — series should match
    // sin(0) + 0 contributions, so coefficient of (1,1) should be 1
    // (since sin(xy) ~ xy for small).
    auto [ x, y ] = TEn< 2, 2 >::variables( std::array< double, 2 >{ 0.0, 0.0 } );
    TEn< 2, 2 > r;
    r <<= tax::sin( x * y );
    std::array< std::size_t, 2 > a00{ 0, 0 };
    std::array< std::size_t, 2 > a11{ 1, 1 };
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a00 ) ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a11 ) ), 1.0, kTol );
}

TEST( Math, BriefExampleEndToEnd )
{
    // Mirrors the API example from the project brief.
    auto x = TE< 5 >::variable( 1.0 );
    auto [ u, v ] = TEn< 3, 2 >::variables( std::array< double, 2 >{ 1.0, 2.0 } );
    auto x_const = TE< 5 >::constant( 3.0 );
    auto z = TE< 5 >::zero();
    ( void ) x;
    ( void ) x_const;
    ( void ) z;

    auto f = u * tax::sin( v ) + u * v;
    TEn< 3, 2 > result;
    result <<= f;

    // f(0, 0) = 1 * sin(2) + 1 * 2
    EXPECT_NEAR( result.value(), std::sin( 2.0 ) + 2.0, kTol );

    // df/du at (0,0) = sin(2) + 2
    std::array< std::size_t, 2 > a10{ 1, 0 };
    EXPECT_NEAR( result.derivative( std::span< const std::size_t >( a10 ) ),
                 std::sin( 2.0 ) + 2.0, kTol );

    // df/dv at (0,0) = u*cos(2) + u = cos(2) + 1
    std::array< std::size_t, 2 > a01{ 0, 1 };
    EXPECT_NEAR( result.derivative( std::span< const std::size_t >( a01 ) ),
                 std::cos( 2.0 ) + 1.0, kTol );
}

TEST( Math, EvalAgreesWithFunction )
{
    // f(x) = exp(x); at x0 = 0, eval at dx = 0.3 should be ~ exp(0.3).
    auto x = TE< 8 >::variable( 0.0 );
    TE< 8 > r;
    r <<= tax::exp( x );
    std::array< double, 1 > dx{ 0.3 };
    EXPECT_NEAR( r.eval( dx ), std::exp( 0.3 ), 1e-9 );
}
