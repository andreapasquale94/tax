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
    r = (tax::exp( x )).eval();
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
    r = (tax::log( x )).eval();
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
    r = (tax::sin( x )).eval();
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
    r = (tax::cos( x )).eval();
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
    sh = (tax::sinh( x )).eval();
    ch = (tax::cosh( x )).eval();
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
    r = (tax::sqrt( x )).eval();
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
    a = (tax::square( x )).eval();
    b = (x * x).eval();
    for ( Eigen::Index i = 0; i < a.coeffs().size(); ++i )
    {
        EXPECT_NEAR( a.coeffs()( i ), b.coeffs()( i ), kTol );
    }
}

TEST( Math, CubeAgreesWithMul )
{
    auto x = TE< 5 >::variable( 0.5 );
    TE< 5 > a, b;
    a = (tax::cube( x )).eval();
    b = (x * x * x).eval();
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
    r = (tax::square( s ) + tax::square( c )).eval();
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
    r = (tax::log( tax::exp( x ) )).eval();
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
    r = (tax::sin( x * y )).eval();
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
    result = (f).eval();

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

TEST( Math, SincosPairAgreesWithSeparateCalls )
{
    auto x = TE< 5 >::variable( 0.7 );
    auto pair = tax::sincos( x );

    TE< 5 > s_pair, c_pair;
    s_pair = (pair.sin()).eval();
    c_pair = pair.cos().eval();  // shares the work already done above

    TE< 5 > s_solo, c_solo;
    s_solo = (tax::sin( x )).eval();
    c_solo = (tax::cos( x )).eval();

    for ( Eigen::Index i = 0; i < s_pair.coeffs().size(); ++i )
    {
        EXPECT_NEAR( s_pair.coeffs()( i ), s_solo.coeffs()( i ), kTol );
        EXPECT_NEAR( c_pair.coeffs()( i ), c_solo.coeffs()( i ), kTol );
    }
}

TEST( Math, SinhcoshPairAgreesWithSeparateCalls )
{
    auto x = TE< 4 >::variable( 0.3 );
    auto pair = tax::sinhcosh( x );

    TE< 4 > s_pair, c_pair;
    s_pair = (pair.sinh()).eval();
    c_pair = (pair.cosh()).eval();

    TE< 4 > s_solo, c_solo;
    s_solo = (tax::sinh( x )).eval();
    c_solo = (tax::cosh( x )).eval();

    for ( Eigen::Index i = 0; i < s_pair.coeffs().size(); ++i )
    {
        EXPECT_NEAR( s_pair.coeffs()( i ), s_solo.coeffs()( i ), kTol );
        EXPECT_NEAR( c_pair.coeffs()( i ), c_solo.coeffs()( i ), kTol );
    }
}

TEST( Math, AtanAtZero )
{
    // atan(x) = x - x^3/3 + x^5/5 - x^7/7
    auto x = TE< 7 >::variable( 0.0 );
    TE< 7 > r;
    r = (tax::atan( x )).eval();
    EXPECT_NEAR( r.coeffs()( 0 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), -1.0 / 3.0, kTol );
    EXPECT_NEAR( r.coeffs()( 5 ), 1.0 / 5.0, kTol );
    EXPECT_NEAR( r.coeffs()( 7 ), -1.0 / 7.0, kTol );
}

TEST( Math, AtanhAtZero )
{
    // atanh(x) = x + x^3/3 + x^5/5
    auto x = TE< 5 >::variable( 0.0 );
    TE< 5 > r;
    r = (tax::atanh( x )).eval();
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), 1.0 / 3.0, kTol );
    EXPECT_NEAR( r.coeffs()( 5 ), 1.0 / 5.0, kTol );
}

TEST( Math, AsinAtZero )
{
    // asin(x) = x + x^3/6 + 3 x^5/40
    auto x = TE< 5 >::variable( 0.0 );
    TE< 5 > r;
    r = (tax::asin( x )).eval();
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), 1.0 / 6.0, kTol );
    EXPECT_NEAR( r.coeffs()( 5 ), 3.0 / 40.0, kTol );
}

TEST( Math, AcosAtZero )
{
    // acos(x) = pi/2 - asin(x)
    auto x = TE< 5 >::variable( 0.0 );
    TE< 5 > r;
    r = (tax::acos( x )).eval();
    EXPECT_NEAR( r.coeffs()( 0 ), M_PI / 2.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), -1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), -1.0 / 6.0, kTol );
}

TEST( Math, AsinhAtZero )
{
    // asinh(x) = x - x^3/6 + 3 x^5 / 40
    auto x = TE< 5 >::variable( 0.0 );
    TE< 5 > r;
    r = (tax::asinh( x )).eval();
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), -1.0 / 6.0, kTol );
    EXPECT_NEAR( r.coeffs()( 5 ), 3.0 / 40.0, kTol );
}

TEST( Math, AcoshAroundTwo )
{
    // acosh near u_0 = 2: F'(u) = 1/sqrt(u^2 - 1).  Verify by composing
    // cosh(acosh(x)) ~ x.
    auto x = TE< 4 >::variable( 2.0 );
    TE< 4 > inv;
    inv = (tax::cosh( tax::acosh( x ) )).eval();
    EXPECT_NEAR( inv.value(), 2.0, 1e-10 );
    EXPECT_NEAR( inv.coeffs()( 1 ), 1.0, 1e-10 );
}

TEST( Math, Log10AtTen )
{
    auto x = TE< 4 >::variable( 10.0 );
    TE< 4 > r;
    r = (tax::log10( x )).eval();
    EXPECT_NEAR( r.value(), 1.0, kTol );
    // d/dx log10(x) = 1 / (x ln 10) = 1/(10 ln 10).
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0 / ( 10.0 * std::log( 10.0 ) ), kTol );
}

TEST( Math, CbrtAroundOne )
{
    // cbrt(1+u) = 1 + u/3 - u^2/9 + 5u^3/81 - ...
    auto x = TE< 4 >::variable( 1.0 );
    TE< 4 > r;
    r = (tax::cbrt( x )).eval();
    EXPECT_NEAR( r.coeffs()( 0 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0 / 3.0, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), -1.0 / 9.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), 5.0 / 81.0, kTol );

    // Composition: cbrt(x)^3 == x.
    TE< 4 > cube_of_root;
    cube_of_root = (tax::cube( tax::cbrt( x ) )).eval();
    for ( Eigen::Index i = 0; i < x.coeffs().size(); ++i )
    {
        EXPECT_NEAR( cube_of_root.coeffs()( i ), x.coeffs()( i ), 1e-10 );
    }
}

TEST( Math, PowIntCompileTime )
{
    auto x = TE< 5 >::variable( 0.5 );
    TE< 5 > r0, r2, r5, rNeg;
    r0 = (tax::pow< 0 >( x )).eval();
    r2 = (tax::pow< 2 >( x )).eval();
    r5 = (tax::pow< 5 >( x )).eval();
    rNeg = (tax::pow< -2 >( x )).eval();

    // pow<0> = 1 (constant).
    EXPECT_NEAR( r0.value(), 1.0, kTol );
    for ( Eigen::Index i = 1; i < r0.coeffs().size(); ++i )
    {
        EXPECT_NEAR( r0.coeffs()( i ), 0.0, kTol );
    }

    // pow<2>(x) == x*x.
    TE< 5 > r2_ref;
    r2_ref = (x * x).eval();
    for ( Eigen::Index i = 0; i < r2.coeffs().size(); ++i )
    {
        EXPECT_NEAR( r2.coeffs()( i ), r2_ref.coeffs()( i ), kTol );
    }

    // pow<5>(x) == x*x*x*x*x.
    TE< 5 > r5_ref;
    r5_ref = (x * x * x * x * x).eval();
    for ( Eigen::Index i = 0; i < r5.coeffs().size(); ++i )
    {
        EXPECT_NEAR( r5.coeffs()( i ), r5_ref.coeffs()( i ), kTol );
    }

    // pow<-2>(x) == 1 / x^2.
    TE< 5 > rNeg_ref;
    rNeg_ref = (TE< 5 >::one() / ( x * x )).eval();
    for ( Eigen::Index i = 0; i < rNeg.coeffs().size(); ++i )
    {
        EXPECT_NEAR( rNeg.coeffs()( i ), rNeg_ref.coeffs()( i ), 1e-10 );
    }
}

TEST( Math, PowRealMatchesIntegerWhenInteger )
{
    auto x = TE< 5 >::variable( 1.5 );
    TE< 5 > r_real, r_int;
    r_real = (tax::pow( x, 3.0 )).eval();
    r_int = (tax::pow< 3 >( x )).eval();
    for ( Eigen::Index i = 0; i < r_real.coeffs().size(); ++i )
    {
        EXPECT_NEAR( r_real.coeffs()( i ), r_int.coeffs()( i ), 1e-10 );
    }
}

TEST( Math, PowRealHalfIsSqrt )
{
    auto x = TE< 5 >::variable( 4.0 );
    TE< 5 > a, b;
    a = (tax::pow( x, 0.5 )).eval();
    b = (tax::sqrt( x )).eval();
    for ( Eigen::Index i = 0; i < a.coeffs().size(); ++i )
    {
        EXPECT_NEAR( a.coeffs()( i ), b.coeffs()( i ), 1e-10 );
    }
}

TEST( Math, HypotTwoArg )
{
    auto [ x, y ] = TEn< 4, 2 >::variables( std::array< double, 2 >{ 3.0, 4.0 } );
    TEn< 4, 2 > r;
    r = (tax::hypot( x, y )).eval();
    EXPECT_NEAR( r.value(), 5.0, 1e-10 );
}

TEST( Math, HypotThreeArg )
{
    auto [ x, y, z ] = TEn< 4, 3 >::variables( std::array< double, 3 >{ 1.0, 2.0, 2.0 } );
    TEn< 4, 3 > r;
    r = (tax::hypot( x, y, z )).eval();
    EXPECT_NEAR( r.value(), 3.0, 1e-10 );
}

TEST( Math, Atan2InAllQuadrants )
{
    for ( auto [ y0, x0 ] : { std::pair{ 1.0, 1.0 }, std::pair{ 1.0, -1.0 },
                              std::pair{ -1.0, -1.0 }, std::pair{ -1.0, 1.0 } } )
    {
        auto [ y, x ] = TEn< 3, 2 >::variables( std::array< double, 2 >{ y0, x0 } );
        TEn< 3, 2 > r;
        r = (tax::atan2( y, x )).eval();
        EXPECT_NEAR( r.value(), std::atan2( y0, x0 ), 1e-10 );
    }
}

TEST( Math, ErfAtZero )
{
    // erf(x) = (2/sqrt(pi)) (x - x^3/3 + x^5/10 - x^7/42 + ...)
    auto x = TE< 5 >::variable( 0.0 );
    TE< 5 > r;
    r = (tax::erf( x )).eval();
    const double k = 2.0 / std::sqrt( M_PI );
    EXPECT_NEAR( r.coeffs()( 0 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), k, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), -k / 3.0, kTol );
    EXPECT_NEAR( r.coeffs()( 5 ), k / 10.0, kTol );
}

TEST( Math, ErfAgreesWithFunctionAtCentre )
{
    auto x = TE< 6 >::variable( 0.4 );
    TE< 6 > r;
    r = (tax::erf( x )).eval();
    EXPECT_NEAR( r.value(), std::erf( 0.4 ), 1e-12 );
}

TEST( Math, EvalAgreesWithFunction )
{
    // f(x) = exp(x); at x0 = 0, eval at dx = 0.3 should be ~ exp(0.3).
    auto x = TE< 8 >::variable( 0.0 );
    TE< 8 > r;
    r = (tax::exp( x )).eval();
    std::array< double, 1 > dx{ 0.3 };
    EXPECT_NEAR( r.eval( dx ), std::exp( 0.3 ), 1e-9 );
}
