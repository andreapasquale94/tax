#include <gtest/gtest.h>

#include <cmath>

#include "modelTestUtils.hpp"

using I = tax::Interval< double >;

// ---------------------------------------------------------------------------
// The worked example of §4.4.1: f(x) = 1/x + x, order 3, x0 = 2, [1.9, 2.1]
// ---------------------------------------------------------------------------

TEST( TMMath, Makino441ReciprocalPlusIdentity )
{
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    const auto f = 1.0 / x + x;

    // (4.16): P = 2.5 + 0.75 (x-2) + 0.125 (x-2)^2 - 0.0625 (x-2)^3.
    EXPECT_NEAR( f.polynomial()[0], 2.5, 1e-13 );
    EXPECT_NEAR( f.polynomial()[1], 0.75, 1e-13 );
    EXPECT_NEAR( f.polynomial()[2], 0.125, 1e-13 );
    EXPECT_NEAR( f.polynomial()[3], -0.0625, 1e-13 );

    // (4.15): remainder = [0, 4.038e-6] (up to outward-rounding ulps).
    EXPECT_GE( f.remainder().lower(), -1e-12 );
    EXPECT_NEAR( f.remainder().upper(), 4.0386107e-6, 1e-9 );

    // Total bound: encloses the exact range [2.42631, 2.57619] (Table 4.3)
    // and stays close to it despite the naive order-sum bounder.
    const auto b = f.bound();
    EXPECT_LE( b.lower(), 2.426312 );
    EXPECT_GE( b.upper(), 2.576185 );
    EXPECT_LT( b.width(), 0.152 );

    tax::test::ExpectEncloses( f, []( const auto& p ) { return 1.0 / p[0] + p[0]; }, 41 );
}

namespace
{

template < int N >
double remainderWidthOrderN()
{
    const auto x = tax::TM< N >::variable( 2.0, I{ 1.9, 2.1 } );
    const auto f = 1.0 / x + x;
    return f.remainder().width();
}

}  // namespace

TEST( TMMath, Table42RemainderShrinksWithOrder )
{
    const std::array< double, 6 > w{ remainderWidthOrderN< 1 >(), remainderWidthOrderN< 2 >(),
                                     remainderWidthOrderN< 3 >(), remainderWidthOrderN< 4 >(),
                                     remainderWidthOrderN< 5 >(), remainderWidthOrderN< 6 >() };

    for ( std::size_t i = 0; i + 1 < w.size(); ++i ) EXPECT_LT( w[i + 1], w[i] );

    // Table 4.2 oracle values (interval widths).
    EXPECT_NEAR( w[0], 1.4579384e-3, 1e-8 );
    EXPECT_NEAR( w[1], 2 * 7.6733603e-5, 1e-9 );
    EXPECT_NEAR( w[2], 4.0386107e-6, 1e-10 );
    EXPECT_NEAR( w[3], 2 * 2.1255845e-7, 1e-11 );
}

// ---------------------------------------------------------------------------
// Individual intrinsics: containment sweeps + remainder sanity
// ---------------------------------------------------------------------------

TEST( TMMath, Exp )
{
    const auto x = tax::TM< 5 >::variable( 0.0, I{ -0.5, 0.5 } );
    const auto f = exp( x );
    EXPECT_GE( f.remainder().lower(), -5e-5 );
    EXPECT_LE( f.remainder().upper(), 5e-5 );
    tax::test::ExpectEncloses( f, []( const auto& p ) { return std::exp( p[0] ); }, 41 );
}

TEST( TMMath, Log )
{
    const auto x = tax::TM< 5 >::variable( 2.0, I{ 1.5, 2.5 } );
    const auto f = log( x );
    tax::test::ExpectEncloses( f, []( const auto& p ) { return std::log( p[0] ); }, 41 );
    EXPECT_NEAR( f.value(), std::log( 2.0 ), 1e-14 );
}

TEST( TMMath, SqrtAndIsqrt )
{
    const auto x = tax::TM< 5 >::variable( 4.0, I{ 3.5, 4.5 } );
    const auto s = sqrt( x );
    tax::test::ExpectEncloses( s, []( const auto& p ) { return std::sqrt( p[0] ); }, 41 );

    const auto is = isqrt( x );
    tax::test::ExpectEncloses( is, []( const auto& p ) { return 1.0 / std::sqrt( p[0] ); }, 41 );

    // sqrt(x) * isqrt(x) must tightly enclose 1.
    const auto one = s * is;
    EXPECT_TRUE( one.bound().contains( 1.0 ) );
    EXPECT_LT( one.bound().width(), 1e-3 );
}

TEST( TMMath, Reciprocal )
{
    const auto x = tax::TM< 4 >::variable( -2.0, I{ -2.3, -1.7 } );
    const auto f = reciprocal( x );  // negative constant part exercises signs
    tax::test::ExpectEncloses( f, []( const auto& p ) { return 1.0 / p[0]; }, 41 );
}

TEST( TMMath, SinCosTan )
{
    const auto x = tax::TM< 7 >::variable( 0.5, I{ 0.2, 0.8 } );
    const auto s = sin( x );
    const auto c = cos( x );
    tax::test::ExpectEncloses( s, []( const auto& p ) { return std::sin( p[0] ); }, 41 );
    tax::test::ExpectEncloses( c, []( const auto& p ) { return std::cos( p[0] ); }, 41 );

    // Pythagoras: sin^2 + cos^2 encloses 1 tightly.
    const auto pyth = square( s ) + square( c );
    EXPECT_TRUE( pyth.bound().contains( 1.0 ) );
    EXPECT_LT( pyth.bound().width(), 1e-6 );

    const auto t = tan( x );
    tax::test::ExpectEncloses( t, []( const auto& p ) { return std::tan( p[0] ); }, 41 );
}

TEST( TMMath, Hyperbolic )
{
    const auto x = tax::TM< 6 >::variable( 0.3, I{ 0.0, 0.6 } );
    const auto sh = sinh( x );
    const auto ch = cosh( x );
    const auto th = tanh( x );
    tax::test::ExpectEncloses( sh, []( const auto& p ) { return std::sinh( p[0] ); }, 41 );
    tax::test::ExpectEncloses( ch, []( const auto& p ) { return std::cosh( p[0] ); }, 41 );
    tax::test::ExpectEncloses( th, []( const auto& p ) { return std::tanh( p[0] ); }, 41 );

    // cosh^2 - sinh^2 encloses 1.
    const auto unit = square( ch ) - square( sh );
    EXPECT_TRUE( unit.bound().contains( 1.0 ) );
    EXPECT_LT( unit.bound().width(), 1e-4 );
}

TEST( TMMath, InverseTrig )
{
    const auto x = tax::TM< 6 >::variable( 0.2, I{ 0.0, 0.4 } );
    const auto as = asin( x );
    const auto ac = acos( x );
    const auto at = atan( x );
    tax::test::ExpectEncloses( as, []( const auto& p ) { return std::asin( p[0] ); }, 41 );
    tax::test::ExpectEncloses( ac, []( const auto& p ) { return std::acos( p[0] ); }, 41 );
    tax::test::ExpectEncloses( at, []( const auto& p ) { return std::atan( p[0] ); }, 41 );

    // asin polynomial coefficients from the derivative recursion at c = 0.2.
    const double c = 0.2, om = 1.0 - c * c;
    EXPECT_NEAR( as.polynomial()[1], 1.0 / std::sqrt( om ), 1e-13 );
    EXPECT_NEAR( as.polynomial()[2], c / ( 2.0 * std::pow( om, 1.5 ) ), 1e-13 );
    EXPECT_NEAR( as.polynomial()[3], ( 1.0 + 2.0 * c * c ) / ( 6.0 * std::pow( om, 2.5 ) ), 1e-13 );
}

TEST( TMMath, AtanAwayFromZero )
{
    const auto x = tax::TM< 5 >::variable( 1.0, I{ 0.5, 1.5 } );
    const auto f = atan( x );
    tax::test::ExpectEncloses( f, []( const auto& p ) { return std::atan( p[0] ); }, 41 );
}

TEST( TMMath, IntegerPowers )
{
    const auto x = tax::TM< 5 >::variable( 1.5, I{ 1.0, 2.0 } );
    const auto cube = pow( x, 3 );
    tax::test::ExpectEncloses( cube, []( const auto& p ) { return p[0] * p[0] * p[0]; }, 41 );
    const auto invsq = pow( x, -2 );
    tax::test::ExpectEncloses( invsq, []( const auto& p ) { return 1.0 / ( p[0] * p[0] ); }, 41 );
    const auto unit = pow( x, 0 );
    EXPECT_DOUBLE_EQ( unit.value(), 1.0 );
    EXPECT_EQ( unit.remainder(), I{} );
}

TEST( TMMath, Division )
{
    const auto x = tax::TM< 5 >::variable( 1.0, I{ 0.6, 1.4 } );
    const auto f = ( x + 1.0 ) / ( x + 2.0 );
    tax::test::ExpectEncloses(
        f, []( const auto& p ) { return ( p[0] + 1.0 ) / ( p[0] + 2.0 ); }, 41 );
}

TEST( TMMath, CompositeExpression )
{
    // Deep composition across several intrinsics.
    const auto x = tax::TM< 6 >::variable( 0.3, I{ 0.1, 0.5 } );
    const auto f = exp( sin( x ) + log( 2.0 + x ) ) / sqrt( 4.0 + x );
    tax::test::ExpectEncloses(
        f,
        []( const auto& p ) {
            return std::exp( std::sin( p[0] ) + std::log( 2.0 + p[0] ) ) / std::sqrt( 4.0 + p[0] );
        },
        41, 1e-11 );
}

TEST( TMMath, MultivariateThesisFigure43 )
{
    // The 2D example of §4.4.2: f(x, y) = sin(1.7 x + 0.5) (y + 2) sin(1.5 y)
    // on [-1, 1] x [-1, 1].
    using TMT = tax::TM< 8, 2 >;
    const TMT::Point x0{ 0.0, 0.0 };
    const TMT::Domain dom{ I{ -1.0, 1.0 }, I{ -1.0, 1.0 } };
    const auto x = TMT::variable< 0 >( x0, dom );
    const auto y = TMT::variable< 1 >( x0, dom );

    const auto f = sin( 1.7 * x + 0.5 ) * ( y + 2.0 ) * sin( 1.5 * y );
    tax::test::ExpectEncloses(
        f,
        []( const auto& p ) {
            return std::sin( 1.7 * p[0] + 0.5 ) * ( p[1] + 2.0 ) * std::sin( 1.5 * p[1] );
        },
        15, 1e-10 );
}

// ---------------------------------------------------------------------------
// Domain-condition violations
// ---------------------------------------------------------------------------

TEST( TMMath, DomainViolationsThrow )
{
    const auto x = tax::TM< 3 >::variable( 0.0, I{ -1.0, 1.0 } );
    EXPECT_THROW( (void)log( x ), std::domain_error );         // crosses 0
    EXPECT_THROW( (void)sqrt( x ), std::domain_error );        // crosses 0
    EXPECT_THROW( (void)isqrt( x ), std::domain_error );       // crosses 0
    EXPECT_THROW( (void)reciprocal( x ), std::domain_error );  // contains 0
    EXPECT_THROW( (void)( 1.0 / x ), std::domain_error );      // contains 0

    const auto wide = tax::TM< 3 >::variable( 0.5, I{ -0.8, 1.2 } );
    EXPECT_THROW( (void)asin( wide ), std::domain_error );  // leaves (-1, 1)
    EXPECT_THROW( (void)acos( wide ), std::domain_error );
}
