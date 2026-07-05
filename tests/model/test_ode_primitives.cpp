#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/model/compose.hpp>

#include "../testUtils.hpp"
#include "modelTestUtils.hpp"

using I = tax::Interval< double >;

// ---------------------------------------------------------------------------
// Partial evaluation: fix
// ---------------------------------------------------------------------------

TEST( TMFix, CollapsesAxisExactly )
{
    // f(x, y) = (x-1) + (x-1)(y-2) + (y-2)^2, x0 = (1, 2).
    using TMT = tax::TM< 2, 2 >;
    const TMT::Point x0{ 1.0, 2.0 };
    const TMT::Domain dom{ I{ 0.0, 2.0 }, I{ 1.0, 3.0 } };
    const auto x = TMT::variable< 0 >( x0, dom );
    const auto y = TMT::variable< 1 >( x0, dom );
    const auto f = ( x - 1.0 ) + ( x - 1.0 ) * ( y - 2.0 ) + ( y - 2.0 ) * ( y - 2.0 );

    // Fix x = 1.5 (displacement 0.5): g(y) = 0.5 + 0.5(y-2) + (y-2)^2.
    const auto g = f.fix< 0 >( 1.5 );
    EXPECT_DOUBLE_EQ( g.value(), 0.5 );
    EXPECT_DOUBLE_EQ( ( g.polynomial().template coeff< 0, 1 >() ), 0.5 );  // dy coeff
    EXPECT_DOUBLE_EQ( ( g.polynomial().template coeff< 1, 0 >() ), 0.0 );  // no x dependence
    EXPECT_DOUBLE_EQ( ( g.polynomial().template coeff< 0, 2 >() ), 1.0 );

    // The fixed axis collapses to a point domain; remainder is unchanged.
    EXPECT_EQ( g.domain()[0], ( I{ 1.5, 1.5 } ) );
    EXPECT_EQ( g.remainder(), f.remainder() );

    // g agrees with f along x = 1.5.
    for ( double dy : { -0.5, 0.0, 0.7 } )
    {
        const auto ge = g.eval( { 0.0, dy } );
        const double truth = 0.5 + 0.5 * dy + dy * dy;
        EXPECT_GE( truth, ge.lower() - 1e-12 );
        EXPECT_LE( truth, ge.upper() + 1e-12 );
    }
}

TEST( TMFix, RuntimeIndexAndErrors )
{
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.5, 2.5 } );
    const auto f = x * x;
    EXPECT_DOUBLE_EQ( f.fix( 0, 2.5 ).value(), 6.25 );         // (2.5)^2
    EXPECT_THROW( (void)f.fix( 0, 3.0 ), std::domain_error );  // outside domain
    EXPECT_THROW( (void)f.fix( 5, 2.0 ), std::out_of_range );  // bad index
}

// ---------------------------------------------------------------------------
// Slot recycling: retarget
// ---------------------------------------------------------------------------

TEST( TMRetarget, ResetsDomainWhenIndependent )
{
    using TMT = tax::TM< 2, 2 >;
    const TMT::Point x0{ 1.0, 0.0 };
    const TMT::Domain dom{ I{ 0.5, 1.5 }, I{ -1.0, 1.0 } };
    const auto x = TMT::variable< 0 >( x0, dom );
    // f depends only on x; fixing y is a no-op that leaves f independent of y.
    const auto f = ( x * x ).fix< 1 >( 0.0 );

    const auto g = f.retarget< 1 >( 0.0, I{ 0.0, 0.25 } );
    EXPECT_EQ( g.domain()[1], ( I{ 0.0, 0.25 } ) );
    EXPECT_DOUBLE_EQ( g.expansionPoint()[1], 0.0 );
    // The x-model is untouched.
    EXPECT_DOUBLE_EQ( g.value(), f.value() );

    // Retargeting a variable the model depends on is rejected.
    EXPECT_THROW( (void)f.retarget< 0 >( 1.0, I{ 0.5, 1.5 } ), std::invalid_argument );
    // Expansion point must lie in the new domain.
    EXPECT_THROW( (void)f.retarget< 1 >( 5.0, I{ 0.0, 0.25 } ), std::invalid_argument );
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

TEST( TMCompose, MatchesDirectSubstitution )
{
    // G(z) = exp(z) composed with H(u) = 0.5 u equals exp(0.5 u).
    const auto z = tax::TM< 5 >::variable( 0.0, I{ -1.0, 1.0 } );
    const auto g = exp( z );
    const auto u = tax::TM< 5 >::variable( 0.0, I{ -1.0, 1.0 } );
    const std::array< tax::TM< 5 >, 1 > h{ 0.5 * u };

    const auto c = tax::model::compose( g, h );
    const auto direct = exp( 0.5 * u );
    tax::test::ExpectCoeffsNear( c.polynomial(), direct.polynomial(), 1e-14 );

    tax::test::ExpectEncloses( c, []( const auto& p ) { return std::exp( 0.5 * p[0] ); }, 41 );
}

TEST( TMCompose, MultivariateOuter )
{
    // G(a, b) = a*b + a over a0 = (0,0), domain [-1,1]^2.
    using TG = tax::TM< 3, 2 >;
    const TG::Point gx0{ 0.0, 0.0 };
    const TG::Domain gdom{ I{ -1.0, 1.0 }, I{ -1.0, 1.0 } };
    const auto a = TG::variable< 0 >( gx0, gdom );
    const auto b = TG::variable< 1 >( gx0, gdom );
    const auto g = a * b + a;

    // Inner H over a single u in [-1,1]: a = 0.5u (range [-0.5,0.5] in [-1,1]),
    // b = 0.25u.
    const auto u = tax::TM< 3 >::variable( 0.0, I{ -1.0, 1.0 } );
    const std::array< tax::TM< 3 >, 2 > h{ 0.5 * u, 0.25 * u };

    const auto c = tax::model::compose( g, h );
    // Expect (0.5u)(0.25u) + 0.5u = 0.125 u^2 + 0.5 u.
    EXPECT_NEAR( c.polynomial().template coeff< 1 >(), 0.5, 1e-14 );
    EXPECT_NEAR( c.polynomial().template coeff< 2 >(), 0.125, 1e-14 );
    tax::test::ExpectEncloses(
        c, []( const auto& p ) { return 0.125 * p[0] * p[0] + 0.5 * p[0]; }, 41 );
}

TEST( TMCompose, DomainViolationThrows )
{
    const auto z = tax::TM< 4 >::variable( 0.0, I{ -1.0, 1.0 } );
    const auto g = 1.0 / ( z + 2.0 );  // valid on [-1,1]
    const auto u = tax::TM< 4 >::variable( 0.0, I{ -1.0, 1.0 } );
    const std::array< tax::TM< 4 >, 1 > big{ 3.0 * u };  // range [-3,3] leaves [-1,1]
    EXPECT_THROW( (void)tax::model::compose( g, big ), std::domain_error );
}

// ---------------------------------------------------------------------------
// Printing and vector helpers
// ---------------------------------------------------------------------------

TEST( TMIo, StreamAndVectorHelpers )
{
    using TMT = tax::TM< 2, 2 >;
    const TMT::Point x0{ 1.0, 2.0 };
    const TMT::Domain dom{ I{ 0.5, 1.5 }, I{ 1.5, 2.5 } };
    const auto x = TMT::variable< 0 >( x0, dom );
    const auto y = TMT::variable< 1 >( x0, dom );
    const std::array< TMT, 2 > state{ x * y, x + y };

    const std::string s = tax::model::to_string( state[0] );
    EXPECT_NE( s.find( "on" ), std::string::npos );  // domain annotation present

    const auto v = tax::model::value( state );
    EXPECT_DOUBLE_EQ( v[0], 2.0 );
    EXPECT_DOUBLE_EQ( v[1], 3.0 );

    const auto b = tax::model::bound( state );
    EXPECT_TRUE( b[0].contains( 2.0 ) );

    // J for [x*y, x+y] at (1,2): [[y, x],[1,1]] = [[2,1],[1,1]].
    const auto J = tax::model::jacobian( state );
    EXPECT_DOUBLE_EQ( J( 0, 0 ), 2.0 );
    EXPECT_DOUBLE_EQ( J( 0, 1 ), 1.0 );
    EXPECT_DOUBLE_EQ( J( 1, 0 ), 1.0 );
    EXPECT_DOUBLE_EQ( J( 1, 1 ), 1.0 );
}
