#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <tax/tax.hpp>

using I = tax::Interval< double >;

// ---------------------------------------------------------------------------
// Table 4.1 — elementary interval arithmetic
// ---------------------------------------------------------------------------

TEST( Interval, AdditionTable41 )
{
    const I r = I{ 1.0, 2.0 } + I{ 3.0, 5.0 };
    EXPECT_NEAR( r.lower(), 4.0, 1e-14 );
    EXPECT_NEAR( r.upper(), 7.0, 1e-14 );
    EXPECT_TRUE( r.contains( I{ 4.0, 7.0 } ) );  // outward rounding: enclosure
}

TEST( Interval, NegationTable41 )
{
    const I r = -I{ 1.0, 2.0 };
    EXPECT_EQ( r.lower(), -2.0 );
    EXPECT_EQ( r.upper(), -1.0 );
}

TEST( Interval, MultiplicationTable41 )
{
    const I r = I{ -1.0, 2.0 } * I{ 3.0, 4.0 };
    EXPECT_NEAR( r.lower(), -4.0, 1e-14 );
    EXPECT_NEAR( r.upper(), 8.0, 1e-14 );
    EXPECT_TRUE( r.contains( I{ -4.0, 8.0 } ) );
}

TEST( Interval, DivisionTable41 )
{
    const I r = I{ 1.0, 2.0 } / I{ 4.0, 8.0 };
    EXPECT_NEAR( r.lower(), 0.125, 1e-14 );
    EXPECT_NEAR( r.upper(), 0.5, 1e-14 );
    EXPECT_THROW( (void)( I{ 1.0, 2.0 } / I{ -1.0, 1.0 } ), std::domain_error );
}

TEST( Interval, BlowUpExample )
{
    // Thesis §4.2: I - I has twice the width of I — the dependency problem.
    const I x{ 1.9, 2.1 };
    const I r = x - x;
    EXPECT_NEAR( r.width(), 0.4, 1e-12 );
    EXPECT_TRUE( r.contains( 0.0 ) );
}

TEST( Interval, SubDistributivity )
{
    // (4.2): I1 (I2 + I3) is contained in I1 I2 + I1 I3.
    const I i1{ -1.0, 2.0 }, i2{ 1.0, 2.0 }, i3{ -3.0, -1.0 };
    const I lhs = i1 * ( i2 + i3 );
    const I rhs = i1 * i2 + i1 * i3;
    EXPECT_TRUE( rhs.contains( lhs ) );
    EXPECT_LT( lhs.width(), rhs.width() );
}

// ---------------------------------------------------------------------------
// Squares and powers
// ---------------------------------------------------------------------------

TEST( Interval, SqrIsSharp )
{
    // (5.4): the square never dips below zero, unlike the generic product.
    const I x{ -1.0, 2.0 };
    const I s = tax::model::sqr( x );
    EXPECT_EQ( s.lower(), 0.0 );
    EXPECT_NEAR( s.upper(), 4.0, 1e-14 );
    const I p = x * x;  // generic product suffers the dependency problem
    EXPECT_LT( p.lower(), -1.9 );
}

TEST( Interval, PowEvenOddNegative )
{
    const I x{ -2.0, 3.0 };
    const I even = tax::model::pow( x, 2 );
    EXPECT_EQ( even.lower(), 0.0 );
    EXPECT_NEAR( even.upper(), 9.0, 1e-13 );

    const I odd = tax::model::pow( x, 3 );
    EXPECT_NEAR( odd.lower(), -8.0, 1e-13 );
    EXPECT_NEAR( odd.upper(), 27.0, 1e-13 );

    const I inv = tax::model::pow( I{ 2.0, 4.0 }, -2 );
    EXPECT_NEAR( inv.lower(), 1.0 / 16.0, 1e-14 );
    EXPECT_NEAR( inv.upper(), 0.25, 1e-14 );

    EXPECT_EQ( tax::model::pow( x, 0 ).lower(), 1.0 );
    EXPECT_EQ( tax::model::pow( x, 0 ).upper(), 1.0 );
}

TEST( Interval, PowMignitudeLowerBound )
{
    const I x{ -3.0, -2.0 };
    const I s = tax::model::pow( x, 2 );
    EXPECT_NEAR( s.lower(), 4.0, 1e-13 );
    EXPECT_NEAR( s.upper(), 9.0, 1e-13 );
}

// ---------------------------------------------------------------------------
// Elementary function enclosures
// ---------------------------------------------------------------------------

TEST( Interval, ExpLog )
{
    const I e = tax::model::exp( I{ 0.0, 1.0 } );
    EXPECT_TRUE( e.contains( 1.0 ) );
    EXPECT_TRUE( e.contains( std::numbers::e ) );
    EXPECT_NEAR( e.lower(), 1.0, 1e-12 );
    EXPECT_NEAR( e.upper(), std::numbers::e, 1e-12 );

    const I l = tax::model::log( I{ 1.0, std::numbers::e } );
    EXPECT_TRUE( l.contains( 0.0 ) );
    EXPECT_TRUE( l.contains( 1.0 ) );
    EXPECT_THROW( (void)tax::model::log( I{ -1.0, 1.0 } ), std::domain_error );
}

TEST( Interval, Sqrt )
{
    const I r = tax::model::sqrt( I{ 4.0, 9.0 } );
    EXPECT_NEAR( r.lower(), 2.0, 1e-13 );
    EXPECT_NEAR( r.upper(), 3.0, 1e-13 );
    EXPECT_THROW( (void)tax::model::sqrt( I{ -1.0, 1.0 } ), std::domain_error );
}

TEST( Interval, SinEnclosesExtrema )
{
    // pi/2 lies inside [0, pi]: the maximum 1 must be reported.
    const I s = tax::model::sin( I{ 0.0, std::numbers::pi } );
    EXPECT_EQ( s.upper(), 1.0 );
    EXPECT_GE( s.lower(), -1e-10 );
    EXPECT_TRUE( s.contains( 1.0 ) );

    // no extremum inside [0.1, 0.2]: endpoint values only
    const I t = tax::model::sin( I{ 0.1, 0.2 } );
    EXPECT_NEAR( t.lower(), std::sin( 0.1 ), 1e-12 );
    EXPECT_NEAR( t.upper(), std::sin( 0.2 ), 1e-12 );

    // width beyond a full period: the trivial enclosure
    const I w = tax::model::sin( I{ 0.0, 10.0 } );
    EXPECT_EQ( w.lower(), -1.0 );
    EXPECT_EQ( w.upper(), 1.0 );
}

TEST( Interval, CosEnclosesExtrema )
{
    // pi lies inside [3, 4]: the minimum -1 must be reported.
    const I c = tax::model::cos( I{ 3.0, 4.0 } );
    EXPECT_EQ( c.lower(), -1.0 );
    EXPECT_NEAR( c.upper(), std::cos( 4.0 ), 1e-12 );

    // 2 pi inside [6, 7]: maximum 1.
    const I d = tax::model::cos( I{ 6.0, 7.0 } );
    EXPECT_EQ( d.upper(), 1.0 );
}

TEST( Interval, SinhCosh )
{
    const I s = tax::model::sinh( I{ -1.0, 2.0 } );
    EXPECT_NEAR( s.lower(), std::sinh( -1.0 ), 1e-12 );
    EXPECT_NEAR( s.upper(), std::sinh( 2.0 ), 1e-12 );

    const I c = tax::model::cosh( I{ -1.0, 2.0 } );
    EXPECT_EQ( c.lower(), 1.0 );  // 0 inside: minimum of cosh
    EXPECT_NEAR( c.upper(), std::cosh( 2.0 ), 1e-11 );

    const I c2 = tax::model::cosh( I{ 1.0, 2.0 } );
    EXPECT_NEAR( c2.lower(), std::cosh( 1.0 ), 1e-12 );
}

// ---------------------------------------------------------------------------
// Set operations and accessors
// ---------------------------------------------------------------------------

TEST( Interval, SetOperations )
{
    const I a{ 0.0, 2.0 }, b{ 1.0, 3.0 };
    EXPECT_EQ( hull( a, b ), ( I{ 0.0, 3.0 } ) );
    EXPECT_EQ( intersect( a, b ), ( I{ 1.0, 2.0 } ) );
    EXPECT_THROW( (void)intersect( I{ 0.0, 1.0 }, I{ 2.0, 3.0 } ), std::domain_error );

    EXPECT_TRUE( a.contains( 0.0 ) );
    EXPECT_TRUE( a.contains( 2.0 ) );
    EXPECT_FALSE( a.contains( 2.5 ) );
    EXPECT_TRUE( a.contains( I{ 0.5, 1.5 } ) );

    EXPECT_DOUBLE_EQ( a.mid(), 1.0 );
    EXPECT_NEAR( a.width(), 2.0, 1e-14 );
    EXPECT_DOUBLE_EQ( ( I{ -3.0, 2.0 } ).mag(), 3.0 );
    EXPECT_DOUBLE_EQ( ( I{ -3.0, 2.0 } ).mig(), 0.0 );
    EXPECT_DOUBLE_EQ( ( I{ 1.0, 2.0 } ).mig(), 1.0 );
}

TEST( Interval, InvalidConstruction )
{
    EXPECT_THROW( (void)I( 2.0, 1.0 ), std::invalid_argument );
}

TEST( Interval, ConstexprArithmetic )
{
    constexpr I a{ 1.0, 2.0 };
    constexpr I b = a + a;
    static_assert( b.lower() <= 2.0 && b.upper() >= 4.0 );
    constexpr I c = tax::model::pow( I{ -1.0, 1.0 }, 2 );
    static_assert( c.lower() == 0.0 );
    static_assert( c.upper() >= 1.0 );
    SUCCEED();
}
