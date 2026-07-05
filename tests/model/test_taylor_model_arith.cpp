#include <gtest/gtest.h>

#include "modelTestUtils.hpp"

using I = tax::Interval< double >;

// ---------------------------------------------------------------------------
// Construction / factories
// ---------------------------------------------------------------------------

TEST( TaylorModel, VariableIsExact )
{
    // Thesis §4.4.1: the identity is represented exactly, remainder [0, 0].
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    EXPECT_DOUBLE_EQ( x.value(), 2.0 );
    EXPECT_DOUBLE_EQ( x.polynomial()[1], 1.0 );
    EXPECT_DOUBLE_EQ( x.polynomial()[2], 0.0 );
    EXPECT_EQ( x.remainder(), I{} );

    const auto disp = x.displacementDomain();
    EXPECT_NEAR( disp[0].lower(), -0.1, 1e-14 );
    EXPECT_NEAR( disp[0].upper(), 0.1, 1e-14 );
}

TEST( TaylorModel, ExpansionPointMustBeInDomain )
{
    using TMT = tax::TM< 2 >;
    EXPECT_THROW( TMT( TMT::Poly::constant( 1.0 ), I{}, { 5.0 }, { I{ 0.0, 1.0 } } ),
                  std::invalid_argument );
}

TEST( TaylorModel, MultivariateVariables )
{
    using TMT = tax::TM< 2, 2 >;
    const TMT::Point x0{ 1.0, -1.0 };
    const TMT::Domain dom{ I{ 0.5, 1.5 }, I{ -1.5, -0.5 } };
    const auto v = TMT::variables( x0, dom );
    EXPECT_DOUBLE_EQ( v[0].value(), 1.0 );
    EXPECT_DOUBLE_EQ( v[1].value(), -1.0 );

    const auto f = v[0] * v[1];
    tax::test::ExpectEncloses( f, []( const auto& x ) { return x[0] * x[1]; }, 9 );
}

// ---------------------------------------------------------------------------
// The blow-up cure: x - x is exactly zero for Taylor models
// ---------------------------------------------------------------------------

TEST( TaylorModel, CancellationHasNoBlowUp )
{
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    const auto d = x - x;
    // Interval arithmetic gives width 0.4 here (§4.2); Taylor models cancel
    // the polynomial dependence exactly.
    EXPECT_LE( d.bound().width(), 1e-15 );
}

// ---------------------------------------------------------------------------
// Multiplication: truncated product + excess bound
// ---------------------------------------------------------------------------

TEST( TaylorModel, MultiplicationExcessIsSharp )
{
    // Order 1: x * x truncates dx^2 entirely; the remainder must hold the
    // excess bound of dx^2 over [-0.5, 0.5], which is exactly [0, 0.25]
    // thanks to the even-power tightening.
    const auto x = tax::TM< 1 >::variable( 0.0, I{ -0.5, 0.5 } );
    const auto p = x * x;
    EXPECT_DOUBLE_EQ( p.polynomial()[0], 0.0 );
    EXPECT_DOUBLE_EQ( p.polynomial()[1], 0.0 );
    EXPECT_GE( p.remainder().lower(), -1e-15 );
    EXPECT_NEAR( p.remainder().upper(), 0.25, 1e-12 );
    tax::test::ExpectEncloses( p, []( const auto& x_ ) { return x_[0] * x_[0]; } );
}

TEST( TaylorModel, MultiplicationContainment )
{
    const auto x = tax::TM< 4 >::variable( 1.0, I{ 0.7, 1.3 } );
    const auto f = ( x - 2.0 ) * ( x + 1.0 ) * x + 3.0;
    tax::test::ExpectEncloses(
        f, []( const auto& p ) { return ( p[0] - 2.0 ) * ( p[0] + 1.0 ) * p[0] + 3.0; } );
}

TEST( TaylorModel, SquareAtLeastAsTightAsProduct )
{
    const auto x = tax::TM< 2 >::variable( 0.5, I{ 0.0, 1.0 } );
    const auto f = x * x * x;  // cubic: truncation excess feeds the remainder
    const auto sq = square( f );
    const auto pr = f * f;
    EXPECT_LE( sq.remainder().width(), pr.remainder().width() + 1e-15 );
    tax::test::ExpectEncloses( sq, []( const auto& p ) {
        const double c = p[0] * p[0] * p[0];
        return c * c;
    } );
}

// ---------------------------------------------------------------------------
// Scalar and interval operands
// ---------------------------------------------------------------------------

TEST( TaylorModel, ScalarOperations )
{
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    const auto f = 2.0 * x - 1.0;
    EXPECT_DOUBLE_EQ( f.value(), 3.0 );
    EXPECT_LE( f.remainder().width(), 1e-15 );
    tax::test::ExpectEncloses( f, []( const auto& p ) { return 2.0 * p[0] - 1.0; } );

    const auto g = ( 1.0 - x ) / 4.0;
    tax::test::ExpectEncloses( g, []( const auto& p ) { return ( 1.0 - p[0] ) / 4.0; } );
}

TEST( TaylorModel, IntervalConstantWidensRemainder )
{
    // Adding an interval [c - r, c + r] shifts the polynomial by the midpoint
    // and widens the remainder by the radius.
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    const auto f = x + I{ -0.1, 0.3 };
    EXPECT_NEAR( f.value(), 2.1, 1e-14 );
    EXPECT_TRUE( f.remainder().contains( I{ -0.199, 0.199 } ) );
    EXPECT_NEAR( f.remainder().width(), 0.4, 1e-12 );
}

TEST( TaylorModel, IntervalScaling )
{
    const auto x = tax::TM< 2 >::variable( 0.0, I{ -1.0, 1.0 } );
    const auto f = x * I{ 1.9, 2.1 };  // s * x for an unknown s in [1.9, 2.1]
    // For every s in [1.9, 2.1] and every dx, s * dx must be enclosed.
    for ( double s : { 1.9, 2.0, 2.1 } )
    {
        for ( double dx : { -1.0, -0.5, 0.0, 0.5, 1.0 } )
        {
            const auto enc = f.eval( { dx } );
            EXPECT_GE( s * dx, enc.lower() - 1e-12 );
            EXPECT_LE( s * dx, enc.upper() + 1e-12 );
        }
    }
}

// ---------------------------------------------------------------------------
// Compatibility and bounds
// ---------------------------------------------------------------------------

TEST( TaylorModel, IncompatibleOperandsThrow )
{
    const auto a = tax::TM< 2 >::variable( 0.0, I{ -1.0, 1.0 } );
    const auto b = tax::TM< 2 >::variable( 0.0, I{ -2.0, 2.0 } );
    EXPECT_THROW( (void)( a + b ), std::invalid_argument );
    EXPECT_THROW( (void)( a * b ), std::invalid_argument );
}

TEST( TaylorModel, EvalOutsideDomainThrows )
{
    const auto x = tax::TM< 2 >::variable( 0.0, I{ -1.0, 1.0 } );
    EXPECT_THROW( (void)x.eval( { 1.5 } ), std::domain_error );
}

TEST( TaylorModel, BoundOfIdentity )
{
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    const auto b = x.bound();
    EXPECT_NEAR( b.lower(), 1.9, 1e-12 );
    EXPECT_NEAR( b.upper(), 2.1, 1e-12 );
}

TEST( TaylorModel, OrderBounds )
{
    // f = 3 + dx + dx^2 on displacement domain [-0.5, 0.5].
    const auto x = tax::TM< 2 >::variable( 0.0, I{ -0.5, 0.5 } );
    const auto f = x * x + x + 3.0;
    EXPECT_NEAR( f.orderBound( 0 ).mid(), 3.0, 1e-14 );
    EXPECT_NEAR( f.orderBound( 1 ).lower(), -0.5, 1e-12 );
    EXPECT_NEAR( f.orderBound( 1 ).upper(), 0.5, 1e-12 );
    EXPECT_GE( f.orderBound( 2 ).lower(), -1e-15 );  // even power: sharp at 0
    EXPECT_NEAR( f.orderBound( 2 ).upper(), 0.25, 1e-12 );
}

TEST( TaylorModel, CompoundAssignment )
{
    auto x = tax::TM< 3 >::variable( 1.0, I{ 0.5, 1.5 } );
    auto f = x;
    f *= x;
    f += 1.0;
    f -= x;
    tax::test::ExpectEncloses( f, []( const auto& p ) { return p[0] * p[0] + 1.0 - p[0]; } );
}
