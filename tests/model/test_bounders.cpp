#include <gtest/gtest.h>

#include <cmath>

#include "modelTestUtils.hpp"

using I = tax::Interval< double >;
using B = tax::Bounder;

namespace
{

/// A quadratic bound must never fall below naive on width, and must always
/// enclose the naive bound's intersection with the true range: we assert the
/// weaker, sufficient property that quadratic is contained in naive (both are
/// valid enclosures, so quadratic ⊆ naive is the tightening guarantee).
template < typename TMT >
void ExpectTightening( const TMT& f )
{
    const auto naive = f.bound( B::Naive );
    const auto quad = f.bound( B::Quadratic );
    EXPECT_TRUE( naive.contains( quad ) )
        << "quadratic bound " << quad << " not contained in naive " << naive;
    EXPECT_LE( quad.width(), naive.width() + 1e-15 );
}

}  // namespace

// ---------------------------------------------------------------------------
// Interior vertex: the headline win of §5.4.3
// ---------------------------------------------------------------------------

TEST( Bounders, InteriorVertexExactRecovery )
{
    // (h - 0.3)^2 on [-1, 1] has true range [0, 1.69]. The naive order-sum
    // bounds -0.6h and h^2 independently and reports a spurious negative
    // lower bound; the exact-quadratic bounder captures the interior vertex
    // at h = 0.3 and recovers the exact minimum 0.
    const auto x = tax::TM< 2 >::variable( 0.0, I{ -1.0, 1.0 } );
    const auto g = ( x - 0.3 ) * ( x - 0.3 );

    const auto naive = g.bound( B::Naive );
    const auto quad = g.bound( B::Quadratic );

    EXPECT_LT( naive.lower(), -0.4 );         // naive is loose below zero
    EXPECT_NEAR( quad.lower(), 0.0, 1e-12 );  // quadratic is exact at the vertex
    EXPECT_NEAR( quad.upper(), 1.69, 1e-12 );
    EXPECT_TRUE( naive.contains( quad ) );
}

TEST( Bounders, VertexOutsideDomainStaysMonotone )
{
    // 0.125 h^2 + 0.75 h has its vertex at h = -3, outside [-0.1, 0.1]; the
    // bounder should return the exact monotone endpoint hull.
    const auto x = tax::TM< 2 >::variable( 0.0, I{ -0.1, 0.1 } );
    const auto g = 0.125 * ( x * x ) + 0.75 * x;
    const auto quad = g.bound( B::Quadratic );
    // Endpoints: g(-0.1) = 0.00125 - 0.075 = -0.07375 ; g(0.1) = 0.07625.
    EXPECT_NEAR( quad.lower(), -0.07375, 1e-12 );
    EXPECT_NEAR( quad.upper(), 0.07625, 1e-12 );
}

// ---------------------------------------------------------------------------
// Thesis §4.4.1 / Table 4.3: sharper total bound for 1/x + x
// ---------------------------------------------------------------------------

TEST( Bounders, ReciprocalPlusIdentityTighterThanNaive )
{
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    const auto f = 1.0 / x + x;

    const auto naive = f.bound( B::Naive );
    const auto quad = f.bound( B::Quadratic );

    // Quadratic must be strictly tighter here and still enclose the exact
    // range [2.42631, 2.57619] (Table 4.3).
    EXPECT_LT( quad.width(), naive.width() );
    EXPECT_LE( quad.lower(), 2.426312 );
    EXPECT_GE( quad.upper(), 2.576185 );
    EXPECT_LT( quad.width(), 0.1502 );  // vs naive ~0.15138, exact 0.14987
}

// ---------------------------------------------------------------------------
// The default is Quadratic; both strategies are valid enclosures
// ---------------------------------------------------------------------------

TEST( Bounders, DefaultIsQuadratic )
{
    const auto x = tax::TM< 2 >::variable( 0.0, I{ -1.0, 1.0 } );
    const auto g = ( x - 0.3 ) * ( x - 0.3 );
    EXPECT_EQ( g.bound(), g.bound( B::Quadratic ) );
    EXPECT_EQ( g.polynomialBound(), g.polynomialBound( B::Quadratic ) );
}

TEST( Bounders, BothStrategiesEncloseTrueRange )
{
    // Sweep the true range and confirm both bounds contain every sample.
    const auto x = tax::TM< 4 >::variable( 1.0, I{ 0.5, 1.5 } );
    const auto f = ( x - 0.8 ) * ( x - 1.2 ) * x + 2.0;
    const auto naive = f.bound( B::Naive );
    const auto quad = f.bound( B::Quadratic );
    for ( int i = 0; i <= 100; ++i )
    {
        const double h = -0.5 + 1.0 * double( i ) / 100.0;
        const double dx = 1.0 + h - 1.0;  // displacement in [-0.5, 0.5]
        const double val = ( ( 1.0 + dx ) - 0.8 ) * ( ( 1.0 + dx ) - 1.2 ) * ( 1.0 + dx ) + 2.0;
        EXPECT_GE( val, naive.lower() - 1e-12 );
        EXPECT_LE( val, naive.upper() + 1e-12 );
        EXPECT_GE( val, quad.lower() - 1e-12 );
        EXPECT_LE( val, quad.upper() + 1e-12 );
    }
    EXPECT_TRUE( naive.contains( quad ) );
}

// ---------------------------------------------------------------------------
// Multivariate: tightening holds and cross terms stay correct
// ---------------------------------------------------------------------------

TEST( Bounders, MultivariateTightening )
{
    using TMT = tax::TM< 3, 2 >;
    const TMT::Point x0{ 0.0, 0.0 };
    const TMT::Domain dom{ I{ -1.0, 1.0 }, I{ -1.0, 1.0 } };
    const auto x = TMT::variable< 0 >( x0, dom );
    const auto y = TMT::variable< 1 >( x0, dom );

    // A form with a diagonal-dominant quadratic part plus a cross term.
    const auto f = ( x - 0.4 ) * ( x - 0.4 ) + ( y + 0.5 ) * ( y + 0.5 ) + 0.3 * ( x * y );
    ExpectTightening( f );

    // Both bounds enclose a scan of the domain corners and interior samples.
    const auto naive = f.bound( B::Naive );
    const auto quad = f.bound( B::Quadratic );
    for ( double hx = -1.0; hx <= 1.0; hx += 0.25 )
    {
        for ( double hy = -1.0; hy <= 1.0; hy += 0.25 )
        {
            const double v =
                ( hx - 0.4 ) * ( hx - 0.4 ) + ( hy + 0.5 ) * ( hy + 0.5 ) + 0.3 * ( hx * hy );
            EXPECT_GE( v, quad.lower() - 1e-12 );
            EXPECT_LE( v, quad.upper() + 1e-12 );
            EXPECT_GE( v, naive.lower() - 1e-12 );
        }
    }
}

TEST( Bounders, TighteningAcrossIntrinsics )
{
    // The tightening never regresses on a spread of nonlinear models.
    ExpectTightening( exp( tax::TM< 5 >::variable( 0.0, I{ -0.5, 0.5 } ) ) );
    ExpectTightening( sin( tax::TM< 6 >::variable( 0.5, I{ 0.0, 1.0 } ) ) );
    ExpectTightening( sqrt( tax::TM< 5 >::variable( 4.0, I{ 3.0, 5.0 } ) ) );
    ExpectTightening( 1.0 / tax::TM< 4 >::variable( 2.0, I{ 1.5, 2.5 } ) );
}

// ---------------------------------------------------------------------------
// Low orders: quadratic must agree with naive (nothing to tighten)
// ---------------------------------------------------------------------------

TEST( Bounders, LinearAndConstantUnaffected )
{
    const auto x1 = tax::TM< 1 >::variable( 2.0, I{ 1.5, 2.5 } );
    const auto lin = 3.0 * x1 - 1.0;
    EXPECT_EQ( lin.bound( B::Naive ), lin.bound( B::Quadratic ) );

    const auto c = tax::TM< 3 >::constant( 4.0, { 0.0 }, { I{ -1.0, 1.0 } } );
    EXPECT_EQ( c.bound( B::Naive ), c.bound( B::Quadratic ) );
}
