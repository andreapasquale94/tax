// Worked computations from K. Makino's PhD thesis (MSUCL-1093, 1998),
// chapters 4-5, kept as regression oracles: every number asserted below is
// quoted (or directly derived) from the thesis text and tables.
#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include "modelTestUtils.hpp"

using I = tax::Interval< double >;

// ---------------------------------------------------------------------------
// §4.2 — interval arithmetic warm-up computations
// ---------------------------------------------------------------------------

TEST( ThesisExamples, Section42IntervalCancellationBlowUp )
{
    // I - I = [a - b, b - a]: twice the width, even though x - x = 0.
    const I x{ 1.9, 2.1 };
    const I d = x - x;
    EXPECT_NEAR( d.lower(), -0.2, 1e-14 );
    EXPECT_NEAR( d.upper(), 0.2, 1e-14 );
    // And I + I = [2a, 2b]:
    const I s = x + x;
    EXPECT_NEAR( s.lower(), 3.8, 1e-14 );
    EXPECT_NEAR( s.upper(), 4.2, 1e-14 );
}

TEST( ThesisExamples, Section441NaiveIntervalEvaluation )
{
    // Straight interval evaluation of f(x) = 1/x + x on [1.9, 2.1]:
    //   1/[1.9, 2.1] + [1.9, 2.1] = [2.37619..., 2.62631...], width 0.25012531
    // (Table 4.3, "Intervals, n_d = 1") — the artificial blow-up that Taylor
    // models remove; the exact range is only [2.42631, 2.57619].
    const I x{ 1.9, 2.1 };
    const I f = 1.0 / x + x;
    EXPECT_NEAR( f.lower(), 1.0 / 2.1 + 1.9, 1e-12 );
    EXPECT_NEAR( f.upper(), 1.0 / 1.9 + 2.1, 1e-12 );
    EXPECT_NEAR( f.width(), 0.25012531, 1e-7 );
    EXPECT_TRUE( f.contains( I{ 2.42632, 2.57618 } ) );  // encloses the true range
}

// ---------------------------------------------------------------------------
// §4.4.1 — the 1/x + x example, step by step
// ---------------------------------------------------------------------------

TEST( ThesisExamples, Section441IdentityModel )
{
    // T_{alpha,i} = (2 + (x - 2), [0, 0]) and B(P_{alpha,ibar}) = [-0.1, 0.1].
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    EXPECT_EQ( x.remainder(), I{} );

    const auto xbar = x - 2.0;
    const auto b = xbar.bound();
    EXPECT_NEAR( b.lower(), -0.1, 1e-14 );
    EXPECT_NEAR( b.upper(), 0.1, 1e-14 );
}

TEST( ThesisExamples, Section441ReciprocalModel )
{
    // By (4.11): T_{alpha,1/i} =
    //   (1/2 - (x-2)/4 + (x-2)^2/8 - (x-2)^3/16, [0, 4.038e-6]).
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    const auto r = reciprocal( x );
    EXPECT_NEAR( r.polynomial()[0], 0.5, 1e-14 );
    EXPECT_NEAR( r.polynomial()[1], -0.25, 1e-14 );
    EXPECT_NEAR( r.polynomial()[2], 0.125, 1e-14 );
    EXPECT_NEAR( r.polynomial()[3], -0.0625, 1e-14 );

    // (4.15): the Lagrange term lands in [0, 4.0386107e-6].
    EXPECT_GE( r.remainder().lower(), -1e-12 );
    EXPECT_NEAR( r.remainder().upper(), 4.0386107e-6, 1e-9 );

    tax::test::ExpectEncloses( r, []( const auto& p ) { return 1.0 / p[0]; }, 41 );
}

namespace
{

template < int N >
double totalBoundWidthOrderN()
{
    const auto x = tax::TM< N >::variable( 2.0, I{ 1.9, 2.1 } );
    return ( 1.0 / x + x ).bound().width();
}

}  // namespace

TEST( ThesisExamples, Table43TaylorModelsBeatIntervals )
{
    // Table 4.3: exact width 0.14987468; a single interval evaluation blows
    // up to 0.25012531. Every Taylor-model bound must fall between the two
    // (our naive order-sum bounder is slightly wider than the thesis's exact
    // polynomial bounder, but far below the interval result), and must not
    // grow with the order.
    const std::array< double, 4 > w{ totalBoundWidthOrderN< 1 >(), totalBoundWidthOrderN< 2 >(),
                                     totalBoundWidthOrderN< 3 >(), totalBoundWidthOrderN< 5 >() };
    for ( double width : w )
    {
        EXPECT_GT( width, 0.14987468 );  // must enclose the exact range
        EXPECT_LT( width, 0.25012531 );  // must beat the naive interval result
    }
    EXPECT_LE( w[1], w[0] + 1e-12 );
    EXPECT_LE( w[2], w[1] + 1e-12 );
    EXPECT_LE( w[3], w[2] + 1e-12 );

    // The enclosure always contains the exact range [2.42631, 2.57619].
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    EXPECT_TRUE( ( 1.0 / x + x ).bound().contains( I{ 2.42632, 2.57618 } ) );
}

// ---------------------------------------------------------------------------
// §4.4.2 — bound enclosures of functions
// ---------------------------------------------------------------------------

TEST( ThesisExamples, Section442Function1D )
{
    // Figure 4.2: f(x) = x (x-1.1) (x+2) (x+2.2) (x+2.5) (x+3) sin(1.7x + 0.5)
    // on [-0.5, 1.0], enclosed by 7th/8th-order Taylor models.
    const auto x = tax::TM< 8 >::variable( 0.25, I{ -0.5, 1.0 } );
    const auto f = x * ( x - 1.1 ) * ( x + 2.0 ) * ( x + 2.2 ) * ( x + 2.5 ) * ( x + 3.0 ) *
                   sin( 1.7 * x + 0.5 );
    const auto ref = []( const auto& p ) {
        return p[0] * ( p[0] - 1.1 ) * ( p[0] + 2.0 ) * ( p[0] + 2.2 ) * ( p[0] + 2.5 ) *
               ( p[0] + 3.0 ) * std::sin( 1.7 * p[0] + 0.5 );
    };
    tax::test::ExpectEncloses( f, ref, 61, 1e-9 );
    EXPECT_TRUE( f.bound().contains( 0.0 ) );  // f(0) = 0 lies in the enclosure
}

TEST( ThesisExamples, Section442Function2D )
{
    // Figures 4.3-4.4: f(x, y) = sin(1.7x + 0.5) (y + 2) sin(1.5y) on
    // [-1, 1]^2, enclosed by moderate-order Taylor models.
    using TMT = tax::TM< 9, 2 >;
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
        21, 1e-10 );
}

// ---------------------------------------------------------------------------
// §5.5.2 — verified multidimensional integral (5.11)
// ---------------------------------------------------------------------------

TEST( ThesisExamples, Section552DoubleIntegral )
{
    // Gradshteyn & Ryzhik §4.621 reference integral:
    //   int_0^{pi/2} int_0^{pi/2} sin(y) sqrt(1 - k^2 sin^2 x sin^2 y)
    //                              / (1 - k^2 sin^2 y) dx dy
    //   = pi / (2 sqrt(1 - k^2)),   k^2 = 0.1  ->  1.655764710966017.
    // The RDA recipe (Fig. 5.3): Taylor-model the integrand, antiderive in
    // both variables, and evaluate the definite integral by
    // inclusion-exclusion over the corners of the box.
    constexpr double k2 = 0.1;
    constexpr double h = std::numbers::pi / 4.0;

    using TMT = tax::TM< 9, 2 >;
    const TMT::Point x0{ h, h };  // expand at the center of [0, pi/2]^2
    const TMT::Domain dom{ I{ 0.0, 2.0 * h }, I{ 0.0, 2.0 * h } };
    const auto x = TMT::variable< 0 >( x0, dom );
    const auto y = TMT::variable< 1 >( x0, dom );

    const auto s2y = square( sin( y ) );
    const auto integrand =
        sin( y ) * sqrt( 1.0 - k2 * square( sin( x ) ) * s2y ) / ( 1.0 - k2 * s2y );

    const auto antideriv = integrand.integ< 0 >().integ< 1 >();
    const auto enclosure = antideriv.eval( { h, h } ) - antideriv.eval( { -h, h } ) -
                           antideriv.eval( { h, -h } ) + antideriv.eval( { -h, -h } );

    const double exact = 1.655764710966017;  // pi / (2 sqrt(0.9))
    EXPECT_TRUE( enclosure.contains( exact ) );
    // Table 5.8 reports width ~0.42 for a 5th-order model on a single box;
    // this 9th-order model on a single box reaches ~0.04 even with the
    // naive polynomial bounder.
    EXPECT_LT( enclosure.width(), 0.05 );
}
