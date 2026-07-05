// End-to-end demonstration that the Taylor-model surface is sufficient to
// build an ODE integrator: a minimal Picard iterator for the harmonic
// oscillator y1' = y2, y2' = -y1, whose exact flow is
//   y1(t) = y1_0 cos t + y2_0 sin t,
//   y2(t) = -y1_0 sin t + y2_0 cos t.
//
// The integrator uses only the public primitives: coordinate-variable
// factories, arithmetic, antiderivation (integ), partial evaluation (fix),
// slot recycling (retarget), and the jacobian helper. Variables are
// (y1_0, y2_0, tau) with tau the local step time.
#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "modelTestUtils.hpp"

using I = tax::Interval< double >;

namespace
{

constexpr int kN = 7;                          // truncation order
using TM = tax::TaylorModel< double, kN, 3 >;  // vars: 0 = y1_0, 1 = y2_0, 2 = tau
using State = std::array< TM, 2 >;

/// One Picard-built step flow over tau in [0, h], starting from `ic` (a state
/// that is constant in tau). Returns the flow as functions of (y1_0, y2_0, tau).
State picardStep( const State& ic )
{
    State r = ic;
    for ( int it = 0; it <= kN; ++it )
    {
        // F(R) evaluated from the previous iterate, then both components
        // updated together: R_i = ic_i + \int_0^tau F_i dtau.
        const TM f0 = r[1];         // y1' = y2
        const TM f1 = -1.0 * r[0];  // y2' = -y1
        r[0] = ic[0] + f0.integ< 2 >();
        r[1] = ic[1] + f1.integ< 2 >();
    }
    return r;
}

}  // namespace

// ---------------------------------------------------------------------------
// The Picard flow reproduces the analytic Taylor series
// ---------------------------------------------------------------------------

TEST( ODEIntegration, FlowMatchesAnalyticSeries )
{
    const double h = 0.3;
    const TM::Point x0{ 1.0, 0.5, 0.0 };
    const TM::Domain dom{ I{ 0.9, 1.1 }, I{ 0.4, 0.6 }, I{ 0.0, h } };
    const State ic{ TM::variable< 0 >( x0, dom ), TM::variable< 1 >( x0, dom ) };

    const State flow = picardStep( ic );

    // y1(tau) = y1_0 cos(tau) + y2_0 sin(tau): coefficient of y1_0*tau^k is the
    // cosine series, of y2_0*tau^k the sine series.
    EXPECT_NEAR( ( flow[0].polynomial().coeff< 1, 0, 0 >() ), 1.0, 1e-14 );
    EXPECT_NEAR( ( flow[0].polynomial().coeff< 1, 0, 2 >() ), -1.0 / 2.0, 1e-14 );
    EXPECT_NEAR( ( flow[0].polynomial().coeff< 1, 0, 4 >() ), 1.0 / 24.0, 1e-14 );
    EXPECT_NEAR( ( flow[0].polynomial().coeff< 0, 1, 1 >() ), 1.0, 1e-14 );
    EXPECT_NEAR( ( flow[0].polynomial().coeff< 0, 1, 3 >() ), -1.0 / 6.0, 1e-14 );
    EXPECT_NEAR( ( flow[0].polynomial().coeff< 0, 1, 5 >() ), 1.0 / 120.0, 1e-14 );

    // y2(tau) = -y1_0 sin(tau) + y2_0 cos(tau).
    EXPECT_NEAR( ( flow[1].polynomial().coeff< 1, 0, 1 >() ), -1.0, 1e-14 );
    EXPECT_NEAR( ( flow[1].polynomial().coeff< 1, 0, 3 >() ), 1.0 / 6.0, 1e-14 );
    EXPECT_NEAR( ( flow[1].polynomial().coeff< 0, 1, 0 >() ), 1.0, 1e-14 );
    EXPECT_NEAR( ( flow[1].polynomial().coeff< 0, 1, 2 >() ), -1.0 / 2.0, 1e-14 );
}

// ---------------------------------------------------------------------------
// The flow encloses the true solution across the whole (IC, time) box
// ---------------------------------------------------------------------------

TEST( ODEIntegration, FlowEnclosesTrueSolution )
{
    const double h = 0.3;
    const TM::Point x0{ 1.0, 0.5, 0.0 };
    const TM::Domain dom{ I{ 0.9, 1.1 }, I{ 0.4, 0.6 }, I{ 0.0, h } };
    const State ic{ TM::variable< 0 >( x0, dom ), TM::variable< 1 >( x0, dom ) };
    const State flow = picardStep( ic );

    // Sweep initial conditions and time; the model must enclose the analytic
    // flow at every sample.
    tax::test::ExpectEncloses(
        flow[0], []( const auto& p ) { return p[0] * std::cos( p[2] ) + p[1] * std::sin( p[2] ); },
        7, 1e-9 );
    tax::test::ExpectEncloses(
        flow[1], []( const auto& p ) { return -p[0] * std::sin( p[2] ) + p[1] * std::cos( p[2] ); },
        7, 1e-9 );
}

// ---------------------------------------------------------------------------
// Step continuation with fix + retarget, and the state-transition matrix
// ---------------------------------------------------------------------------

TEST( ODEIntegration, MultiStepContinuationAndJacobian )
{
    const double h = 0.25;
    const int steps = 4;  // integrate to t = 1.0
    const TM::Point x0{ 1.0, 0.0, 0.0 };
    const TM::Domain dom{ I{ 0.9, 1.1 }, I{ -0.1, 0.1 }, I{ 0.0, h } };

    State state{ TM::variable< 0 >( x0, dom ), TM::variable< 1 >( x0, dom ) };

    for ( int s = 0; s < steps; ++s )
    {
        const State flow = picardStep( state );
        // Advance to the step endpoint tau = h, then recycle the tau slot for
        // the next step's [0, h] window.
        state[0] = flow[0].fix< 2 >( h ).retarget< 2 >( 0.0, I{ 0.0, h } );
        state[1] = flow[1].fix< 2 >( h ).retarget< 2 >( 0.0, I{ 0.0, h } );
    }

    const double t = steps * h;  // 1.0

    // Tolerances here reflect the genuine truncation error of a 7th-order,
    // 4-step integration (~1e-8), not the primitives — the coefficient and
    // tau-column checks above/below stay exact.

    // Constant part = flow of the central initial condition (1, 0):
    //   y1(t) = cos t, y2(t) = -sin t.
    EXPECT_NEAR( state[0].value(), std::cos( t ), 1e-6 );
    EXPECT_NEAR( state[1].value(), -std::sin( t ), 1e-6 );

    // State-transition matrix d(state)/d(y1_0, y2_0) = the linear flow map:
    //   [[ cos t,  sin t], [-sin t, cos t]]  (tau-column is 0 after fix).
    const auto J = tax::model::jacobian( state );
    EXPECT_NEAR( J( 0, 0 ), std::cos( t ), 1e-6 );
    EXPECT_NEAR( J( 0, 1 ), std::sin( t ), 1e-6 );
    EXPECT_NEAR( J( 1, 0 ), -std::sin( t ), 1e-6 );
    EXPECT_NEAR( J( 1, 1 ), std::cos( t ), 1e-6 );
    EXPECT_NEAR( J( 0, 2 ), 0.0, 1e-14 );

    // Energy y1^2 + y2^2 is conserved: the enclosure of the propagated state
    // at the central IC stays near 1.
    const double e = state[0].value() * state[0].value() + state[1].value() * state[1].value();
    EXPECT_NEAR( e, 1.0, 1e-6 );
}
