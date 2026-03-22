#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include <tax/tax.hpp>
#include <tax/ads/ads_tree.hpp>
#include <tax/ads/box.hpp>
#include <tax/ode/taylor_integrator.hpp>

using namespace tax;

// =============================================================================
// step_da: verify Taylor coefficients for a single DA step
// =============================================================================

// dx/dt = v, dv/dt = -x  (harmonic oscillator)
// With DA state expanded around (x0, v0) = (1, 0)
TEST( IntegrateAds, StepDaHarmonicOscillator )
{
    constexpr int N = 10;  // time Taylor order
    constexpr int P = 2;   // DA order
    constexpr int D = 2;   // state dimension

    using DA = TEn< P, D >;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.1, 0.1 } };
    auto x0 = ode::make_da_state< P, D >( box );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    auto [p, h] = ode::step_da< N, P, D >( f, x0, 0.0, 1e-16 );

    EXPECT_GT( h, 0.0 );

    // The 0th Taylor coefficient should equal the initial state
    // x_da(0)[0] should be the DA polynomial for x0 = 1.0 + 0.1*δx
    EXPECT_NEAR( p( 0 )[0].value(), 1.0, 1e-14 );
    EXPECT_NEAR( p( 1 )[0].value(), 0.0, 1e-14 );

    // The 1st Taylor coefficient should equal f(x0, t0):
    // dx/dt = v → p(0)[1] = x0_v = 0 + 0.1*δv
    // dv/dt = -x → p(1)[1] = -x0 = -(1 + 0.1*δx)
    EXPECT_NEAR( p( 0 )[1].value(), 0.0, 1e-14 );
    EXPECT_NEAR( p( 1 )[1].value(), -1.0, 1e-14 );

    // 2nd Taylor coefficient: x''(t)/2 = -x(t)/2, so p(0)[2] = -x0/2
    EXPECT_NEAR( p( 0 )[2].value(), -0.5, 1e-14 );
}

// =============================================================================
// propagate_box: linear ODE, P=1 should give exact flow map
// =============================================================================

// Harmonic oscillator: solution is linear in initial conditions.
// x(t) = x0*cos(t) + v0*sin(t)
// v(t) = -x0*sin(t) + v0*cos(t)
TEST( IntegrateAds, PropagateBoxLinearHarmonicOscillator )
{
    constexpr int N = 20;
    constexpr int P = 1;
    constexpr int D = 2;

    using DA = TEn< P, D >;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.1, 0.1 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    const double tmax = std::numbers::pi / 2.0;
    auto xf = ode::propagate_box< N, P, D >( f, box, 0.0, tmax, 1e-16 );

    // At center (δ=0): x(π/2) = cos(π/2) = 0, v(π/2) = -sin(π/2) = -1
    EXPECT_NEAR( xf( 0 ).value(), 0.0, 1e-10 );
    EXPECT_NEAR( xf( 1 ).value(), -1.0, 1e-10 );

    // Linear coefficients: ∂x/∂(δx) = hw_x * cos(t) = 0.1*0 = 0
    //                      ∂x/∂(δv) = hw_v * sin(t) = 0.1*1 = 0.1
    MultiIndex< D > e_dx{ 1, 0 };
    MultiIndex< D > e_dv{ 0, 1 };
    EXPECT_NEAR( xf( 0 ).coeff( e_dx ), 0.0, 1e-10 );
    EXPECT_NEAR( xf( 0 ).coeff( e_dv ), 0.1, 1e-10 );

    // ∂v/∂(δx) = hw_x * (-sin(t)) = 0.1*(-1) = -0.1
    // ∂v/∂(δv) = hw_v * cos(t)    = 0.1*0    = 0
    EXPECT_NEAR( xf( 1 ).coeff( e_dx ), -0.1, 1e-10 );
    EXPECT_NEAR( xf( 1 ).coeff( e_dv ), 0.0, 1e-10 );
}

// =============================================================================
// propagate_box: verify point evaluation matches direct integration
// =============================================================================

TEST( IntegrateAds, PropagateBoxPointEvaluation )
{
    constexpr int N = 20;
    constexpr int P = 3;
    constexpr int D = 2;

    using DA = TEn< P, D >;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.1, 0.1 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    const double tmax = 1.0;
    auto xf_da = ode::propagate_box< N, P, D >( f, box, 0.0, tmax, 1e-16 );

    // Evaluate at δ = (0.5, -0.3) → x0 = 1.05, v0 = -0.03
    DA::Input delta{ 0.5, -0.3 };
    double x0_pt = 1.0 + 0.1 * 0.5;
    double v0_pt = 0.0 + 0.1 * ( -0.3 );

    double x_exact = x0_pt * std::cos( tmax ) + v0_pt * std::sin( tmax );
    double v_exact = -x0_pt * std::sin( tmax ) + v0_pt * std::cos( tmax );

    double x_da = xf_da( 0 ).eval( delta );
    double v_da = xf_da( 1 ).eval( delta );

    EXPECT_NEAR( x_da, x_exact, 1e-10 );
    EXPECT_NEAR( v_da, v_exact, 1e-10 );
}

// =============================================================================
// integrate_ads: no splitting needed for a linear system
// =============================================================================

TEST( IntegrateAds, NoSplitLinearSystem )
{
    constexpr int N = 20;
    constexpr int P = 2;
    constexpr int D = 2;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.1, 0.1 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    auto tree = ode::integrate_ads< N, P >( f, box, 0.0, 1.0, 1e-16, 1e-6 );

    // For a linear system, a single subdomain with P>=1 should suffice.
    EXPECT_EQ( tree.num_done(), 1 );
    EXPECT_TRUE( tree.empty() );
}

// =============================================================================
// integrate_ads: nonlinear ODE triggers splitting
// =============================================================================

// Duffing-like oscillator: dv/dt = -x - x^3  (cubic nonlinearity)
TEST( IntegrateAds, SplitsNonlinearODE )
{
    constexpr int N = 15;
    constexpr int P = 3;
    constexpr int D = 2;

    // Large initial-condition domain to force splitting.
    Box< double, D > box{ { 1.0, 0.0 }, { 0.5, 0.5 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 ) - x( 0 ) * x( 0 ) * x( 0 );
    };

    auto tree = ode::integrate_ads< N, P >( f, box, 0.0, 3.0, 1e-12, 1e-4, 6 );

    // With a strong nonlinearity and large domain, ADS should split.
    EXPECT_GT( tree.num_done(), 1 );
    EXPECT_TRUE( tree.empty() );
}

// =============================================================================
// integrate_ads: point accuracy across subdomains
// =============================================================================

TEST( IntegrateAds, PointAccuracyAcrossSubdomains )
{
    constexpr int N = 15;
    constexpr int P = 3;
    constexpr int D = 2;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.5, 0.5 } };

    auto f_rhs = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 ) - x( 0 ) * x( 0 ) * x( 0 );
    };

    const double tmax = 2.0;
    auto tree = ode::integrate_ads< N, P >( f_rhs, box, 0.0, tmax, 1e-12, 1e-4, 8 );

    // Test a few sample points by comparing ADS result to direct integration.
    const std::array< std::array< double, 2 >, 3 > test_points = {
        { { 1.2, 0.1 }, { 0.8, -0.3 }, { 1.0, 0.0 } }
    };

    for ( const auto& pt : test_points )
    {
        // Find the leaf containing this initial condition.
        // Normalise to δ coordinates.
        std::array< double, D > delta;
        for ( int k = 0; k < D; ++k )
            delta[k] = ( pt[k] - box.center[k] ) / box.half_width[k];

        // Check it's within the domain.
        bool in_domain = true;
        for ( int k = 0; k < D; ++k )
            if ( std::abs( delta[k] ) > 1.0 ) in_domain = false;
        ASSERT_TRUE( in_domain );

        // Find the leaf in the ADS tree.
        int idx = tree.find_leaf( { pt[0], pt[1] } );
        ASSERT_GE( idx, 0 );

        const auto& leaf = tree.node( idx ).leaf();
        // Normalise to the leaf's local box.
        std::array< double, D > local_delta;
        for ( int k = 0; k < D; ++k )
            local_delta[k] = ( pt[k] - leaf.box.center[k] ) / leaf.box.half_width[k];

        double x_ads = leaf.tte.state( 0 ).eval( local_delta );
        double v_ads = leaf.tte.state( 1 ).eval( local_delta );

        // Direct integration for reference.
        Eigen::Vector2d x0_pt( pt[0], pt[1] );
        auto f_direct = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
            dx( 0 ) = x( 1 );
            dx( 1 ) = -x( 0 ) - x( 0 ) * x( 0 ) * x( 0 );
        };
        auto sol_ref = ode::integrate< N >( f_direct, x0_pt, 0.0, tmax, 1e-16 );

        // The ADS result should be close to the direct integration.
        EXPECT_NEAR( x_ads, sol_ref.x.back()( 0 ), 1e-2 );
        EXPECT_NEAR( v_ads, sol_ref.x.back()( 1 ), 1e-2 );
    }
}

// =============================================================================
// Kepler problem with ADS
// =============================================================================

TEST( IntegrateAds, KeplerOrbitSplits )
{
    constexpr int N = 15;
    constexpr int P = 3;
    constexpr int D = 4;

    // Near-circular orbit with perturbation in position and velocity.
    Box< double, D > box{
        { 1.0, 0.0, 0.0, 1.0 },         // center
        { 0.01, 0.01, 0.01, 0.05 }       // small perturbations in all dims
    };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        using std::sqrt;
        auto r2 = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 );
        auto r  = sqrt( r2 );
        auto r3 = r2 * r;
        dx( 0 ) = x( 2 );
        dx( 1 ) = x( 3 );
        dx( 2 ) = -x( 0 ) / r3;
        dx( 3 ) = -x( 1 ) / r3;
    };

    const double tmax = std::numbers::pi;  // half orbit
    auto tree = ode::integrate_ads< N, P >( f, box, 0.0, tmax, 1e-14, 1e-3, 4 );

    EXPECT_TRUE( tree.empty() );
    EXPECT_GE( tree.num_done(), 1 );

    // Evaluate at the center point and compare with direct integration.
    Eigen::Vector< double, D > x0c;
    x0c << 1.0, 0.0, 0.0, 1.0;

    auto sol_ref = ode::integrate< N >( f, x0c, 0.0, tmax, 1e-16 );

    // Find the leaf containing the center — search done leaves directly
    // to handle boundary cases where find_leaf may pick a neighbour.
    bool found = false;
    for ( int di : tree.done_leaves() )
    {
        const auto& leaf = tree.node( di ).leaf();
        if ( !leaf.box.contains( box.center ) ) continue;

        std::array< double, D > local_delta{};
        for ( int k = 0; k < D; ++k )
            local_delta[k] =
                ( box.center[k] - leaf.box.center[k] ) / leaf.box.half_width[k];

        for ( int k = 0; k < D; ++k )
        {
            double val_ads = leaf.tte.state( k ).eval( local_delta );
            EXPECT_NEAR( val_ads, sol_ref.x.back()( k ), 1e-4 );
        }
        found = true;
        break;
    }
    EXPECT_TRUE( found );
}
