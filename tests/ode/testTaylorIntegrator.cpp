#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

#include <tax/tax.hpp>
#include <tax/ode/taylor_integrator.hpp>

using namespace tax;

// =============================================================================
// Scalar ODE tests
// =============================================================================

// dx/dt = x  →  x(t) = x0 * exp(t)
TEST( TaylorIntegratorScalar, ExponentialGrowth )
{
    constexpr int N = 25;
    const double x0 = 1.0;
    const double t0 = 0.0;
    const double tmax = 1.0;
    const double abstol = 1e-20;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto sol = ode::taylorinteg< N >( f, x0, t0, tmax, abstol );

    EXPECT_GT( sol.t.size(), 1u );
    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), std::exp( tmax ), 1e-14 );
}

// dx/dt = -x  →  x(t) = x0 * exp(-t)
TEST( TaylorIntegratorScalar, ExponentialDecay )
{
    constexpr int N = 25;
    const double x0 = 3.0;
    const double t0 = 0.0;
    const double tmax = 2.0;
    const double abstol = 1e-20;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return -x; };

    auto sol = ode::taylorinteg< N >( f, x0, t0, tmax, abstol );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), x0 * std::exp( -tmax ), 1e-14 );
}

// dx/dt = cos(t)  →  x(t) = sin(t)   (x(0) = 0)
TEST( TaylorIntegratorScalar, CosineForcing )
{
    constexpr int N = 25;
    const double x0 = 0.0;
    const double t0 = 0.0;
    const double tmax = std::numbers::pi;
    const double abstol = 1e-20;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) {
        using std::cos;
        return cos( t );
    };

    auto sol = ode::taylorinteg< N >( f, x0, t0, tmax, abstol );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), std::sin( tmax ), 1e-13 );
}

// dx/dt = 2*t  →  x(t) = t^2   (x(0) = 0)
TEST( TaylorIntegratorScalar, Quadratic )
{
    constexpr int N = 10;
    const double x0 = 0.0;
    const double t0 = 0.0;
    const double tmax = 5.0;
    const double abstol = 1e-20;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) { return 2.0 * t; };

    auto sol = ode::taylorinteg< N >( f, x0, t0, tmax, abstol );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), tmax * tmax, 1e-12 );
}

// Backward integration: dx/dt = x, integrate from 1 to 0
TEST( TaylorIntegratorScalar, BackwardIntegration )
{
    constexpr int N = 25;
    const double x0 = std::exp( 1.0 );
    const double t0 = 1.0;
    const double tmax = 0.0;
    const double abstol = 1e-20;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto sol = ode::taylorinteg< N >( f, x0, t0, tmax, abstol );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), 1.0, 1e-13 );
}

// =============================================================================
// Scalar ODE with trange
// =============================================================================

TEST( TaylorIntegratorScalar, ExponentialTrange )
{
    constexpr int N = 25;
    const double x0 = 1.0;
    const double abstol = 1e-20;

    std::vector< double > trange;
    for ( int i = 0; i <= 10; ++i ) trange.push_back( i * 0.1 );

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto sol = ode::taylorinteg< N >( f, x0, trange, abstol );

    ASSERT_EQ( sol.t.size(), trange.size() );
    ASSERT_EQ( sol.x.size(), trange.size() );

    for ( std::size_t i = 0; i < trange.size(); ++i )
    {
        EXPECT_NEAR( sol.x[i], std::exp( trange[i] ), 1e-14 ) << "  at t=" << trange[i];
    }
}

// =============================================================================
// Vector ODE tests
// =============================================================================

// Simple harmonic oscillator: dx1/dt = x2, dx2/dt = -x1
// Solution: x1(t) = cos(t), x2(t) = -sin(t) for x0 = (1, 0)
TEST( TaylorIntegratorVector, HarmonicOscillator )
{
    constexpr int N = 25;
    constexpr int D = 2;
    const double t0 = 0.0;
    const double tmax = 2.0 * std::numbers::pi;
    const double abstol = 1e-20;

    Eigen::Vector2d x0( 1.0, 0.0 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    auto sol = ode::taylorinteg< N >( f, x0, t0, tmax, abstol );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back()( 0 ), std::cos( tmax ), 1e-12 );
    EXPECT_NEAR( sol.x.back()( 1 ), std::sin( tmax ) * ( -1.0 ), 1e-12 );

    // After a full period, should return to initial conditions
    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-12 );
    EXPECT_NEAR( sol.x.back()( 1 ), 0.0, 1e-12 );
}

// Decoupled system: dx1/dt = x1, dx2/dt = -x2
// Solution: x1(t) = e^t, x2(t) = e^{-t}
TEST( TaylorIntegratorVector, DecoupledExponentials )
{
    constexpr int N = 25;
    const double t0 = 0.0;
    const double tmax = 1.0;
    const double abstol = 1e-20;

    Eigen::Vector2d x0( 1.0, 1.0 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 0 );
        dx( 1 ) = -x( 1 );
    };

    auto sol = ode::taylorinteg< N >( f, x0, t0, tmax, abstol );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back()( 0 ), std::exp( tmax ), 1e-14 );
    EXPECT_NEAR( sol.x.back()( 1 ), std::exp( -tmax ), 1e-14 );
}

// Kepler problem (2D): two-body problem in Cartesian coordinates
// dx/dt = vx, dy/dt = vy, dvx/dt = -x/r^3, dvy/dt = -y/r^3
// Circular orbit: x0=(1,0,0,1), period = 2*pi
TEST( TaylorIntegratorVector, KeplerCircularOrbit )
{
    constexpr int N = 25;
    const double t0 = 0.0;
    const double tmax = 2.0 * std::numbers::pi;
    const double abstol = 1e-20;

    Eigen::Vector< double, 4 > x0;
    x0 << 1.0, 0.0, 0.0, 1.0;  // x, y, vx, vy

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        using std::sqrt;
        auto r2 = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 );
        auto r = sqrt( r2 );
        auto r3 = r2 * r;
        dx( 0 ) = x( 2 );
        dx( 1 ) = x( 3 );
        dx( 2 ) = -x( 0 ) / r3;
        dx( 3 ) = -x( 1 ) / r3;
    };

    auto sol = ode::taylorinteg< N >( f, x0, t0, tmax, abstol );

    // After one full orbit, should return to initial conditions
    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 1 ), 0.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 2 ), 0.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 3 ), 1.0, 1e-10 );

    // Check energy conservation: E = 0.5*v^2 - 1/r = -0.5
    double r =
        std::sqrt( sol.x.back()( 0 ) * sol.x.back()( 0 ) + sol.x.back()( 1 ) * sol.x.back()( 1 ) );
    double v2 = sol.x.back()( 2 ) * sol.x.back()( 2 ) + sol.x.back()( 3 ) * sol.x.back()( 3 );
    double energy = 0.5 * v2 - 1.0 / r;
    EXPECT_NEAR( energy, -0.5, 1e-10 );
}

// =============================================================================
// Vector ODE with trange
// =============================================================================

TEST( TaylorIntegratorVector, HarmonicOscillatorTrange )
{
    constexpr int N = 25;
    const double abstol = 1e-20;

    Eigen::Vector2d x0( 1.0, 0.0 );

    std::vector< double > trange;
    for ( int i = 0; i <= 20; ++i ) trange.push_back( i * 0.1 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    auto sol = ode::taylorinteg< N >( f, x0, trange, abstol );

    ASSERT_EQ( sol.x.size(), trange.size() );

    for ( std::size_t i = 0; i < trange.size(); ++i )
    {
        EXPECT_NEAR( sol.x[i]( 0 ), std::cos( trange[i] ), 1e-14 ) << "  x1 at t=" << trange[i];
        EXPECT_NEAR( sol.x[i]( 1 ), -std::sin( trange[i] ), 1e-14 ) << "  x2 at t=" << trange[i];
    }
}

// =============================================================================
// Lower-level API tests
// =============================================================================

// Test jetcoeffs directly: dx/dt = x with x0 = 1 → x[k] = 1/k!
TEST( TaylorIntegratorJetcoeffs, ScalarExponential )
{
    constexpr int N = 10;
    using TTE = TE< N >;

    TTE t_da{};
    t_da[0] = 0.0;
    t_da[1] = 1.0;

    TTE x_da{};
    x_da[0] = 1.0;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    ode::jetcoeffs( x_da, t_da, f );

    // x[k] should be 1/k!
    double factorial = 1.0;
    for ( int k = 0; k <= N; ++k )
    {
        if ( k > 0 ) factorial *= k;
        EXPECT_NEAR( x_da[k], 1.0 / factorial, 1e-15 ) << "  coeff k=" << k;
    }
}

// Test stepsize
TEST( TaylorIntegratorStepsize, BasicStepsize )
{
    constexpr int N = 10;
    using TTE = TE< N >;

    TTE x_da{};
    x_da[0] = 1.0;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    TTE t_da{};
    t_da[0] = 0.0;
    t_da[1] = 1.0;

    ode::jetcoeffs( x_da, t_da, f );

    double h = ode::stepsize( x_da, 1e-20 );
    EXPECT_GT( h, 0.0 );
    EXPECT_LT( h, 100.0 );  // Sanity check
}
