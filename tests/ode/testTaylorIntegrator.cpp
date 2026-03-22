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

    auto sol = ode::integrate< N >( f, x0, t0, tmax, abstol );

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

    auto sol = ode::integrate< N >( f, x0, t0, tmax, abstol );

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

    auto sol = ode::integrate< N >( f, x0, t0, tmax, abstol );

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

    auto sol = ode::integrate< N >( f, x0, t0, tmax, abstol );

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

    auto sol = ode::integrate< N >( f, x0, t0, tmax, abstol );

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

    auto sol = ode::integrate< N >( f, x0, trange, abstol );

    ASSERT_EQ( sol.t.size(), trange.size() );
    ASSERT_EQ( sol.x.size(), trange.size() );

    for ( std::size_t i = 0; i < trange.size(); ++i )
    {
        EXPECT_NEAR( sol.x[i], std::exp( trange[i] ), 1e-14 ) << "  at t=" << trange[i];
    }
}

// =============================================================================
// Dense output (scalar)
// =============================================================================

TEST( TaylorIntegratorScalar, DenseOutput )
{
    constexpr int N = 25;
    const double x0 = 1.0;
    const double t0 = 0.0;
    const double tmax = 2.0;
    const double abstol = 1e-20;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto sol = ode::integrate< N >( f, x0, t0, tmax, abstol );

    // Polynomials should be stored
    EXPECT_FALSE( sol.p.empty() );
    EXPECT_EQ( sol.p.size(), sol.t.size() - 1 );

    // Evaluate at arbitrary intermediate times via operator()
    for ( double t = 0.0; t <= 2.0; t += 0.07 )
    {
        EXPECT_NEAR( sol( t ), std::exp( t ), 1e-13 ) << "  at t=" << t;
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
    const double t0 = 0.0;
    const double tmax = 2.0 * std::numbers::pi;
    const double abstol = 1e-20;

    Eigen::Vector2d x0( 1.0, 0.0 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    auto sol = ode::integrate< N >( f, x0, t0, tmax, abstol );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );

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

    auto sol = ode::integrate< N >( f, x0, t0, tmax, abstol );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back()( 0 ), std::exp( tmax ), 1e-14 );
    EXPECT_NEAR( sol.x.back()( 1 ), std::exp( -tmax ), 1e-14 );
}

// Kepler problem (2D): two-body in Cartesian coordinates
// Circular orbit: x0=(1,0,0,1), period = 2π
TEST( TaylorIntegratorVector, KeplerCircularOrbit )
{
    constexpr int N = 25;
    const double t0 = 0.0;
    const double tmax = 2.0 * std::numbers::pi;
    const double abstol = 1e-20;

    Eigen::Vector< double, 4 > x0;
    x0 << 1.0, 0.0, 0.0, 1.0;

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

    auto sol = ode::integrate< N >( f, x0, t0, tmax, abstol );

    // After one full orbit
    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 1 ), 0.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 2 ), 0.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 3 ), 1.0, 1e-10 );

    // Energy conservation: E = 0.5*v² - 1/r = -0.5
    const auto& xf = sol.x.back();
    double r = std::sqrt( xf( 0 ) * xf( 0 ) + xf( 1 ) * xf( 1 ) );
    double v2 = xf( 2 ) * xf( 2 ) + xf( 3 ) * xf( 3 );
    EXPECT_NEAR( 0.5 * v2 - 1.0 / r, -0.5, 1e-10 );
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

    auto sol = ode::integrate< N >( f, x0, trange, abstol );

    ASSERT_EQ( sol.x.size(), trange.size() );

    for ( std::size_t i = 0; i < trange.size(); ++i )
    {
        EXPECT_NEAR( sol.x[i]( 0 ), std::cos( trange[i] ), 1e-14 ) << "  x1 at t=" << trange[i];
        EXPECT_NEAR( sol.x[i]( 1 ), -std::sin( trange[i] ), 1e-14 ) << "  x2 at t=" << trange[i];
    }
}

// =============================================================================
// Dense output (vector)
// =============================================================================

TEST( TaylorIntegratorVector, DenseOutput )
{
    constexpr int N = 25;
    const double abstol = 1e-20;

    Eigen::Vector2d x0( 1.0, 0.0 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    auto sol = ode::integrate< N >( f, x0, 0.0, 3.0, abstol );

    EXPECT_FALSE( sol.p.empty() );

    for ( double t = 0.0; t <= 3.0; t += 0.13 )
    {
        auto y = sol( t );
        EXPECT_NEAR( y( 0 ), std::cos( t ), 1e-13 ) << "  x1 at t=" << t;
        EXPECT_NEAR( y( 1 ), -std::sin( t ), 1e-13 ) << "  x2 at t=" << t;
    }
}

// =============================================================================
// Low-level API: step returns TTE
// =============================================================================

// step() should return a TTE whose coefficients are 1/k! for dx/dt = x
TEST( TaylorIntegratorStep, ScalarReturnsPolynomial )
{
    constexpr int N = 10;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto [p, h] = ode::step< N >( f, 1.0, 0.0, 1e-20 );

    // p[k] should be 1/k!
    double factorial = 1.0;
    for ( int k = 0; k <= N; ++k )
    {
        if ( k > 0 ) factorial *= k;
        EXPECT_NEAR( p[k], 1.0 / factorial, 1e-15 ) << "  coeff k=" << k;
    }

    // h should be positive and finite
    EXPECT_GT( h, 0.0 );
    EXPECT_LT( h, 100.0 );
}

TEST( TaylorIntegratorStep, ScalarPolynomialEval )
{
    constexpr int N = 25;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto [p, h] = ode::step< N >( f, 1.0, 0.0, 1e-20 );

    // Evaluating the polynomial at h should give exp(h)
    EXPECT_NEAR( p.eval( h ), std::exp( h ), 1e-14 );
    EXPECT_NEAR( p.eval( 0.5 ), std::exp( 0.5 ), 1e-14 );
}
