#include "testUtils.hpp"
#include <tax/ode/taylor.hpp>

#include <Eigen/Core>
#include <cmath>

using namespace tax;
using namespace tax::ode;

// =============================================================================
// taylorStep — single step accuracy
// =============================================================================

// y' = y,  y(0)=1  =>  y(t) = e^t
// At order N the Taylor method is exact for polynomials of degree <= N,
// so the error should be O(h^(N+1)).
TEST( TaylorStep, ScalarExponential )
{
    constexpr int N = 10;

    auto rhs = []( auto t, auto y ) -> decltype( y )
    {
        decltype( y ) out( 1 );
        out( 0 ) = y( 0 );   // f(t,y) = y  (independent of t)
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 1.0;

    const double h    = 0.5;
    auto         y1   = taylorStep< N >( rhs, 0.0, y0, h );
    const double exact = std::exp( h );

    EXPECT_NEAR( y1( 0 ), exact, 1e-10 );
}

// y1' = y2,  y2' = -y1,  y(0)=(1,0)  =>  (cos t, -sin t)
TEST( TaylorStep, HarmonicOscillator )
{
    constexpr int N = 12;

    auto rhs = []( auto t, auto y ) -> decltype( y )
    {
        decltype( y ) out( 2 );
        out( 0 ) =  y( 1 );
        out( 1 ) = -y( 0 );
        return out;
    };

    Eigen::Vector2d y0{ 1.0, 0.0 };

    const double h  = 0.3;
    auto         y1 = taylorStep< N >( rhs, 0.0, y0, h );

    EXPECT_NEAR( y1( 0 ),  std::cos( h ), 1e-10 );
    EXPECT_NEAR( y1( 1 ), -std::sin( h ), 1e-10 );
}

// Non-autonomous: y' = t*y,  y(0)=1  =>  y(t) = exp(t^2/2)
TEST( TaylorStep, NonAutonomousScalar )
{
    constexpr int N = 12;

    auto rhs = []( auto t, auto y ) -> decltype( y )
    {
        decltype( y ) out( 1 );
        out( 0 ) = t * y( 0 );
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 1.0;

    const double h     = 0.4;
    auto         y1    = taylorStep< N >( rhs, 0.0, y0, h );
    const double exact = std::exp( h * h / 2.0 );

    EXPECT_NEAR( y1( 0 ), exact, 1e-10 );
}

// Fixed-size Eigen vector (Vector3d) dispatching
TEST( TaylorStep, FixedSizeVector )
{
    constexpr int N = 10;

    // Three decoupled exponentials: y'_i = (i+1)*y_i
    auto rhs = []( auto t, auto y ) -> decltype( y )
    {
        decltype( y ) out( 3 );
        out( 0 ) = 1.0 * y( 0 );
        out( 1 ) = 2.0 * y( 1 );
        out( 2 ) = 3.0 * y( 2 );
        return out;
    };

    Eigen::Vector3d y0{ 1.0, 1.0, 1.0 };

    const double h  = 0.1;
    auto         y1 = taylorStep< N >( rhs, 0.0, y0, h );

    EXPECT_NEAR( y1( 0 ), std::exp( 1.0 * h ), 1e-10 );
    EXPECT_NEAR( y1( 1 ), std::exp( 2.0 * h ), 1e-10 );
    EXPECT_NEAR( y1( 2 ), std::exp( 3.0 * h ), 1e-10 );
}

// =============================================================================
// taylorIntegrate — full trajectory with adaptive step size
// =============================================================================

TEST( TaylorIntegrate, ScalarExponential )
{
    constexpr int N = 10;

    auto rhs = []( auto t, auto y ) -> decltype( y )
    {
        decltype( y ) out( 1 );
        out( 0 ) = y( 0 );
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 1.0;

    auto result = taylorIntegrate< N >( rhs, 0.0, 2.0, y0, 0.5, 1e-10, 1e-10 );

    ASSERT_FALSE( result.t.empty() );
    EXPECT_NEAR( result.t.back(), 2.0, 1e-12 );
    EXPECT_NEAR( result.y.back()( 0 ), std::exp( 2.0 ), 1e-8 );
}

TEST( TaylorIntegrate, HarmonicOscillator )
{
    constexpr int N = 12;

    auto rhs = []( auto t, auto y ) -> decltype( y )
    {
        decltype( y ) out( 2 );
        out( 0 ) =  y( 1 );
        out( 1 ) = -y( 0 );
        return out;
    };

    Eigen::Vector2d y0{ 1.0, 0.0 };

    auto result = taylorIntegrate< N >( rhs, 0.0, 2.0, y0, 0.5, 1e-10, 1e-10 );

    ASSERT_FALSE( result.t.empty() );
    const double tf = result.t.back();
    EXPECT_NEAR( tf, 2.0, 1e-12 );
    EXPECT_NEAR( result.y.back()( 0 ),  std::cos( tf ), 1e-8 );
    EXPECT_NEAR( result.y.back()( 1 ), -std::sin( tf ), 1e-8 );
}

// Check that the adaptive step size keeps all intermediate points accurate
TEST( TaylorIntegrate, TrajectoryAccuracy )
{
    constexpr int N = 10;

    // y' = y,  y(0)=1
    auto rhs = []( auto t, auto y ) -> decltype( y )
    {
        decltype( y ) out( 1 );
        out( 0 ) = y( 0 );
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 1.0;

    auto result = taylorIntegrate< N >( rhs, 0.0, 1.0, y0, 0.2, 1e-10, 1e-10 );

    for ( std::size_t k = 0; k < result.t.size(); ++k )
    {
        const double t     = result.t[k];
        const double exact = std::exp( t );
        EXPECT_NEAR( result.y[k]( 0 ), exact, 1e-8 )
            << "  at t=" << t << " (step " << k << ")";
    }
}

// Verify initial condition is the first entry
TEST( TaylorIntegrate, InitialConditionIncluded )
{
    constexpr int N = 5;

    auto rhs = []( auto t, auto y ) -> decltype( y )
    {
        decltype( y ) out( 1 );
        out( 0 ) = y( 0 );
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 3.14;

    auto result = taylorIntegrate< N >( rhs, 0.5, 1.0, y0, 0.1 );

    ASSERT_GE( result.t.size(), 1u );
    EXPECT_NEAR( result.t.front(), 0.5, 1e-14 );
    EXPECT_NEAR( result.y.front()( 0 ), 3.14, 1e-14 );
}

// =============================================================================
// Two-body (Kepler) problem
//
// State: [x, y, vx, vy]
// RHS:   x'=vx,  y'=vy,  vx'=-x/r³,  vy'=-y/r³   (μ = 1)
//
// Circular orbit with r₀ = 1, period T = 2π:
//   x(t) = cos(t),   y(t) = sin(t)
//  vx(t) = -sin(t), vy(t) = cos(t)
//
// Conserved quantities:
//   Energy: E = (vx²+vy²)/2 - 1/r = -0.5
//   Angular momentum: L = x*vy - y*vx = 1.0
// =============================================================================

namespace
{
// RHS of the two-body problem (μ = 1). Works with any DA<N> or double.
auto kepler_rhs = []( auto t, auto y ) -> decltype( y )
{
    auto r2 = y( 0 ) * y( 0 ) + y( 1 ) * y( 1 );  // r²
    auto r3 = r2 * sqrt( r2 );                       // r³
    decltype( y ) out( 4 );
    out( 0 ) =  y( 2 );
    out( 1 ) =  y( 3 );
    out( 2 ) = -y( 0 ) / r3;
    out( 3 ) = -y( 1 ) / r3;
    return out;
};
}  // namespace

// Single Taylor step for the circular orbit
TEST( TwoBody, SingleStep )
{
    constexpr int N = 14;

    // Circular orbit: x=1, y=0, vx=0, vy=1
    Eigen::Vector4d y0{ 1.0, 0.0, 0.0, 1.0 };

    const double h  = 0.3;
    auto         y1 = taylorStep< N >( kepler_rhs, 0.0, y0, h );

    EXPECT_NEAR( y1( 0 ),  std::cos( h ), 1e-9 );   // x
    EXPECT_NEAR( y1( 1 ),  std::sin( h ), 1e-9 );   // y
    EXPECT_NEAR( y1( 2 ), -std::sin( h ), 1e-9 );   // vx
    EXPECT_NEAR( y1( 3 ),  std::cos( h ), 1e-9 );   // vy
}

// Integrate for one full period and verify return to initial state
TEST( TwoBody, FullPeriod )
{
    constexpr int N = 16;

    Eigen::Vector4d y0{ 1.0, 0.0, 0.0, 1.0 };
    const double    T  = 2.0 * M_PI;

    auto result = taylorIntegrate< N >( kepler_rhs, 0.0, T, y0, 0.5, 1e-10, 1e-10 );

    ASSERT_FALSE( result.t.empty() );
    EXPECT_NEAR( result.t.back(), T, 1e-12 );

    const auto& yf = result.y.back();
    EXPECT_NEAR( yf( 0 ),  1.0, 1e-8 );   // x → 1
    EXPECT_NEAR( yf( 1 ),  0.0, 1e-8 );   // y → 0
    EXPECT_NEAR( yf( 2 ),  0.0, 1e-8 );   // vx → 0
    EXPECT_NEAR( yf( 3 ),  1.0, 1e-8 );   // vy → 1
}

// Energy and angular momentum are conserved at every reported step
TEST( TwoBody, ConservedQuantities )
{
    constexpr int N = 16;

    Eigen::Vector4d y0{ 1.0, 0.0, 0.0, 1.0 };
    const double    T  = 2.0 * M_PI;

    // Reference conserved values for the circular orbit
    const double E0 = 0.5 * ( y0( 2 ) * y0( 2 ) + y0( 3 ) * y0( 3 ) )
                      - 1.0 / std::sqrt( y0( 0 ) * y0( 0 ) + y0( 1 ) * y0( 1 ) );
    const double L0 = y0( 0 ) * y0( 3 ) - y0( 1 ) * y0( 2 );

    auto result = taylorIntegrate< N >( kepler_rhs, 0.0, T, y0, 0.5, 1e-10, 1e-10 );

    for ( std::size_t k = 0; k < result.t.size(); ++k )
    {
        const auto& y   = result.y[k];
        const double r   = std::sqrt( y( 0 ) * y( 0 ) + y( 1 ) * y( 1 ) );
        const double E   = 0.5 * ( y( 2 ) * y( 2 ) + y( 3 ) * y( 3 ) ) - 1.0 / r;
        const double L   = y( 0 ) * y( 3 ) - y( 1 ) * y( 2 );

        EXPECT_NEAR( E, E0, 1e-7 ) << "  energy drift at step " << k;
        EXPECT_NEAR( L, L0, 1e-7 ) << "  angular momentum drift at step " << k;
    }
}
