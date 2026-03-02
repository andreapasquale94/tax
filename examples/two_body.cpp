/**
 * @file examples/two_body.cpp
 * @brief Two-body (Kepler) problem integrated with the Taylor ODE method.
 *
 * Demonstrates use of tax::ode::TaylorIntegrator for a nonlinear autonomous
 * ODE with a known analytical solution.  The Keplerian orbit conserves energy
 * and angular momentum, so both quantities are printed alongside the trajectory
 * to illustrate the long-term accuracy of the high-order Taylor integrator.
 *
 * Problem
 * -------
 * Two bodies with gravitational parameter μ = 1.  One body fixed at origin,
 * the other at position (x, y) with velocity (vx, vy):
 *
 *   dx/dt  =  vx
 *   dy/dt  =  vy
 *   dvx/dt = -x / r³
 *   dvy/dt = -y / r³
 *
 *   r = sqrt(x² + y²)
 *
 * Circular orbit (r₀ = 1):
 *   IC:       x=1, y=0, vx=0, vy=1
 *   Period:   T = 2π
 *   Solution: x(t)=cos(t), y(t)=sin(t), vx=-sin(t), vy=cos(t)
 *
 * Build & run (from the repo root with Eigen enabled)
 * ---------------------------------------------------
 *   mkdir -p build && cd build
 *   cmake .. -DTAX_ENABLE_EIGEN=ON -DTAX_BUILD_EXAMPLES=ON
 *   make two_body
 *   ./examples/two_body
 */

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tax/ode/taylor.hpp>

using namespace tax;
using namespace tax::ode;

// ---------------------------------------------------------------------------
// Kepler RHS — works with DA<N> (auto t/y) as well as plain doubles
// ---------------------------------------------------------------------------
static auto kepler_rhs = []( auto /*t*/, auto y ) -> decltype( y ) {
    auto r2 = y( 0 ) * y( 0 ) + y( 1 ) * y( 1 );
    auto r3 = r2 * sqrt( r2 );  // tax::sqrt propagates through DA
    decltype( y ) f( 4 );
    f( 0 ) = y( 2 );
    f( 1 ) = y( 3 );
    f( 2 ) = -y( 0 ) / r3;
    f( 3 ) = -y( 1 ) / r3;
    return f;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
template < typename Derived >
static double energy( const Eigen::MatrixBase< Derived >& y )
{
    const double r = std::sqrt( y( 0 ) * y( 0 ) + y( 1 ) * y( 1 ) );
    return 0.5 * ( y( 2 ) * y( 2 ) + y( 3 ) * y( 3 ) ) - 1.0 / r;
}

template < typename Derived >
static double angmom( const Eigen::MatrixBase< Derived >& y )
{
    return y( 0 ) * y( 3 ) - y( 1 ) * y( 2 );
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    constexpr int N = 40;

    // Circular orbit initial conditions (μ = 1, r₀ = 1)
    Eigen::Vector4d y0{ 1.0, 0.0, 0.0, 1.0 };

    const int nperiod = 50;
    const double T = nperiod * 2.0 * M_PI;  // 5 orbital periods
    const double h0 = 0.5;                  // initial step guess
    const double atol = 1e-12;
    const double rtol = 1e-12;

    std::cout << "Two-body (Kepler) problem  —  Taylor integrator order " << N << "\n";
    std::cout << "Integrating for " << nperiod << " periods  (T = " << 2.0 * M_PI
              << " per orbit)\n";
    std::cout << "Tolerance: atol = rtol = " << atol << "\n\n";

    TaylorIntegratorOptions options;
    options.atol = atol;
    options.rtol = rtol;
    auto integrator = makeTaylorIntegrator< N >( kepler_rhs, options );
    auto result = integrator.integrate( 0.0, T, y0, h0 );

    const double E0 = energy( result.y.front() );
    const double L0 = angmom( result.y.front() );

    // Print header
    std::cout << std::setw( 10 ) << "t" << std::setw( 14 ) << "x" << std::setw( 14 ) << "y"
              << std::setw( 14 ) << "x_exact" << std::setw( 14 ) << "y_exact" << std::setw( 14 )
              << "pos_error" << std::setw( 14 ) << "dE" << std::setw( 14 ) << "dL"
              << "\n";
    std::cout << std::string( 112, '-' ) << "\n";

    std::cout << std::scientific << std::setprecision( 5 );

    // Print every ~10th point to keep output readable
    const std::size_t stride = std::max( std::size_t{ 1 }, result.t.size() / 50 );

    for ( std::size_t k = 0; k < result.t.size(); k += stride )
    {
        const double t = result.t[k];
        const auto& y = result.y[k];
        const double x_ex = std::cos( t );
        const double y_ex = std::sin( t );
        const double perr = std::hypot( y( 0 ) - x_ex, y( 1 ) - y_ex );
        const double dE = energy( y ) - E0;
        const double dL = angmom( y ) - L0;

        std::cout << std::setw( 10 ) << t << std::setw( 14 ) << y( 0 ) << std::setw( 14 ) << y( 1 )
                  << std::setw( 14 ) << x_ex << std::setw( 14 ) << y_ex << std::setw( 14 ) << perr
                  << std::setw( 14 ) << dE << std::setw( 14 ) << dL << "\n";
    }

    // Always print the final point
    {
        const double t = result.t.back();
        const auto& y = result.y.back();
        const double x_ex = std::cos( t );
        const double y_ex = std::sin( t );
        const double perr = std::hypot( y( 0 ) - x_ex, y( 1 ) - y_ex );
        const double dE = energy( y ) - E0;
        const double dL = angmom( y ) - L0;

        std::cout << std::setw( 10 ) << t << std::setw( 14 ) << y( 0 ) << std::setw( 14 ) << y( 1 )
                  << std::setw( 14 ) << x_ex << std::setw( 14 ) << y_ex << std::setw( 14 ) << perr
                  << std::setw( 14 ) << dE << std::setw( 14 ) << dL << "\n";
    }

    std::cout << "\nTotal steps: " << result.t.size() - 1 << "\n";
    std::cout << "Final position error: "
              << std::hypot( result.y.back()( 0 ) - std::cos( T ),
                             result.y.back()( 1 ) - std::sin( T ) )
              << "\n";
    std::cout << "Final energy error:   " << std::abs( energy( result.y.back() ) - E0 ) << "\n";
    std::cout << "Final ang-mom error:  " << std::abs( angmom( result.y.back() ) - L0 ) << "\n";

    return 0;
}
