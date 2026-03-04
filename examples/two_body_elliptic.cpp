/**
 * @file examples/two_body_elliptic.cpp
 * @brief Planar elliptical two-body orbit with Taylor ODE integration.
 *
 * State y = (x, y, vx, vy), mu = 1:
 *   dx/dt  = vx
 *   dy/dt  = vy
 *   dvx/dt = -x / r^3
 *   dvy/dt = -y / r^3
 *   r      = sqrt(x^2 + y^2)
 *
 * Initial condition at periapsis for chosen (a, e):
 *   rp = a(1-e),  ra = a(1+e)
 *   vp = sqrt(mu * (2/rp - 1/a))
 *   y0 = (rp, 0, 0, vp)
 *
 * Build & run:
 *   cmake -S . -B build -DTAX_BUILD_EXAMPLES=ON
 *   cmake --build build --target two_body_elliptic
 *   ./build/examples/two_body_elliptic
 */

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tax/ode/taylor.hpp>

using namespace tax;
using namespace tax::ode;

static auto kepler_rhs = []( auto /*t*/, auto y ) -> decltype( y ) {
    auto r2 = y( 0 ) * y( 0 ) + y( 1 ) * y( 1 );
    auto r3 = r2 * sqrt( r2 );

    decltype( y ) out( 4 );
    out( 0 ) = y( 2 );
    out( 1 ) = y( 3 );
    out( 2 ) = -y( 0 ) / r3;
    out( 3 ) = -y( 1 ) / r3;
    return out;
};

template < typename Derived >
static double radius( const Eigen::MatrixBase< Derived >& y )
{
    return std::hypot( y( 0 ), y( 1 ) );
}

template < typename Derived >
static double energy( const Eigen::MatrixBase< Derived >& y )
{
    const double r = radius( y );
    return 0.5 * ( y( 2 ) * y( 2 ) + y( 3 ) * y( 3 ) ) - 1.0 / r;
}

template < typename Derived >
static double angular_momentum( const Eigen::MatrixBase< Derived >& y )
{
    return y( 0 ) * y( 3 ) - y( 1 ) * y( 2 );
}

int main()
{
    constexpr int N = 50;
    constexpr double mu = 1.0;

    // Ellipse parameters.
    const double a = 1.0;
    const double e = 0.6;
    const double rp = a * ( 1.0 - e );
    const double ra = a * ( 1.0 + e );
    const double vp = std::sqrt( mu * ( 2.0 / rp - 1.0 / a ) );
    const double period = 2.0 * M_PI * std::sqrt( a * a * a / mu );

    // Start at periapsis.
    Eigen::Vector4d y0{ rp, 0.0, 0.0, vp };

    TaylorIntegratorOptions options;
    options.atol = 1e-14;
    options.rtol = 1e-14;

    const int nperiod = 50;
    const double tf = nperiod * period;
    const double h0 = 1e-6;

    auto integrator = makeTaylorIntegrator< N >( kepler_rhs, options );
    auto result = integrator.integrate( 0.0, tf, y0, h0 );

    const double E0 = energy( y0 );
    const double H0 = angular_momentum( y0 );

    double r_min = std::numeric_limits< double >::infinity();
    double r_max = 0.0;
    double max_abs_dE = 0.0;
    double max_abs_dH = 0.0;

    for ( const auto& y : result.y )
    {
        const double r = radius( y );
        const double dE = energy( y ) - E0;
        const double dH = angular_momentum( y ) - H0;

        r_min = std::min( r_min, r );
        r_max = std::max( r_max, r );
        max_abs_dE = std::max( max_abs_dE, std::abs( dE ) );
        max_abs_dH = std::max( max_abs_dH, std::abs( dH ) );
    }

    const auto& yf = result.y.back();
    const double final_state_err = ( yf - y0 ).norm();

    std::cout << std::scientific << std::setprecision( 6 );
    std::cout << "Elliptic planar two-body orbit (mu=1)\n";
    std::cout << "a=" << a << ", e=" << e << ", rp=" << rp << ", ra=" << ra << "\n";
    std::cout << "period=" << period << ", propagated tf=" << tf << "\n";
    std::cout << "accepted steps=" << ( result.t.size() - 1 ) << "\n\n";

    std::cout << "Max |dE| over trajectory: " << max_abs_dE << "\n";
    std::cout << "Max |dH| over trajectory: " << max_abs_dH << "\n";
    std::cout << "||y(tf)-y0|| " << nperiod << " periods): " << final_state_err << "\n";

    return 0;
}
