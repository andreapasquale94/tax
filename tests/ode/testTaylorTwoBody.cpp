#include <Eigen/Core>
#include <cmath>
#include <tax/ode/taylor.hpp>

#include "../testUtils.hpp"

using namespace tax;
using namespace tax::ode;

namespace
{

auto kepler_rhs = []( auto /*t*/, auto y ) -> decltype( y ) {
    auto r2 = y( 0 ) * y( 0 ) + y( 1 ) * y( 1 );  // r^2
    auto r3 = r2 * sqrt( r2 );                    // r^3

    decltype( y ) out( 4 );
    out( 0 ) = y( 2 );
    out( 1 ) = y( 3 );
    out( 2 ) = -y( 0 ) / r3;
    out( 3 ) = -y( 1 ) / r3;
    return out;
};

}  // namespace

TEST( TwoBody, SingleStep )
{
    constexpr int N = 14;

    Eigen::Vector4d y0{ 1.0, 0.0, 0.0, 1.0 };
    auto integrator = makeTaylorIntegrator< N >( kepler_rhs );

    const double h = 0.3;
    auto y1 = integrator.step( 0.0, y0, h );

    EXPECT_NEAR( y1( 0 ), std::cos( h ), 1e-9 );
    EXPECT_NEAR( y1( 1 ), std::sin( h ), 1e-9 );
    EXPECT_NEAR( y1( 2 ), -std::sin( h ), 1e-9 );
    EXPECT_NEAR( y1( 3 ), std::cos( h ), 1e-9 );
}

TEST( TwoBody, FullPeriod )
{
    constexpr int N = 16;

    Eigen::Vector4d y0{ 1.0, 0.0, 0.0, 1.0 };
    const double T = 2.0 * M_PI;

    TaylorIntegratorOptions options;
    options.atol = 1e-10;
    options.rtol = 1e-10;

    auto integrator = makeTaylorIntegrator< N >( kepler_rhs, options );
    auto result = integrator.integrate( 0.0, T, y0, 0.5 );

    ASSERT_FALSE( result.t.empty() );
    EXPECT_NEAR( result.t.back(), T, 1e-12 );

    const auto& yf = result.y.back();
    EXPECT_NEAR( yf( 0 ), 1.0, 1e-8 );
    EXPECT_NEAR( yf( 1 ), 0.0, 1e-8 );
    EXPECT_NEAR( yf( 2 ), 0.0, 1e-8 );
    EXPECT_NEAR( yf( 3 ), 1.0, 1e-8 );
}

TEST( TwoBody, ConservedQuantities )
{
    constexpr int N = 16;

    Eigen::Vector4d y0{ 1.0, 0.0, 0.0, 1.0 };
    const double T = 2.0 * M_PI;

    const double E0 = 0.5 * ( y0( 2 ) * y0( 2 ) + y0( 3 ) * y0( 3 ) ) -
                      1.0 / std::sqrt( y0( 0 ) * y0( 0 ) + y0( 1 ) * y0( 1 ) );
    const double L0 = y0( 0 ) * y0( 3 ) - y0( 1 ) * y0( 2 );

    TaylorIntegratorOptions options;
    options.atol = 1e-10;
    options.rtol = 1e-10;

    auto integrator = makeTaylorIntegrator< N >( kepler_rhs, options );
    auto result = integrator.integrate( 0.0, T, y0, 0.5 );

    for ( std::size_t k = 0; k < result.t.size(); ++k )
    {
        const auto& y = result.y[k];
        const double r = std::sqrt( y( 0 ) * y( 0 ) + y( 1 ) * y( 1 ) );
        const double E = 0.5 * ( y( 2 ) * y( 2 ) + y( 3 ) * y( 3 ) ) - 1.0 / r;
        const double L = y( 0 ) * y( 3 ) - y( 1 ) * y( 2 );

        EXPECT_NEAR( E, E0, 1e-7 ) << "energy drift at step " << k;
        EXPECT_NEAR( L, L0, 1e-7 ) << "angular momentum drift at step " << k;
    }
}
