#include <Eigen/Core>
#include <cmath>
#include <tax/ode/taylor.hpp>

#include "../testUtils.hpp"

using namespace tax;
using namespace tax::ode;

// y' = y,  y(0)=1  =>  y(t) = e^t
TEST( TaylorStep, ScalarExponential )
{
    constexpr int N = 10;

    auto rhs = []( auto /*t*/, auto y ) -> decltype( y ) {
        decltype( y ) out( 1 );
        out( 0 ) = y( 0 );
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 1.0;

    auto integrator = makeTaylorIntegrator< N >( rhs );
    const double h = 0.5;
    auto y1 = integrator.step( 0.0, y0, h );

    EXPECT_NEAR( y1( 0 ), std::exp( h ), 1e-10 );
}

// y1' = y2,  y2' = -y1,  y(0)=(1,0)  =>  (cos t, -sin t)
TEST( TaylorStep, HarmonicOscillator )
{
    constexpr int N = 12;

    auto rhs = []( auto /*t*/, auto y ) -> decltype( y ) {
        decltype( y ) out( 2 );
        out( 0 ) = y( 1 );
        out( 1 ) = -y( 0 );
        return out;
    };

    Eigen::Vector2d y0{ 1.0, 0.0 };

    auto integrator = makeTaylorIntegrator< N >( rhs );
    const double h = 0.3;
    auto y1 = integrator.step( 0.0, y0, h );

    EXPECT_NEAR( y1( 0 ), std::cos( h ), 1e-10 );
    EXPECT_NEAR( y1( 1 ), -std::sin( h ), 1e-10 );
}

// Non-autonomous: y' = t*y,  y(0)=1  =>  y(t) = exp(t^2/2)
TEST( TaylorStep, NonAutonomousScalar )
{
    constexpr int N = 12;

    auto rhs = []( auto t, auto y ) -> decltype( y ) {
        decltype( y ) out( 1 );
        out( 0 ) = t * y( 0 );
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 1.0;

    auto integrator = makeTaylorIntegrator< N >( rhs );
    const double h = 0.4;
    auto y1 = integrator.step( 0.0, y0, h );

    EXPECT_NEAR( y1( 0 ), std::exp( h * h / 2.0 ), 1e-10 );
}

TEST( TaylorStep, FixedSizeVector )
{
    constexpr int N = 10;

    auto rhs = []( auto /*t*/, auto y ) -> decltype( y ) {
        decltype( y ) out( 3 );
        out( 0 ) = 1.0 * y( 0 );
        out( 1 ) = 2.0 * y( 1 );
        out( 2 ) = 3.0 * y( 2 );
        return out;
    };

    Eigen::Vector3d y0{ 1.0, 1.0, 1.0 };

    auto integrator = makeTaylorIntegrator< N >( rhs );
    const double h = 0.1;
    auto y1 = integrator.step( 0.0, y0, h );

    EXPECT_NEAR( y1( 0 ), std::exp( 1.0 * h ), 1e-10 );
    EXPECT_NEAR( y1( 1 ), std::exp( 2.0 * h ), 1e-10 );
    EXPECT_NEAR( y1( 2 ), std::exp( 3.0 * h ), 1e-10 );
}
