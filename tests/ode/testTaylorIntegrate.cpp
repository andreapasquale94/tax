#include "../testUtils.hpp"
#include <tax/ode/taylor.hpp>

#include <Eigen/Core>
#include <cmath>
#include <type_traits>

using namespace tax;
using namespace tax::ode;

namespace
{

template < int N >
struct ConstantStepController
{
    using options_type = TaylorIntegratorOptions;
    static constexpr int order = N;

    explicit ConstantStepController( options_type options ) : options_( options ) {}

    [[nodiscard]] const options_type& options() const noexcept { return options_; }

    template < typename Vec, typename Series >
    [[nodiscard]] double
    nextStep( double h, double /*tf*/, const Vec& /*y*/, const Series& /*yDA*/ ) const
    {
        return h;
    }

  private:
    options_type options_;
};

}  // namespace

TEST( TaylorIntegrate, ScalarExponential )
{
    constexpr int N = 10;

    auto rhs = []( auto /*t*/, auto y ) -> decltype( y )
    {
        decltype( y ) out( 1 );
        out( 0 ) = y( 0 );
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 1.0;

    TaylorIntegratorOptions options;
    options.atol = 1e-10;
    options.rtol = 1e-10;

    auto integrator = makeTaylorIntegrator< N >( rhs, options );
    auto result     = integrator.integrate( 0.0, 2.0, y0, 0.5 );

    static_assert( std::is_same_v< decltype( result ), Solution< Eigen::Matrix< double, 1, 1 > > > );

    ASSERT_FALSE( result.t.empty() );
    EXPECT_NEAR( result.t.back(), 2.0, 1e-12 );
    EXPECT_NEAR( result.y.back()( 0 ), std::exp( 2.0 ), 1e-8 );
}

TEST( TaylorIntegrate, HarmonicOscillator )
{
    constexpr int N = 12;

    auto rhs = []( auto /*t*/, auto y ) -> decltype( y )
    {
        decltype( y ) out( 2 );
        out( 0 ) = y( 1 );
        out( 1 ) = -y( 0 );
        return out;
    };

    Eigen::Vector2d y0{ 1.0, 0.0 };

    TaylorIntegratorOptions options;
    options.atol = 1e-10;
    options.rtol = 1e-10;

    auto integrator = makeTaylorIntegrator< N >( rhs, options );
    auto result     = integrator.integrate( 0.0, 2.0, y0, 0.5 );

    ASSERT_FALSE( result.t.empty() );
    const double tf = result.t.back();
    EXPECT_NEAR( tf, 2.0, 1e-12 );
    EXPECT_NEAR( result.y.back()( 0 ), std::cos( tf ), 1e-8 );
    EXPECT_NEAR( result.y.back()( 1 ), -std::sin( tf ), 1e-8 );
}

TEST( TaylorIntegrate, SolutionAccuracy )
{
    constexpr int N = 10;

    auto rhs = []( auto /*t*/, auto y ) -> decltype( y )
    {
        decltype( y ) out( 1 );
        out( 0 ) = y( 0 );
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 1.0;

    TaylorIntegratorOptions options;
    options.atol = 1e-10;
    options.rtol = 1e-10;

    auto integrator = makeTaylorIntegrator< N >( rhs, options );
    auto result     = integrator.integrate( 0.0, 1.0, y0, 0.2 );

    for ( std::size_t k = 0; k < result.t.size(); ++k )
    {
        const double t = result.t[k];
        EXPECT_NEAR( result.y[k]( 0 ), std::exp( t ), 1e-8 )
            << "at t=" << t << " step=" << k;
    }
}

TEST( TaylorIntegrate, InitialConditionIncluded )
{
    constexpr int N = 5;

    auto rhs = []( auto /*t*/, auto y ) -> decltype( y )
    {
        decltype( y ) out( 1 );
        out( 0 ) = y( 0 );
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 3.14;

    auto integrator = makeTaylorIntegrator< N >( rhs );
    auto result     = integrator.integrate( 0.5, 1.0, y0, 0.1 );

    ASSERT_GE( result.t.size(), 1u );
    EXPECT_NEAR( result.t.front(), 0.5, 1e-14 );
    EXPECT_NEAR( result.y.front()( 0 ), 3.14, 1e-14 );
}

TEST( TaylorIntegrate, CustomStepControllerComposition )
{
    constexpr int N = 8;

    auto rhs = []( auto /*t*/, auto y ) -> decltype( y )
    {
        decltype( y ) out( 1 );
        out( 0 ) = y( 0 );
        return out;
    };

    int calls = 0;
    auto controller = [&calls]( double h, double /*tf*/, const auto&, const auto& ) -> double
    {
        ++calls;
        return h;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 1.0;

    auto integrator = makeTaylorIntegrator< N >( rhs, controller );
    auto result     = integrator.integrate( 0.0, 0.25, y0, 0.05 );

    ASSERT_FALSE( result.t.empty() );
    EXPECT_EQ( calls, int( result.t.size() - 1 ) );
    EXPECT_NEAR( result.t.back(), 0.25, 1e-12 );
}

TEST( TaylorIntegrate, FiniteUsesOptionsMaxSteps )
{
    constexpr int N = 8;

    auto rhs = []( auto /*t*/, auto y ) -> decltype( y )
    {
        decltype( y ) out( 1 );
        out( 0 ) = y( 0 );
        return out;
    };

    Eigen::Matrix< double, 1, 1 > y0;
    y0( 0 ) = 1.0;

    TaylorIntegratorOptions options;
    options.maxSteps = 0;

    auto integrator = makeTaylorIntegrator< N >( rhs, options );
    auto result     = integrator.integrate( 0.0, 1.0, y0, 0.1 );

    ASSERT_EQ( result.t.size(), std::size_t( 1 ) );
    EXPECT_NEAR( result.t.front(), 0.0, 1e-14 );
    EXPECT_NEAR( result.y.front()( 0 ), 1.0, 1e-14 );
}

