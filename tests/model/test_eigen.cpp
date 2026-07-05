// Eigen compatibility layer for Taylor models: TaylorModel as an Eigen scalar.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

#include "modelTestUtils.hpp"

using I = tax::Interval< double >;

// ---------------------------------------------------------------------------
// Domain-agnostic abstract constants (the enabling mechanism)
// ---------------------------------------------------------------------------

TEST( ModelEigen, AbstractConstantAdoptsPartnerDomain )
{
    const auto x = tax::TM< 3 >::variable( 2.0, I{ 1.9, 2.1 } );
    const tax::TM< 3 > zero{ 0.0 };
    const tax::TM< 3 > one{ 1.0 };
    EXPECT_TRUE( zero.isAbstractConstant() );

    // A constant adopts its partner's domain; the result is concrete and equal
    // to the partner (for 0) / shifted (for 1).
    const auto a = zero + x;
    EXPECT_FALSE( a.isAbstractConstant() );
    EXPECT_EQ( a.domain()[0], ( I{ 1.9, 2.1 } ) );
    tax::test::ExpectEncloses( a, []( const auto& p ) { return p[0]; } );

    const auto b = one * x;  // 1 * x
    tax::test::ExpectEncloses( b, []( const auto& p ) { return p[0]; } );

    // Constant-with-constant stays abstract.
    const auto c = one + tax::TM< 3 >{ 2.0 };
    EXPECT_TRUE( c.isAbstractConstant() );
    EXPECT_DOUBLE_EQ( c.value(), 3.0 );
}

TEST( ModelEigen, IncompatibleConcreteStillThrows )
{
    const auto x = tax::TM< 2 >::variable( 0.0, I{ -1.0, 1.0 } );
    const auto y = tax::TM< 2 >::variable( 0.0, I{ -2.0, 2.0 } );
    EXPECT_THROW( (void)( x + y ), std::invalid_argument );
    EXPECT_THROW( (void)( x * y ), std::invalid_argument );
}

// ---------------------------------------------------------------------------
// Eigen containers and products
// ---------------------------------------------------------------------------

TEST( ModelEigen, MatrixVectorProductWithLiterals )
{
    using TM = tax::TM< 5, 2 >;
    const TM::Point x0{ 1.0, 0.0 };
    const TM::Domain dom{ I{ 0.9, 1.1 }, I{ -0.1, 0.1 } };
    const auto y = tax::model::variables< double, 5, 2 >( x0, dom );

    const double th = 0.3;
    Eigen::Matrix< TM, 2, 2 > A;
    A << TM{ std::cos( th ) }, TM{ std::sin( th ) }, TM{ -std::sin( th ) }, TM{ std::cos( th ) };

    // Rotation-matrix (abstract literals) times the coordinate-variable vector
    // (concrete) — the variational-ODE STM * state pattern.
    const Eigen::Vector< TM, 2 > w = A * y;
    EXPECT_NEAR( w( 0 ).value(), std::cos( th ), 1e-14 );
    EXPECT_NEAR( w( 1 ).value(), -std::sin( th ), 1e-14 );

    // w encloses R(theta) applied to the true initial condition across the box.
    tax::test::ExpectEncloses(
        w( 0 ), [&]( const auto& p ) { return std::cos( th ) * p[0] + std::sin( th ) * p[1]; }, 7 );
    tax::test::ExpectEncloses(
        w( 1 ), [&]( const auto& p ) { return -std::sin( th ) * p[0] + std::cos( th ) * p[1]; },
        7 );
}

TEST( ModelEigen, IdentityZeroAndMatrixProduct )
{
    using TM = tax::TM< 4, 2 >;
    const TM::Point x0{ 0.5, -0.5 };
    const TM::Domain dom{ I{ 0.0, 1.0 }, I{ -1.0, 0.0 } };
    const auto y = tax::model::variables< double, 4, 2 >( x0, dom );

    // Identity() and the implied Scalar(0)/Scalar(1) literals compose with the
    // real-domain variables: (I - I) y = 0 exactly.
    const Eigen::Matrix< TM, 2, 2 > Id = Eigen::Matrix< TM, 2, 2 >::Identity();
    const Eigen::Vector< TM, 2 > z = Id * y - y;
    EXPECT_LT( z( 0 ).bound().width(), 1e-14 );
    EXPECT_LT( z( 1 ).bound().width(), 1e-14 );
    EXPECT_NEAR( z( 0 ).value(), 0.0, 1e-15 );

    // Matrix-matrix product composes two rotations.
    const double a = 0.2, b = 0.5;
    Eigen::Matrix< TM, 2, 2 > Ra, Rb;
    Ra << TM{ std::cos( a ) }, TM{ -std::sin( a ) }, TM{ std::sin( a ) }, TM{ std::cos( a ) };
    Rb << TM{ std::cos( b ) }, TM{ -std::sin( b ) }, TM{ std::sin( b ) }, TM{ std::cos( b ) };
    const Eigen::Matrix< TM, 2, 2 > Rab = Ra * Rb;  // R(a+b)
    EXPECT_NEAR( Rab( 0, 0 ).value(), std::cos( a + b ), 1e-14 );
    EXPECT_NEAR( Rab( 1, 0 ).value(), std::sin( a + b ), 1e-14 );
}

// ---------------------------------------------------------------------------
// Eigen-vector helpers
// ---------------------------------------------------------------------------

TEST( ModelEigen, VectorHelpers )
{
    using TM = tax::TM< 3, 2 >;
    const TM::Point x0{ 1.0, 2.0 };
    const TM::Domain dom{ I{ 0.5, 1.5 }, I{ 1.5, 2.5 } };
    const auto v = tax::model::variables< double, 3, 2 >( x0, dom );

    Eigen::Vector< TM, 2 > state;
    state << v( 0 ) * v( 1 ), v( 0 ) + v( 1 );

    const auto val = tax::model::value( state );
    EXPECT_DOUBLE_EQ( val( 0 ), 2.0 );
    EXPECT_DOUBLE_EQ( val( 1 ), 3.0 );

    const auto bnd = tax::model::bound( state );
    EXPECT_TRUE( bnd( 0 ).contains( 2.0 ) );

    // J for [x*y, x+y] at (1,2) = [[y,x],[1,1]] = [[2,1],[1,1]].
    const auto J = tax::model::jacobian( state );
    EXPECT_DOUBLE_EQ( J( 0, 0 ), 2.0 );
    EXPECT_DOUBLE_EQ( J( 0, 1 ), 1.0 );
    EXPECT_DOUBLE_EQ( J( 1, 0 ), 1.0 );
    EXPECT_DOUBLE_EQ( J( 1, 1 ), 1.0 );
}

// ---------------------------------------------------------------------------
// A one-step variational integrator expressed with Eigen
// ---------------------------------------------------------------------------

TEST( ModelEigen, VariationalHarmonicStep )
{
    // Harmonic oscillator flow over one Eigen matrix-vector application: with
    // the exact STM R(t) = [[cos t, sin t],[-sin t, cos t]] as an Eigen matrix
    // of literals, propagate the coordinate-variable state and confirm the
    // enclosure matches the analytic flow across the initial-condition box.
    using TM = tax::TM< 6, 2 >;
    const double t = 0.4;
    const TM::Point x0{ 1.0, 0.5 };
    const TM::Domain dom{ I{ 0.8, 1.2 }, I{ 0.3, 0.7 } };
    const auto r0 = tax::model::variables< double, 6, 2 >( x0, dom );

    Eigen::Matrix< TM, 2, 2 > R;
    R << TM{ std::cos( t ) }, TM{ std::sin( t ) }, TM{ -std::sin( t ) }, TM{ std::cos( t ) };
    const Eigen::Vector< TM, 2 > r = R * r0;

    tax::test::ExpectEncloses(
        r( 0 ), [&]( const auto& p ) { return std::cos( t ) * p[0] + std::sin( t ) * p[1]; }, 7 );
    tax::test::ExpectEncloses(
        r( 1 ), [&]( const auto& p ) { return -std::sin( t ) * p[0] + std::cos( t ) * p[1]; }, 7 );

    // The state-transition matrix read back from the propagated state is R.
    const auto J = tax::model::jacobian( r );
    EXPECT_NEAR( J( 0, 0 ), std::cos( t ), 1e-14 );
    EXPECT_NEAR( J( 0, 1 ), std::sin( t ), 1e-14 );
    EXPECT_NEAR( J( 1, 0 ), -std::sin( t ), 1e-14 );
}
