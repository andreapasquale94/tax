#include <tax/eigen/eval.hpp>
#include <tax/eigen/variables.hpp>

#include "../testUtils.hpp"

TEST( EigenEval, DenseContainerUnivariate )
{
    auto t = TE< 3 >::variable( 0.0 );
    Eigen::Matrix< TE< 3 >, 2, 1 > vec;
    vec( 0 ) = sin( t );
    vec( 1 ) = exp( t );

    auto vals = tax::eval( vec, 0.5 );
    EXPECT_NEAR( vals( 0 ), std::sin( 0.5 ), 1e-2 );
    EXPECT_NEAR( vals( 1 ), std::exp( 0.5 ), 1e-2 );
}

TEST( EigenEval, DenseContainerMultivariatePointType )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 0.0, 0.0 } );
    Eigen::Matrix< TEn< 3, 2 >, 2, 1 > vec;
    vec( 0 ) = x + y;
    vec( 1 ) = x * y;

    auto vals = tax::eval( vec, TEn< 3, 2 >::Input{ 1.0, 2.0 } );
    EXPECT_NEAR( vals( 0 ), 3.0, kTol );
    EXPECT_NEAR( vals( 1 ), 2.0, kTol );
}

TEST( EigenEval, DenseContainerEigenDisplacement )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 0.0, 0.0 } );
    Eigen::Matrix< TEn< 3, 2 >, 2, 1 > vec;
    vec( 0 ) = x + y;
    vec( 1 ) = x * y;

    Eigen::Vector2d dx;
    dx << 1.0, 2.0;
    auto vals = tax::eval( vec, dx );
    EXPECT_NEAR( vals( 0 ), 3.0, kTol );
    EXPECT_NEAR( vals( 1 ), 2.0, kTol );
}

TEST( EigenEval, ScalarTTEEigenDisplacement )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 0.0, 0.0 } );
    TEn< 3, 2 > f = x * x + y * y;

    Eigen::Vector2d dx;
    dx << 1.0, 2.0;
    double val = tax::eval( f, dx );
    EXPECT_NEAR( val, 5.0, kTol );
}

TEST( EigenEval, ScalarTTE )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    TEn< 3, 2 > f = x + y;
    EXPECT_NEAR( tax::eval( f, TEn< 3, 2 >::Input{ 0.5, 0.25 } ), 3.75, kTol );

    auto t = TE< 5 >::variable( 0.0 );
    TE< 5 > g = sin( t );
    EXPECT_NEAR( tax::eval( g, 0.5 ), std::sin( 0.5 ), 1e-4 );
}
