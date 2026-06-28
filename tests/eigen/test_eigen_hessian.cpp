#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/la.hpp>

TEST( EigenHessian, OfQuadratic )
{
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< tax::TE< 3, 2 > >( x0 );
    auto f = v( 0 ) * v( 0 ) + 3.0 * v( 0 ) * v( 1 ) + v( 1 ) * v( 1 );
    auto H = tax::la::hessian( f );
    EXPECT_NEAR( H( 0, 0 ), 2.0, 1e-12 );
    EXPECT_NEAR( H( 0, 1 ), 3.0, 1e-12 );
    EXPECT_NEAR( H( 1, 0 ), 3.0, 1e-12 );
    EXPECT_NEAR( H( 1, 1 ), 2.0, 1e-12 );
}

TEST( EigenHessian, HessianOfSumOfSquaresMatchesAnalytic )
{
    // f = x² + y²  at  (x0, y0) = (1.0, 2.0)
    // H(0,0) = 2,  H(0,1) = H(1,0) = 0,  H(1,1) = 2
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< tax::TE< 3, 2 > >( x0 );
    auto f = v( 0 ) * v( 0 ) + v( 1 ) * v( 1 );
    auto H = tax::la::hessian( f );
    EXPECT_NEAR( H( 0, 0 ), 2.0, 1e-12 );
    EXPECT_NEAR( H( 0, 1 ), 0.0, 1e-12 );
    EXPECT_NEAR( H( 1, 0 ), 0.0, 1e-12 );
    EXPECT_NEAR( H( 1, 1 ), 2.0, 1e-12 );
}
