#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/la.hpp>

TEST( EigenJacobian, OfLinearMap )
{
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< tax::TE< 3, 2 > >( x0 );
    tax::la::VecNT< 2, tax::TE< 3, 2 > > F;
    F( 0 ) = v( 0 ) + 2.0 * v( 1 );
    F( 1 ) = 3.0 * v( 0 ) - v( 1 );
    auto J = tax::la::jacobian( F );
    EXPECT_NEAR( J( 0, 0 ), 1.0, 1e-12 );
    EXPECT_NEAR( J( 0, 1 ), 2.0, 1e-12 );
    EXPECT_NEAR( J( 1, 0 ), 3.0, 1e-12 );
    EXPECT_NEAR( J( 1, 1 ), -1.0, 1e-12 );
}

TEST( EigenJacobian, OfNonlinearMap )
{
    Eigen::Vector2d x0{ 1.0, 1.0 };
    auto v = tax::la::variables< tax::TE< 3, 2 > >( x0 );
    tax::la::VecNT< 2, tax::TE< 3, 2 > > F;
    F( 0 ) = v( 0 ) * v( 1 );  // dF0/dx = y = 1, dF0/dy = x = 1
    F( 1 ) = v( 0 ) * v( 0 );  // dF1/dx = 2x = 2, dF1/dy = 0
    auto J = tax::la::jacobian( F );
    EXPECT_NEAR( J( 0, 0 ), 1.0, 1e-12 );
    EXPECT_NEAR( J( 0, 1 ), 1.0, 1e-12 );
    EXPECT_NEAR( J( 1, 0 ), 2.0, 1e-12 );
    EXPECT_NEAR( J( 1, 1 ), 0.0, 1e-12 );
}
