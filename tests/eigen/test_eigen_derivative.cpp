#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/la.hpp>

TEST( EigenDerivative, ElementWiseCompileTime )
{
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< tax::TE< 3, 2 > >( x0 );
    tax::la::VecNT< 2, tax::TE< 3, 2 > > F;
    F( 0 ) = v( 0 ) * v( 1 );  // d/dx = y = 2; d/dy = x = 1
    F( 1 ) = v( 0 ) + v( 1 );  // d/dx = 1;     d/dy = 1
    auto df_dx = tax::la::derivative< 1, 0 >( F );
    EXPECT_NEAR( df_dx( 0 ), 2.0, 1e-12 );  // d(x*y)/dx = y = 2 at x0
    EXPECT_NEAR( df_dx( 1 ), 1.0, 1e-12 );
}

TEST( EigenDerivative, SecondOrderDerivative )
{
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< tax::TE< 4, 2 > >( x0 );
    // f = x^2 => d^2f/dx^2 = 2
    auto f_sq = v( 0 ) * v( 0 );
    tax::la::VecNT< 1, tax::TE< 4, 2 > > F;
    F( 0 ) = f_sq;
    auto d2f = tax::la::derivative< 2, 0 >( F );
    EXPECT_NEAR( d2f( 0 ), 2.0, 1e-12 );
}
