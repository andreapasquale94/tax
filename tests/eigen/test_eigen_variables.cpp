#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/la.hpp>

TEST( EigenVariables, FromEigenVector )
{
    Eigen::Vector3d x0{ 1.0, 2.0, 3.0 };
    auto v = tax::la::variables< tax::TE< 3, 3 > >( x0 );
    EXPECT_NEAR( v( 0 ).value(), 1.0, 1e-15 );
    EXPECT_NEAR( v( 1 ).value(), 2.0, 1e-15 );
    EXPECT_NEAR( v( 2 ).value(), 3.0, 1e-15 );
    // Check that the linear coefficient of each variable is 1 and cross-terms 0
    EXPECT_NEAR( ( v( 0 ).coeff< 1, 0, 0 >() ), 1.0, 1e-15 );
    EXPECT_NEAR( ( v( 1 ).coeff< 0, 1, 0 >() ), 1.0, 1e-15 );
    EXPECT_NEAR( ( v( 2 ).coeff< 0, 0, 1 >() ), 1.0, 1e-15 );
    // Cross-terms should be zero
    EXPECT_NEAR( ( v( 0 ).coeff< 0, 1, 0 >() ), 0.0, 1e-15 );
}

TEST( EigenVariables, Bivariate )
{
    Eigen::Vector2d x0{ 3.0, -1.0 };
    auto v = tax::la::variables< tax::TE< 2, 2 > >( x0 );
    EXPECT_NEAR( v( 0 ).value(), 3.0, 1e-15 );
    EXPECT_NEAR( v( 1 ).value(), -1.0, 1e-15 );
    EXPECT_NEAR( ( v( 0 ).coeff< 1, 0 >() ), 1.0, 1e-15 );
    EXPECT_NEAR( ( v( 1 ).coeff< 0, 1 >() ), 1.0, 1e-15 );
    EXPECT_NEAR( ( v( 0 ).coeff< 0, 1 >() ), 0.0, 1e-15 );
}
