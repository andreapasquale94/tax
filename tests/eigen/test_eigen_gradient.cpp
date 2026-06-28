#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/la.hpp>

TEST( EigenGradient, OfQuadratic )
{
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< tax::TE< 3, 2 > >( x0 );
    auto f = v( 0 ) * v( 0 ) + 2.0 * v( 0 ) * v( 1 );
    // df/dx = 2x + 2y; df/dy = 2x  (at x=1, y=2)
    auto g = tax::la::gradient( f );
    EXPECT_NEAR( g( 0 ), 2.0 * 1.0 + 2.0 * 2.0, 1e-12 );
    EXPECT_NEAR( g( 1 ), 2.0 * 1.0, 1e-12 );
}

TEST( EigenGradient, MethodMatchesFreeFunction )
{
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< tax::TE< 3, 2 > >( x0 );
    auto f  = v( 0 ) * v( 1 );
    auto g1 = tax::la::gradient( f );
    auto g2 = tax::la::gradient( f );
    EXPECT_NEAR( ( g1 - g2 ).norm(), 0.0, 1e-15 );
}

TEST( EigenGradient, Univariate )
{
    auto x = tax::TE< 3 >::variable( 2.0 );
    auto f  = x * x * x;  // f = x^3, f' = 3x^2 = 12 at x=2
    auto g  = tax::la::gradient( f );
    EXPECT_NEAR( g( 0 ), 12.0, 1e-12 );
}
