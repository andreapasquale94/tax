#include <tax/eigen/eval.hpp>
#include <tax/eigen/invert_map.hpp>
#include <tax/eigen/variables.hpp>

#include "../testUtils.hpp"

TEST( InvertMap, UnivariateSinMatchesAsin )
{
    constexpr int N = 9;

    auto x = TE< N >::variable( 0.0 );
    Eigen::Matrix< TE< N >, 1, 1 > map;
    map( 0 ) = sin( x );

    auto inv = tax::invert( map );
    TE< N > ref = asin( x );

    ExpectCoeffsNear( inv( 0 ), ref, 1e-14 );
}

TEST( InvertMap, BivariateLinearMap )
{
    constexpr int N = 4;

    auto [x, y] = TEn< N, 2 >::variables( { 0.0, 0.0 } );
    Eigen::Matrix< TEn< N, 2 >, 2, 1 > map;
    map( 0 ) = 2.0 * x + y;
    map( 1 ) = x + 3.0 * y;

    auto inv = tax::invert( map );

    TEn< N, 2 > ref0 = 0.6 * x - 0.2 * y;
    TEn< N, 2 > ref1 = -0.2 * x + 0.4 * y;
    ExpectCoeffsNear( inv( 0 ), ref0, 1e-14 );
    ExpectCoeffsNear( inv( 1 ), ref1, 1e-14 );
}

TEST( InvertMap, BivariateNonlinearRoundTrip )
{
    constexpr int N = 15;

    auto [x, y] = TEn< N, 2 >::variables( { 0.0, 0.0 } );
    Eigen::Matrix< TEn< N, 2 >, 2, 1 > map;
    map( 0 ) = x + y + x * y;
    map( 1 ) = y + x * x;

    auto inv = tax::invert( map );

    {
        Eigen::Vector2d dx;
        dx << 1e-2, -1e-2;
        const auto u = tax::eval( inv, dx );
        const auto back = tax::eval( map, u );
        EXPECT_NEAR( back( 0 ), dx( 0 ), 1e-13 );
        EXPECT_NEAR( back( 1 ), dx( 1 ), 1e-13 );
    }

    {
        Eigen::Vector2d dx;
        dx << -1e-2, 1e-2;
        const auto u = tax::eval( inv, dx );
        const auto back = tax::eval( map, u );
        EXPECT_NEAR( back( 0 ), dx( 0 ), 1e-13 );
        EXPECT_NEAR( back( 1 ), dx( 1 ), 1e-13 );
    }
}

TEST( InvertMap, RejectsSizeMismatch )
{
    constexpr int N = 3;
    using DA = TEn< N, 2 >;

    auto [x, y] = DA::variables( { 0.0, 0.0 } );
    Eigen::Matrix< DA, Eigen::Dynamic, 1 > bad( 1 );
    bad( 0 ) = x + y;

    EXPECT_THROW( (void)tax::invert( bad ), std::invalid_argument );
}

TEST( InvertMap, RejectsSingularLinearPart )
{
    constexpr int N = 3;
    auto [x, y] = TEn< N, 2 >::variables( { 0.0, 0.0 } );

    Eigen::Matrix< TEn< N, 2 >, 2, 1 > map;
    map( 0 ) = x + y;
    map( 1 ) = 2.0 * x + 2.0 * y;

    EXPECT_THROW( (void)tax::invert( map ), std::invalid_argument );
}
