#include <tax/eigen/value.hpp>
#include <tax/eigen/variables.hpp>

#include "../testUtils.hpp"

TEST( EigenValue, DenseRank1 )
{
    auto t = TE< 3 >::variable( 2.0 );
    Eigen::Matrix< TE< 3 >, 3, 1 > vec;
    vec( 0 ) = sin( t );
    vec( 1 ) = cos( t );
    vec( 2 ) = t * t;

    auto vals = tax::value( vec );
    EXPECT_NEAR( vals( 0 ), std::sin( 2.0 ), kTol );
    EXPECT_NEAR( vals( 1 ), std::cos( 2.0 ), kTol );
    EXPECT_NEAR( vals( 2 ), 4.0, kTol );
}

TEST( EigenValue, DenseRank2 )
{
    auto t = TE< 3 >::variable( 1.0 );
    Eigen::Matrix< TE< 3 >, 2, 2 > mat;
    mat( 0, 0 ) = exp( t );
    mat( 0, 1 ) = sin( t );
    mat( 1, 0 ) = cos( t );
    mat( 1, 1 ) = t;

    auto vals = tax::value( mat );
    EXPECT_NEAR( vals( 0, 0 ), std::exp( 1.0 ), kTol );
    EXPECT_NEAR( vals( 0, 1 ), std::sin( 1.0 ), kTol );
    EXPECT_NEAR( vals( 1, 0 ), std::cos( 1.0 ), kTol );
    EXPECT_NEAR( vals( 1, 1 ), 1.0, kTol );
}

TEST( EigenValue, DenseMultivariate )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Matrix< TEn< 3, 2 >, 2, 1 > vec;
    vec( 0 ) = x * y;
    vec( 1 ) = sin( x ) + cos( y );

    auto vals = tax::value( vec );
    EXPECT_NEAR( vals( 0 ), 2.0, kTol );
    EXPECT_NEAR( vals( 1 ), std::sin( 1.0 ) + std::cos( 2.0 ), kTol );
}

TEST( EigenValue, TensorRank3Univariate )
{
    auto t = TE< 3 >::variable( 2.0 );
    Eigen::Tensor< TE< 3 >, 3 > ten( 2, 1, 1 );
    ten( 0, 0, 0 ) = sin( t );
    ten( 1, 0, 0 ) = t * t;

    auto vals = tax::value( ten );
    EXPECT_NEAR( vals( 0, 0, 0 ), std::sin( 2.0 ), kTol );
    EXPECT_NEAR( vals( 1, 0, 0 ), 4.0, kTol );
}

TEST( EigenValue, ScalarTTE )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    TEn< 3, 2 > f = x + y;
    EXPECT_NEAR( tax::value( f ), 3.0, kTol );
}
