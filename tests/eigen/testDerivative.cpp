#include <tax/eigen/derivative.hpp>
#include <tax/eigen/value.hpp>
#include <tax/eigen/variables.hpp>

#include "../testUtils.hpp"
#include <type_traits>

TEST( EigenDerivative, DenseRank1RuntimeOrder1 )
{
    auto t = TE< 3 >::variable( 2.0 );
    Eigen::Matrix< TE< 3 >, 3, 1 > vec;
    vec( 0 ) = sin( t );
    vec( 1 ) = cos( t );
    vec( 2 ) = t * t;

    auto d1 = tax::derivative( vec, 1 );
    EXPECT_NEAR( d1( 0 ), std::cos( 2.0 ), kTol );
    EXPECT_NEAR( d1( 1 ), -std::sin( 2.0 ), kTol );
    EXPECT_NEAR( d1( 2 ), 4.0, kTol );
}

TEST( EigenDerivative, DenseRank1RuntimeOrder2 )
{
    auto t = TE< 3 >::variable( 2.0 );
    Eigen::Matrix< TE< 3 >, 3, 1 > vec;
    vec( 0 ) = sin( t );
    vec( 1 ) = cos( t );
    vec( 2 ) = t * t;

    auto d2 = tax::derivative( vec, 2 );
    EXPECT_NEAR( d2( 0 ), -std::sin( 2.0 ), kTol );
    EXPECT_NEAR( d2( 1 ), -std::cos( 2.0 ), kTol );
    EXPECT_NEAR( d2( 2 ), 2.0, kTol );
}

TEST( EigenDerivative, DenseCompileTimeUnivariate )
{
    auto t = TE< 3 >::variable( 1.0 );
    Eigen::Matrix< TE< 3 >, 2, 1 > vec;
    vec( 0 ) = exp( t );
    vec( 1 ) = sin( t );

    auto d1 = tax::derivative< 1 >( vec );
    EXPECT_NEAR( d1( 0 ), std::exp( 1.0 ), kTol );
    EXPECT_NEAR( d1( 1 ), std::cos( 1.0 ), kTol );

    auto d2 = tax::derivative< 2 >( vec );
    EXPECT_NEAR( d2( 0 ), std::exp( 1.0 ), kTol );
    EXPECT_NEAR( d2( 1 ), -std::sin( 1.0 ), kTol );
}

TEST( EigenDerivative, DenseRank2RuntimeOrder1 )
{
    auto t = TE< 3 >::variable( 1.0 );
    Eigen::Matrix< TE< 3 >, 2, 2 > mat;
    mat( 0, 0 ) = exp( t );
    mat( 0, 1 ) = sin( t );
    mat( 1, 0 ) = cos( t );
    mat( 1, 1 ) = t;

    auto d1 = tax::derivative( mat, 1 );
    EXPECT_NEAR( d1( 0, 0 ), std::exp( 1.0 ), kTol );
    EXPECT_NEAR( d1( 0, 1 ), std::cos( 1.0 ), kTol );
    EXPECT_NEAR( d1( 1, 0 ), -std::sin( 1.0 ), kTol );
    EXPECT_NEAR( d1( 1, 1 ), 1.0, kTol );
}

TEST( EigenDerivative, DenseOrder0MatchesValue )
{
    auto t = TE< 3 >::variable( 0.5 );
    Eigen::Matrix< TE< 3 >, 2, 1 > vec;
    vec( 0 ) = sin( t );
    vec( 1 ) = exp( t );

    auto vals = tax::value( vec );
    auto d0 = tax::derivative( vec, 0 );
    EXPECT_NEAR( d0( 0 ), vals( 0 ), kTol );
    EXPECT_NEAR( d0( 1 ), vals( 1 ), kTol );
}

TEST( EigenDerivative, DenseMultivariateRuntimeMultiIndex )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Matrix< TEn< 3, 2 >, 2, 1 > vec;
    vec( 0 ) = x * y;
    vec( 1 ) = sin( x ) + cos( y );

    auto dx = tax::derivative( vec, tax::MultiIndex< 2 >{ 1, 0 } );
    EXPECT_NEAR( dx( 0 ), 2.0, kTol );
    EXPECT_NEAR( dx( 1 ), std::cos( 1.0 ), kTol );

    auto dy = tax::derivative( vec, tax::MultiIndex< 2 >{ 0, 1 } );
    EXPECT_NEAR( dy( 0 ), 1.0, kTol );
    EXPECT_NEAR( dy( 1 ), -std::sin( 2.0 ), kTol );
}

TEST( EigenDerivative, DenseMultivariateCompileTimeMultiIndex )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Matrix< TEn< 3, 2 >, 2, 1 > vec;
    vec( 0 ) = x * y;
    vec( 1 ) = sin( x ) + cos( y );

    auto dx = tax::derivative< 1, 0 >( vec );
    EXPECT_NEAR( dx( 0 ), 2.0, kTol );
    EXPECT_NEAR( dx( 1 ), std::cos( 1.0 ), kTol );

    auto dy = tax::derivative< 0, 1 >( vec );
    EXPECT_NEAR( dy( 0 ), 1.0, kTol );
    EXPECT_NEAR( dy( 1 ), -std::sin( 2.0 ), kTol );

    auto dxy = tax::derivative< 1, 1 >( vec );
    EXPECT_NEAR( dxy( 0 ), 1.0, kTol );
    EXPECT_NEAR( dxy( 1 ), 0.0, kTol );
}

TEST( EigenDerivative, TensorRuntimeUnivariate )
{
    auto t = TE< 3 >::variable( 2.0 );
    Eigen::Tensor< TE< 3 >, 3 > ten( 2, 1, 1 );
    ten( 0, 0, 0 ) = sin( t );
    ten( 1, 0, 0 ) = t * t;

    auto d1 = tax::derivative( ten, 1 );
    EXPECT_NEAR( d1( 0, 0, 0 ), std::cos( 2.0 ), kTol );
    EXPECT_NEAR( d1( 1, 0, 0 ), 4.0, kTol );
}

TEST( EigenDerivative, GradientUnivariate )
{
    auto t = TE< 3 >::variable( 2.0 );
    TE< 3 > f = sin( t );

    auto g = tax::gradient( f );
    static_assert( decltype( g )::RowsAtCompileTime == 1 );
    EXPECT_NEAR( g( 0 ), std::cos( 2.0 ), kTol );
}

TEST( EigenDerivative, GradientMultivariate )
{
    auto [x, y] = TEn< 2, 2 >::variables( { 1.0, 2.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;

    auto g = tax::gradient( f );
    static_assert( decltype( g )::RowsAtCompileTime == 2 );
    EXPECT_NEAR( g( 0 ), 4.0, kTol );
    EXPECT_NEAR( g( 1 ), 5.0, kTol );
}

TEST( EigenDerivative, JacobianVector )
{
    auto [x, y] = TEn< 2, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Matrix< TEn< 2, 2 >, 3, 1 > vec;
    vec( 0 ) = x * y;
    vec( 1 ) = sin( x );
    vec( 2 ) = x + y * y;

    auto J = tax::jacobian( vec );
    EXPECT_EQ( J.rows(), 3 );
    EXPECT_EQ( J.cols(), 2 );
    EXPECT_NEAR( J( 0, 0 ), 2.0, kTol );
    EXPECT_NEAR( J( 0, 1 ), 1.0, kTol );
    EXPECT_NEAR( J( 1, 0 ), std::cos( 1.0 ), kTol );
    EXPECT_NEAR( J( 1, 1 ), 0.0, kTol );
    EXPECT_NEAR( J( 2, 0 ), 1.0, kTol );
    EXPECT_NEAR( J( 2, 1 ), 4.0, kTol );
}

TEST( EigenDerivative, ScalarOrder1And2Objects )
{
    auto [x, y] = TEn< 2, 2 >::variables( { 1.0, 2.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;

    const auto g = tax::derivative< 1 >( f );
    static_assert( std::is_same_v< std::remove_cvref_t< decltype( g ) >, Eigen::Vector2d > );
    EXPECT_NEAR( g( 0 ), 4.0, kTol );
    EXPECT_NEAR( g( 1 ), 5.0, kTol );

    const auto h = tax::derivative< 2 >( f );
    static_assert( std::is_same_v< std::remove_cvref_t< decltype( h ) >, Eigen::Matrix2d > );
    EXPECT_NEAR( h( 0, 0 ), 2.0, kTol );
    EXPECT_NEAR( h( 0, 1 ), 1.0, kTol );
    EXPECT_NEAR( h( 1, 0 ), 1.0, kTol );
    EXPECT_NEAR( h( 1, 1 ), 2.0, kTol );
}

TEST( EigenDerivative, ScalarOrder3UsesTensor )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 0.0, 0.0 } );
    TEn< 3, 2 > f = x * x * x + x * x * y + x * y * y + y * y * y;

    const auto d3 = tax::derivative< 3 >( f );
    static_assert( std::is_same_v< std::remove_cvref_t< decltype( d3 ) >,
                                   Eigen::Tensor< double, 3, Eigen::RowMajor > > );
    EXPECT_NEAR( d3( 0, 0, 0 ), 6.0, kTol );
    EXPECT_NEAR( d3( 0, 0, 1 ), 2.0, kTol );
    EXPECT_NEAR( d3( 0, 1, 1 ), 2.0, kTol );
    EXPECT_NEAR( d3( 1, 1, 1 ), 6.0, kTol );
}
