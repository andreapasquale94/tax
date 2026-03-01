#include <tax/eigen/tensor_function.hpp>

#include "../testUtils.hpp"

// =============================================================================
// value — rank 1 (vector)
// =============================================================================

TEST( TensorFunction, ValueRank1 )
{
    auto t = DA< 3 >::variable( 2.0 );
    Eigen::Tensor< DA< 3 >, 1 > vec( 3 );
    vec( 0 ) = sin( t );
    vec( 1 ) = cos( t );
    vec( 2 ) = t * t;

    auto vals = tax::value( vec );
    EXPECT_NEAR( vals( 0 ), std::sin( 2.0 ), kTol );
    EXPECT_NEAR( vals( 1 ), std::cos( 2.0 ), kTol );
    EXPECT_NEAR( vals( 2 ), 4.0, kTol );
}

// =============================================================================
// derivative (runtime int) — rank 1
// =============================================================================

TEST( TensorFunction, DerivativeRank1Order1 )
{
    auto t = DA< 3 >::variable( 2.0 );
    Eigen::Tensor< DA< 3 >, 1 > vec( 3 );
    vec( 0 ) = sin( t );
    vec( 1 ) = cos( t );
    vec( 2 ) = t * t;

    auto d1 = tax::derivative( vec, 1 );
    EXPECT_NEAR( d1( 0 ), std::cos( 2.0 ), kTol );
    EXPECT_NEAR( d1( 1 ), -std::sin( 2.0 ), kTol );
    EXPECT_NEAR( d1( 2 ), 4.0, kTol );  // d/dt(t^2) = 2t = 4
}

TEST( TensorFunction, DerivativeRank1Order2 )
{
    auto t = DA< 3 >::variable( 2.0 );
    Eigen::Tensor< DA< 3 >, 1 > vec( 3 );
    vec( 0 ) = sin( t );
    vec( 1 ) = cos( t );
    vec( 2 ) = t * t;

    auto d2 = tax::derivative( vec, 2 );
    EXPECT_NEAR( d2( 0 ), -std::sin( 2.0 ), kTol );
    EXPECT_NEAR( d2( 1 ), -std::cos( 2.0 ), kTol );
    EXPECT_NEAR( d2( 2 ), 2.0, kTol );
}

// =============================================================================
// derivative<...> (compile-time) — univariate
// =============================================================================

TEST( TensorFunction, DerivativeCompileTimeUnivariate )
{
    auto t = DA< 3 >::variable( 1.0 );
    Eigen::Tensor< DA< 3 >, 1 > vec( 2 );
    vec( 0 ) = exp( t );
    vec( 1 ) = sin( t );

    auto d1 = tax::derivative< 1 >( vec );
    EXPECT_NEAR( d1( 0 ), std::exp( 1.0 ), kTol );
    EXPECT_NEAR( d1( 1 ), std::cos( 1.0 ), kTol );

    auto d2 = tax::derivative< 2 >( vec );
    EXPECT_NEAR( d2( 0 ), std::exp( 1.0 ), kTol );
    EXPECT_NEAR( d2( 1 ), -std::sin( 1.0 ), kTol );
}

// =============================================================================
// value — rank 2 (matrix)
// =============================================================================

TEST( TensorFunction, ValueRank2 )
{
    auto t = DA< 3 >::variable( 1.0 );
    Eigen::Tensor< DA< 3 >, 2 > mat( 2, 2 );
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

// =============================================================================
// derivative — rank 2
// =============================================================================

TEST( TensorFunction, DerivativeRank2Order1 )
{
    auto t = DA< 3 >::variable( 1.0 );
    Eigen::Tensor< DA< 3 >, 2 > mat( 2, 2 );
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

// =============================================================================
// derivative(0) == value
// =============================================================================

TEST( TensorFunction, DerivativeOrder0IsValue )
{
    auto t = DA< 3 >::variable( 0.5 );
    Eigen::Tensor< DA< 3 >, 1 > vec( 2 );
    vec( 0 ) = sin( t );
    vec( 1 ) = exp( t );

    auto vals = tax::value( vec );
    auto d0 = tax::derivative( vec, 0 );
    EXPECT_NEAR( d0( 0 ), vals( 0 ), kTol );
    EXPECT_NEAR( d0( 1 ), vals( 1 ), kTol );
}

// =============================================================================
// Multivariate: value and derivative with multi-index
// =============================================================================

TEST( TensorFunction, MultivariateValue )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Tensor< DAn< 3, 2 >, 1 > vec( 2 );
    vec( 0 ) = x * y;
    vec( 1 ) = sin( x ) + cos( y );

    auto vals = tax::value( vec );
    EXPECT_NEAR( vals( 0 ), 2.0, kTol );
    EXPECT_NEAR( vals( 1 ), std::sin( 1.0 ) + std::cos( 2.0 ), kTol );
}

TEST( TensorFunction, MultivariateDerivativeRuntime )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Tensor< DAn< 3, 2 >, 1 > vec( 2 );
    vec( 0 ) = x * y;
    vec( 1 ) = sin( x ) + cos( y );

    // d/dx
    auto dx = tax::derivative( vec, tax::MultiIndex< 2 >{ 1, 0 } );
    EXPECT_NEAR( dx( 0 ), 2.0, kTol );
    EXPECT_NEAR( dx( 1 ), std::cos( 1.0 ), kTol );

    // d/dy
    auto dy = tax::derivative( vec, tax::MultiIndex< 2 >{ 0, 1 } );
    EXPECT_NEAR( dy( 0 ), 1.0, kTol );
    EXPECT_NEAR( dy( 1 ), -std::sin( 2.0 ), kTol );
}

// =============================================================================
// Multivariate: derivative<...> compile-time
// =============================================================================

TEST( TensorFunction, MultivariateDerivativeCompileTime )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Tensor< DAn< 3, 2 >, 1 > vec( 2 );
    vec( 0 ) = x * y;
    vec( 1 ) = sin( x ) + cos( y );

    // d/dx
    auto dx = tax::derivative< 1, 0 >( vec );
    EXPECT_NEAR( dx( 0 ), 2.0, kTol );
    EXPECT_NEAR( dx( 1 ), std::cos( 1.0 ), kTol );

    // d/dy
    auto dy = tax::derivative< 0, 1 >( vec );
    EXPECT_NEAR( dy( 0 ), 1.0, kTol );
    EXPECT_NEAR( dy( 1 ), -std::sin( 2.0 ), kTol );

    // d2/dxdy
    auto dxy = tax::derivative< 1, 1 >( vec );
    EXPECT_NEAR( dxy( 0 ), 1.0, kTol );  // d2/dxdy(xy) = 1
    EXPECT_NEAR( dxy( 1 ), 0.0, kTol );  // d2/dxdy(sin(x)+cos(y)) = 0
}

TEST( TensorFunction, TensorFromStaticVectorMatrix )
{
    Eigen::Matrix< double, 3, 1 > x0;
    x0 << 1.0, 2.0, 3.0;

    auto tx = DAn< 2, 3 >::tensor( x0 );
    static_assert( decltype( tx )::RowsAtCompileTime == 3 );
    static_assert( decltype( tx )::ColsAtCompileTime == 1 );

    EXPECT_EQ( tx.rows(), Eigen::Index( 3 ) );
    EXPECT_EQ( tx.cols(), Eigen::Index( 1 ) );
    EXPECT_NEAR( tx( 0, 0 ).value(), 1.0, kTol );
    EXPECT_NEAR( tx( 1, 0 ).value(), 2.0, kTol );
    EXPECT_NEAR( tx( 2, 0 ).value(), 3.0, kTol );
    EXPECT_NEAR( tx( 0, 0 ).derivative( tax::MultiIndex< 3 >{ 1, 0, 0 } ), 1.0, kTol );
    EXPECT_NEAR( tx( 1, 0 ).derivative( tax::MultiIndex< 3 >{ 0, 1, 0 } ), 1.0, kTol );
    EXPECT_NEAR( tx( 2, 0 ).derivative( tax::MultiIndex< 3 >{ 0, 0, 1 } ), 1.0, kTol );
}

TEST( TensorFunction, TensorFromStaticMatrix )
{
    Eigen::Matrix< double, 2, 2 > x0;
    x0 << 1.0, 2.0, 3.0, 4.0;

    auto tx = DAn< 2, 4 >::tensor( x0 );
    static_assert( decltype( tx )::RowsAtCompileTime == 2 );
    static_assert( decltype( tx )::ColsAtCompileTime == 2 );

    EXPECT_EQ( tx.rows(), Eigen::Index( 2 ) );
    EXPECT_EQ( tx.cols(), Eigen::Index( 2 ) );
    EXPECT_NEAR( tx( 0, 0 ).value(), 1.0, kTol );
    EXPECT_NEAR( tx( 0, 1 ).value(), 2.0, kTol );
    EXPECT_NEAR( tx( 1, 0 ).value(), 3.0, kTol );
    EXPECT_NEAR( tx( 1, 1 ).value(), 4.0, kTol );
    EXPECT_NEAR( tx( 0, 0 ).derivative( tax::MultiIndex< 4 >{ 1, 0, 0, 0 } ), 1.0, kTol );
    EXPECT_NEAR( tx( 0, 1 ).derivative( tax::MultiIndex< 4 >{ 0, 1, 0, 0 } ), 1.0, kTol );
    EXPECT_NEAR( tx( 1, 0 ).derivative( tax::MultiIndex< 4 >{ 0, 0, 1, 0 } ), 1.0, kTol );
    EXPECT_NEAR( tx( 1, 1 ).derivative( tax::MultiIndex< 4 >{ 0, 0, 0, 1 } ), 1.0, kTol );
}

