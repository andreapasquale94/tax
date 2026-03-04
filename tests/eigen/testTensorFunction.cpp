#include <tax/eigen/tensor_function.hpp>

#include "../testUtils.hpp"

// =============================================================================
// value — rank 1 (vector)
// =============================================================================

TEST( TensorFunction, ValueRank1 )
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

// =============================================================================
// derivative (runtime int) — rank 1
// =============================================================================

TEST( TensorFunction, DerivativeRank1Order1 )
{
    auto t = TE< 3 >::variable( 2.0 );
    Eigen::Matrix< TE< 3 >, 3, 1 > vec;
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

// =============================================================================
// derivative<...> (compile-time) — univariate
// =============================================================================

TEST( TensorFunction, DerivativeCompileTimeUnivariate )
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

// =============================================================================
// value — rank 2 (matrix)
// =============================================================================

TEST( TensorFunction, ValueRank2 )
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

// =============================================================================
// derivative — rank 2
// =============================================================================

TEST( TensorFunction, DerivativeRank2Order1 )
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

// =============================================================================
// derivative(0) == value
// =============================================================================

TEST( TensorFunction, DerivativeOrder0IsValue )
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

// =============================================================================
// Multivariate: value and derivative with multi-index
// =============================================================================

TEST( TensorFunction, MultivariateValue )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Matrix< TEn< 3, 2 >, 2, 1 > vec;
    vec( 0 ) = x * y;
    vec( 1 ) = sin( x ) + cos( y );

    auto vals = tax::value( vec );
    EXPECT_NEAR( vals( 0 ), 2.0, kTol );
    EXPECT_NEAR( vals( 1 ), std::sin( 1.0 ) + std::cos( 2.0 ), kTol );
}

TEST( TensorFunction, MultivariateDerivativeRuntime )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Matrix< TEn< 3, 2 >, 2, 1 > vec;
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
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Matrix< TEn< 3, 2 >, 2, 1 > vec;
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

    auto tx = tensor< TEn< 2, 3 > >( x0 );
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

    auto tx = tensor< TEn< 2, 4 > >( x0 );
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

TEST( TensorFunction, TensorRank3Univariate )
{
    auto t = TE< 3 >::variable( 2.0 );
    Eigen::Tensor< TE< 3 >, 3 > ten( 2, 1, 1 );
    ten( 0, 0, 0 ) = sin( t );
    ten( 1, 0, 0 ) = t * t;

    auto vals = tax::value( ten );
    EXPECT_NEAR( vals( 0, 0, 0 ), std::sin( 2.0 ), kTol );
    EXPECT_NEAR( vals( 1, 0, 0 ), 4.0, kTol );

    auto d1 = tax::derivative( ten, 1 );
    EXPECT_NEAR( d1( 0, 0, 0 ), std::cos( 2.0 ), kTol );
    EXPECT_NEAR( d1( 1, 0, 0 ), 4.0, kTol );
}

// =============================================================================
// gradient
// =============================================================================

TEST( TensorFunction, GradientUnivariate )
{
    auto t = TE< 3 >::variable( 2.0 );
    TE< 3 > f = sin( t );

    auto g = tax::gradient( f );
    static_assert( decltype( g )::RowsAtCompileTime == 1 );
    EXPECT_NEAR( g( 0 ), std::cos( 2.0 ), kTol );
}

TEST( TensorFunction, GradientMultivariate )
{
    auto [x, y] = TEn< 2, 2 >::variables( { 1.0, 2.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;

    auto g = tax::gradient( f );
    static_assert( decltype( g )::RowsAtCompileTime == 2 );
    // df/dx = 2x + y = 4, df/dy = x + 2y = 5
    EXPECT_NEAR( g( 0 ), 4.0, kTol );
    EXPECT_NEAR( g( 1 ), 5.0, kTol );
}

// =============================================================================
// jacobian
// =============================================================================

TEST( TensorFunction, JacobianVector )
{
    auto [x, y] = TEn< 2, 2 >::variables( { 1.0, 2.0 } );
    Eigen::Matrix< TEn< 2, 2 >, 3, 1 > vec;
    vec( 0 ) = x * y;
    vec( 1 ) = sin( x );
    vec( 2 ) = x + y * y;

    auto J = tax::jacobian( vec );
    EXPECT_EQ( J.rows(), 3 );
    EXPECT_EQ( J.cols(), 2 );

    // d(xy)/dx = y = 2, d(xy)/dy = x = 1
    EXPECT_NEAR( J( 0, 0 ), 2.0, kTol );
    EXPECT_NEAR( J( 0, 1 ), 1.0, kTol );
    // d(sin(x))/dx = cos(x), d(sin(x))/dy = 0
    EXPECT_NEAR( J( 1, 0 ), std::cos( 1.0 ), kTol );
    EXPECT_NEAR( J( 1, 1 ), 0.0, kTol );
    // d(x+y^2)/dx = 1, d(x+y^2)/dy = 2y = 4
    EXPECT_NEAR( J( 2, 0 ), 1.0, kTol );
    EXPECT_NEAR( J( 2, 1 ), 4.0, kTol );
}

// =============================================================================
// container eval
// =============================================================================

TEST( TensorFunction, EvalContainerUnivariate )
{
    auto t = TE< 3 >::variable( 0.0 );
    Eigen::Matrix< TE< 3 >, 2, 1 > vec;
    vec( 0 ) = sin( t );
    vec( 1 ) = exp( t );

    auto vals = tax::eval( vec, 0.5 );
    EXPECT_NEAR( vals( 0 ), std::sin( 0.5 ), 1e-2 );
    EXPECT_NEAR( vals( 1 ), std::exp( 0.5 ), 1e-2 );
}

TEST( TensorFunction, EvalContainerMultivariatePointType )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 0.0, 0.0 } );
    Eigen::Matrix< TEn< 3, 2 >, 2, 1 > vec;
    vec( 0 ) = x + y;
    vec( 1 ) = x * y;

    auto vals = tax::eval( vec, TEn< 3, 2 >::point_type{ 1.0, 2.0 } );
    EXPECT_NEAR( vals( 0 ), 3.0, kTol );
    EXPECT_NEAR( vals( 1 ), 2.0, kTol );
}

TEST( TensorFunction, EvalContainerEigenDisplacement )
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

TEST( TensorFunction, EvalSingleDAEigenDisplacement )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 0.0, 0.0 } );
    TEn< 3, 2 > f = x * x + y * y;

    Eigen::Vector2d dx;
    dx << 1.0, 2.0;
    double val = tax::eval( f, dx );
    EXPECT_NEAR( val, 5.0, kTol );
}
