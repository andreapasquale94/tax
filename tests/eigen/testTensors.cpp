#include <tax/eigen/tensors.hpp>

#include "../testUtils.hpp"

TEST( EigenTensors, DerivativeTensorGradientAndHessian )
{
    auto [x, y] = TEn< 2, 2 >::variables( { 1.0, 2.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;

    const auto g = tax::derivative< 1 >( f );
    EXPECT_NEAR( g( 0 ), 4.0, kTol );
    EXPECT_NEAR( g( 1 ), 5.0, kTol );

    const auto h = tax::derivative< 2 >( f );
    EXPECT_NEAR( h( 0, 0 ), 2.0, kTol );
    EXPECT_NEAR( h( 0, 1 ), 1.0, kTol );
    EXPECT_NEAR( h( 1, 0 ), 1.0, kTol );
    EXPECT_NEAR( h( 1, 1 ), 2.0, kTol );
}

TEST( EigenTensors, CoeffTensorOrder2Normalized )
{
    auto [x, y] = TEn< 2, 2 >::variables( { 0.0, 0.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;

    const auto c2 = tax::coeff< 2 >( f );
    EXPECT_NEAR( c2( 0, 0 ), 1.0, kTol );  // coeff x^2
    EXPECT_NEAR( c2( 1, 1 ), 1.0, kTol );  // coeff y^2
    EXPECT_NEAR( c2( 0, 1 ), 0.5, kTol );  // normalized split of xy term
    EXPECT_NEAR( c2( 1, 0 ), 0.5, kTol );  // normalized split of xy term
}
