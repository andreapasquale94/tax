#include <tax/eigen/variables.hpp>

#include "../testUtils.hpp"

TEST( EigenVariables, TensorFromStaticVector )
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

TEST( EigenVariables, TensorFromStaticMatrix )
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
