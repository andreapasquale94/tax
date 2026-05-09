// SPDX-License-Identifier: BSD-3-Clause
//
// Static storage tests: factories, accessors, eval, derivative.

#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "tax/tax.hpp"

using tax::TE;
using tax::TEn;

constexpr double kTol = 1e-12;

TEST( StaticStorage, ZeroAndConstantAndOne )
{
    auto z = TE< 3 >::zero();
    EXPECT_EQ( z.value(), 0.0 );

    auto k = TE< 3 >::constant( 4.5 );
    EXPECT_EQ( k.value(), 4.5 );

    auto u = TE< 3 >::one();
    EXPECT_EQ( u.value(), 1.0 );
}

TEST( StaticStorage, Univariate )
{
    auto x = TE< 5 >::variable( 1.5 );
    EXPECT_EQ( x.value(), 1.5 );
    EXPECT_EQ( x.coeffs()( 0 ), 1.5 );
    EXPECT_EQ( x.coeffs()( 1 ), 1.0 );
    for ( int i = 2; i < 6; ++i )
    {
        EXPECT_EQ( x.coeffs()( i ), 0.0 );
    }
}

TEST( StaticStorage, MultivariateVariables )
{
    auto [ x, y ] = TEn< 3, 2 >::variables( std::array< double, 2 >{ 2.0, 5.0 } );
    EXPECT_EQ( x.value(), 2.0 );
    EXPECT_EQ( y.value(), 5.0 );
    // x has dx slot at flatIndex({1,0}) = 1
    std::array< std::size_t, 2 > a10{ 1, 0 };
    std::array< std::size_t, 2 > a01{ 0, 1 };
    EXPECT_EQ( x.coeff( std::span< const std::size_t >( a10 ) ), 1.0 );
    EXPECT_EQ( x.coeff( std::span< const std::size_t >( a01 ) ), 0.0 );
    EXPECT_EQ( y.coeff( std::span< const std::size_t >( a10 ) ), 0.0 );
    EXPECT_EQ( y.coeff( std::span< const std::size_t >( a01 ) ), 1.0 );
}

TEST( StaticStorage, EvalUnivariate )
{
    // f(x) = x with center 2.0; eval at displacement 0.5 -> 2.5
    auto x = TE< 4 >::variable( 2.0 );
    std::array< double, 1 > dx{ 0.5 };
    EXPECT_NEAR( x.eval( dx ), 2.5, kTol );
}

TEST( StaticStorage, EvalMultivariate )
{
    auto [ x, y ] = TEn< 3, 2 >::variables( std::array< double, 2 >{ 1.0, 2.0 } );
    std::array< double, 2 > dx{ 0.1, -0.2 };
    EXPECT_NEAR( x.eval( dx ), 1.1, kTol );
    EXPECT_NEAR( y.eval( dx ), 1.8, kTol );
}

TEST( StaticStorage, DerivativeApplyFactorialScaling )
{
    // For x = x0 + dx, the coefficient at degree 1 is 1, so derivative
    // (1!) * 1 = 1.
    auto x = TE< 3 >::variable( 0.0 );
    std::array< std::size_t, 1 > a1{ 1 };
    EXPECT_EQ( x.derivative( std::span< const std::size_t >( a1 ) ), 1.0 );
}

TEST( StaticStorage, ArrayOverloads )
{
    auto [ x, y ] = TEn< 3, 2 >::variables( std::array< double, 2 >{ 1.0, 2.0 } );
    // Direct braced-init goes through the std::array overload.
    EXPECT_EQ( x.coeff( { 1, 0 } ), 1.0 );
    EXPECT_EQ( x.coeff( { 0, 1 } ), 0.0 );
    EXPECT_EQ( y.coeff( { 0, 1 } ), 1.0 );
    EXPECT_EQ( x.derivative( { 1, 0 } ), 1.0 );
    EXPECT_NEAR( x.eval( { 0.1, -0.2 } ), 1.1, 1e-12 );
}

TEST( StaticStorage, NormsAreReasonable )
{
    auto x = TE< 3 >::variable( 1.5 );
    EXPECT_DOUBLE_EQ( x.coeffsNormInf(), 1.5 );
    EXPECT_DOUBLE_EQ( x.coeffsNorm< 1 >(), 2.5 );  // |1.5| + |1| + 0 + 0
    EXPECT_DOUBLE_EQ( x.coeffsNorm< 2 >(), std::sqrt( 1.5 * 1.5 + 1.0 ) );
}
