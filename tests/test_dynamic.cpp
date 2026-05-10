// SPDX-License-Identifier: BSD-3-Clause
//
// DynamicTaylorExpansion (DynTE) tests.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "tax/tax.hpp"

using DynTE = tax::DynTE< double >;

static constexpr double kTol = 1e-11;

TEST( Dynamic, ZeroAndConstant )
{
    auto z = DynTE::zero( 3, 2 );
    EXPECT_EQ( z.order(), 3u );
    EXPECT_EQ( z.nvars(), 2u );
    EXPECT_EQ( z.value(), 0.0 );

    auto k = DynTE::constant( 2.5, 3, 2 );
    EXPECT_EQ( k.value(), 2.5 );
}

TEST( Dynamic, Variable )
{
    auto x = DynTE::variable( 1.5, 3, 1, 0 );
    EXPECT_EQ( x.value(), 1.5 );
    std::array< std::size_t, 1 > a1{ 1 };
    EXPECT_EQ( x.coeff( std::span< const std::size_t >( a1 ) ), 1.0 );
}

TEST( Dynamic, AddTwoUnivariates )
{
    auto x = DynTE::variable( 1.0, 3, 1, 0 );
    auto y = DynTE::variable( 2.0, 3, 1, 0 );
    DynTE r( 3, 1 );
    r = (x + y).eval();
    EXPECT_NEAR( r.value(), 3.0, kTol );
    std::array< std::size_t, 1 > a1{ 1 };
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a1 ) ), 2.0, kTol );
}

TEST( Dynamic, MultiplyMultivariate )
{
    auto vars = DynTE::variables( std::vector< double >{ 1.0, 2.0 }, 2 );
    auto& x = vars[ 0 ];
    auto& y = vars[ 1 ];
    DynTE r( 2, 2 );
    r = (x * y).eval();
    EXPECT_NEAR( r.value(), 2.0, kTol );
    std::array< std::size_t, 2 > a10{ 1, 0 };
    std::array< std::size_t, 2 > a01{ 0, 1 };
    std::array< std::size_t, 2 > a11{ 1, 1 };
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a10 ) ), 2.0, kTol );
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a01 ) ), 1.0, kTol );
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a11 ) ), 1.0, kTol );
}

TEST( Dynamic, ExpAtZero )
{
    auto x = DynTE::variable( 0.0, 4, 1, 0 );
    DynTE r( 4, 1 );
    r = (tax::exp( x )).eval();
    EXPECT_NEAR( r.coeffs()( 0 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), 0.5, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), 1.0 / 6.0, kTol );
    EXPECT_NEAR( r.coeffs()( 4 ), 1.0 / 24.0, kTol );
}

TEST( Dynamic, SinAtZero )
{
    auto x = DynTE::variable( 0.0, 5, 1, 0 );
    DynTE r( 5, 1 );
    r = (tax::sin( x )).eval();
    EXPECT_NEAR( r.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), -1.0 / 6.0, kTol );
    EXPECT_NEAR( r.coeffs()( 5 ), 1.0 / 120.0, kTol );
}

TEST( Dynamic, MixedStaticDynamicRejected )
{
    // This intentionally does NOT compile if uncommented:
    //   auto sx = tax::TE<3>::variable(1.0);
    //   auto dx = DynTE::variable(1.0, 3, 1, 0);
    //   auto bad = sx + dx;
    SUCCEED();
}
