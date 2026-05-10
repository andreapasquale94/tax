// SPDX-License-Identifier: BSD-3-Clause
//
// Arithmetic operator tests: +, -, *, / and scalar variants.

#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "tax/tax.hpp"

using tax::TE;
using tax::TEn;

static constexpr double kTol = 1e-11;

// Helper: compare two TTEs coefficient-wise.
template < class Tte >
void expectAllNear( const Tte& a, const Tte& b, double tol = kTol )
{
    ASSERT_EQ( a.coeffs().size(), b.coeffs().size() );
    for ( Eigen::Index i = 0; i < a.coeffs().size(); ++i )
    {
        EXPECT_NEAR( a.coeffs()( i ), b.coeffs()( i ), tol ) << "coeff " << i;
    }
}

TEST( Arithmetic, AddTwoUnivariates )
{
    auto x = TE< 3 >::variable( 1.0 );
    auto y = TE< 3 >::variable( 2.0 );
    TE< 3 > result;
    result = (x + y).eval();
    EXPECT_NEAR( result.value(), 3.0, kTol );
    // Coeff at degree 1 = 2 (each contributes 1).
    EXPECT_NEAR( result.coeffs()( 1 ), 2.0, kTol );
}

TEST( Arithmetic, SubtractTwoUnivariates )
{
    auto x = TE< 3 >::variable( 5.0 );
    auto y = TE< 3 >::variable( 2.0 );
    TE< 3 > result;
    result = (x - y).eval();
    EXPECT_NEAR( result.value(), 3.0, kTol );
    EXPECT_NEAR( result.coeffs()( 1 ), 0.0, kTol );  // 1 - 1 = 0
}

TEST( Arithmetic, NegateUnivariate )
{
    auto x = TE< 3 >::variable( 1.5 );
    TE< 3 > result;
    result = (-x).eval();
    EXPECT_NEAR( result.value(), -1.5, kTol );
    EXPECT_NEAR( result.coeffs()( 1 ), -1.0, kTol );
}

TEST( Arithmetic, ScalarAdd )
{
    auto x = TE< 3 >::variable( 0.0 );
    TE< 3 > r1, r2;
    r1 = (x + 7.0).eval();
    r2 = (7.0 + x).eval();
    EXPECT_NEAR( r1.value(), 7.0, kTol );
    EXPECT_NEAR( r2.value(), 7.0, kTol );
    EXPECT_NEAR( r1.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r2.coeffs()( 1 ), 1.0, kTol );
}

TEST( Arithmetic, ScalarSubtract )
{
    auto x = TE< 3 >::variable( 4.0 );
    TE< 3 > r1, r2;
    r1 = (x - 1.0).eval();
    r2 = (10.0 - x).eval();
    EXPECT_NEAR( r1.value(), 3.0, kTol );
    EXPECT_NEAR( r2.value(), 6.0, kTol );
    EXPECT_NEAR( r1.coeffs()( 1 ), 1.0, kTol );
    EXPECT_NEAR( r2.coeffs()( 1 ), -1.0, kTol );
}

TEST( Arithmetic, ScalarMul )
{
    auto x = TE< 3 >::variable( 2.0 );
    TE< 3 > r;
    r = (3.0 * x).eval();
    EXPECT_NEAR( r.value(), 6.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 3.0, kTol );
}

TEST( Arithmetic, MultiplyUnivariates )
{
    // (x)(y) where x = 2 + dx, y = 3 + dx (univariate same variable)
    auto x = TE< 3 >::variable( 2.0 );
    auto y = TE< 3 >::variable( 3.0 );
    TE< 3 > r;
    r = (x * y).eval();
    // x*y as 1D Taylor in dx: (2+dx)(3+dx) = 6 + 5 dx + 1 dx^2 + 0 dx^3
    EXPECT_NEAR( r.value(), 6.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 5.0, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), 0.0, kTol );
}

TEST( Arithmetic, MultiplyMultivariate )
{
    auto [ x, y ] = TEn< 2, 2 >::variables( std::array< double, 2 >{ 1.0, 2.0 } );
    TEn< 2, 2 > r;
    r = (x * y).eval();
    // (1+dx)(2+dy) = 2 + 2 dx + 1 dy + dx*dy
    EXPECT_NEAR( r.value(), 2.0, kTol );
    std::array< std::size_t, 2 > a10{ 1, 0 };
    std::array< std::size_t, 2 > a01{ 0, 1 };
    std::array< std::size_t, 2 > a11{ 1, 1 };
    std::array< std::size_t, 2 > a20{ 2, 0 };
    std::array< std::size_t, 2 > a02{ 0, 2 };
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a10 ) ), 2.0, kTol );
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a01 ) ), 1.0, kTol );
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a11 ) ), 1.0, kTol );
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a20 ) ), 0.0, kTol );
    EXPECT_NEAR( r.coeff( std::span< const std::size_t >( a02 ) ), 0.0, kTol );
}

TEST( Arithmetic, DivideAndReciprocal )
{
    // 1 / (1 + x) at x=0 = 1 - x + x^2 - x^3 + ...
    auto x = TE< 4 >::variable( 0.0 );
    TE< 4 > r;
    r = (TE< 4 >::one() / ( TE< 4 >::one() + x )).eval();
    EXPECT_NEAR( r.coeffs()( 0 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), -1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), -1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 4 ), 1.0, kTol );
}

TEST( Arithmetic, NestedExpression )
{
    // ((x+1) * (x-1)) = x^2 - 1
    auto x = TE< 3 >::variable( 0.0 );
    TE< 3 > r;
    r = (( x + 1.0 ) * ( x - 1.0 )).eval();
    EXPECT_NEAR( r.coeffs()( 0 ), -1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 1 ), 0.0, kTol );
    EXPECT_NEAR( r.coeffs()( 2 ), 1.0, kTol );
    EXPECT_NEAR( r.coeffs()( 3 ), 0.0, kTol );
}
