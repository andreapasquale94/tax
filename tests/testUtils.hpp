#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <tax/tax.hpp>

using namespace tax;

static constexpr double kTol = 1e-10;

/// @brief Check that every coefficient of two DA values is within `tol`.
template < typename DA_ >
static void ExpectCoeffsNear( const DA_& a, const DA_& b, double tol = kTol )
{
    for ( std::size_t k = 0; k < DA_::nCoefficients; ++k )
        EXPECT_NEAR( double( a[k] ), double( b[k] ), tol ) << "  coeff k=" << k;
}

/// @brief Check that every coefficient of a DA value matches `expected`.
template < typename DA_, std::size_t S >
static void ExpectCoeffsNear(
    const DA_& a, const std::array< typename DA_::Data::value_type, S >& expected,
    double tol = kTol )
{
    static_assert( S == DA_::nCoefficients );
    for ( std::size_t k = 0; k < DA_::nCoefficients; ++k )
        EXPECT_NEAR( double( a[k] ), double( expected[k] ), tol ) << "  coeff k=" << k;
}
