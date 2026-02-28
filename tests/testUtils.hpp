#pragma once

#include <gtest/gtest.h>
#include <tax/tax.hpp>
#include <cmath>
#include <algorithm>

using namespace tax;

static constexpr double kTol  = 1e-12;
static constexpr double kTolF = 1e-5f;   // looser tolerance for float

/// Check that every coefficient of two DA values is within tol of each other.
template <typename DA_>
static void ExpectCoeffsNear(const DA_& a, const DA_& b,
                             double tol = kTol) {
    for (std::size_t k = 0; k < DA_::ncoef; ++k)
        EXPECT_NEAR(double(a[k]), double(b[k]), tol) << "  coeff k=" << k;
}

/// Check that every coefficient of a DA value matches the given array.
template <typename DA_, std::size_t S>
static void ExpectCoeffsNear(const DA_& a,
                             const std::array<typename DA_::coeff_array::value_type, S>& expected,
                             double tol = kTol) {
    static_assert(S == DA_::ncoef);
    for (std::size_t k = 0; k < DA_::ncoef; ++k)
        EXPECT_NEAR(double(a[k]), double(expected[k]), tol) << "  coeff k=" << k;
}
