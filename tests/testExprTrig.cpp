#include "testUtils.hpp"

// =============================================================================
// Sin — Taylor series of sin(f)
// =============================================================================

TEST(Sin, ConstantSin) {
    DAd<3> a{1.0};
    DAd<3> r = sin(a);
    EXPECT_NEAR(r.value(), std::sin(1.0), kTol);
    for (std::size_t k = 1; k < DAd<3>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(Sin, SinOfVariable) {
    // sin(x) at x0=0: coeffs are 0, 1, 0, -1/6, 0, 1/120, ...
    auto x = DAd<5>::variable<0>({0.0});
    DAd<5> r = sin(x);
    EXPECT_NEAR(r[0],  0.0,       kTol);
    EXPECT_NEAR(r[1],  1.0,       kTol);
    EXPECT_NEAR(r[2],  0.0,       kTol);
    EXPECT_NEAR(r[3], -1.0/6.0,   kTol);
    EXPECT_NEAR(r[4],  0.0,       kTol);
    EXPECT_NEAR(r[5],  1.0/120.0, kTol);
}

TEST(Sin, DerivativeCheck) {
    // d/dx sin(x) at x0=pi/4: value = sin(pi/4), deriv = cos(pi/4)
    constexpr double x0 = M_PI / 4.0;
    auto x = DAd<3>::variable<0>({x0});
    DAd<3> r = sin(x);
    EXPECT_NEAR(r.value(), std::sin(x0), kTol);
    EXPECT_NEAR(r.derivative({1}), std::cos(x0), kTol);
}

TEST(Sin, Bivariate) {
    auto [x, y] = DAMd<3,2>::variables({1.0, 2.0});
    DAMd<3,2> r = sin(x);
    // sin(x) should not depend on y
    EXPECT_NEAR(r.coeff({0,1}), 0.0, kTol);
    EXPECT_NEAR(r.coeff({0,0}), std::sin(1.0), kTol);
}

TEST(Sin, OfExpression) {
    DAd<4> a{M_PI / 6.0};
    DAd<4> r = sin(a + a);  // sin(pi/3)
    EXPECT_NEAR(r.value(), std::sin(M_PI / 3.0), kTol);
}

// =============================================================================
// Cos — Taylor series of cos(f)
// =============================================================================

TEST(Cos, ConstantCos) {
    DAd<3> a{1.0};
    DAd<3> r = cos(a);
    EXPECT_NEAR(r.value(), std::cos(1.0), kTol);
    for (std::size_t k = 1; k < DAd<3>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(Cos, CosOfVariable) {
    // cos(x) at x0=0: coeffs are 1, 0, -1/2, 0, 1/24, ...
    auto x = DAd<5>::variable<0>({0.0});
    DAd<5> r = cos(x);
    EXPECT_NEAR(r[0],  1.0,       kTol);
    EXPECT_NEAR(r[1],  0.0,       kTol);
    EXPECT_NEAR(r[2], -0.5,       kTol);
    EXPECT_NEAR(r[3],  0.0,       kTol);
    EXPECT_NEAR(r[4],  1.0/24.0,  kTol);
    EXPECT_NEAR(r[5],  0.0,       kTol);
}

TEST(Cos, DerivativeCheck) {
    // d/dx cos(x) at x0=pi/4: value = cos(pi/4), deriv = -sin(pi/4)
    constexpr double x0 = M_PI / 4.0;
    auto x = DAd<3>::variable<0>({x0});
    DAd<3> r = cos(x);
    EXPECT_NEAR(r.value(), std::cos(x0), kTol);
    EXPECT_NEAR(r.derivative({1}), -std::sin(x0), kTol);
}

TEST(Cos, Bivariate) {
    auto [x, y] = DAMd<3,2>::variables({1.0, 2.0});
    DAMd<3,2> r = cos(x);
    EXPECT_NEAR(r.coeff({0,1}), 0.0, kTol);
    EXPECT_NEAR(r.coeff({0,0}), std::cos(1.0), kTol);
}

TEST(Cos, OfExpression) {
    DAd<4> a{M_PI / 6.0};
    DAd<4> r = cos(a + a);  // cos(pi/3)
    EXPECT_NEAR(r.value(), std::cos(M_PI / 3.0), kTol);
}

// =============================================================================
// Pythagorean identity: sin^2 + cos^2 = 1
// =============================================================================

TEST(SinCosIdentity, PythagoreanUnivariate) {
    auto x = DAd<5>::variable<0>({1.5});
    DAd<5> s = sin(x);
    DAd<5> c = cos(x);
    DAd<5> r = square(s) + square(c);
    // Should be identically 1
    EXPECT_NEAR(r[0], 1.0, kTol);
    for (std::size_t k = 1; k < DAd<5>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(SinCosIdentity, PythagoreanBivariate) {
    auto [x, y] = DAMd<4,2>::variables({0.7, -1.3});
    DAMd<4,2> s = sin(x + y);
    DAMd<4,2> c = cos(x + y);
    DAMd<4,2> r = square(s) + square(c);
    EXPECT_NEAR(r[0], 1.0, kTol);
    for (std::size_t k = 1; k < DAMd<4,2>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

// =============================================================================
// SinCos — returns both sin and cos in one call
// =============================================================================

TEST(SinCos, MatchesSeparate) {
    auto x = DAd<5>::variable<0>({1.5});
    auto [s, c] = sincos(x);
    DAd<5> s2 = sin(x);
    DAd<5> c2 = cos(x);
    ExpectCoeffsNear(s, s2);
    ExpectCoeffsNear(c, c2);
}

TEST(SinCos, MatchesSeparateBivariate) {
    auto [x, y] = DAMd<4,2>::variables({0.7, -1.3});
    auto expr = x * y + x;
    auto [s, c] = sincos(expr);
    DAMd<4,2> s2 = sin(expr);
    DAMd<4,2> c2 = cos(expr);
    ExpectCoeffsNear(s, s2);
    ExpectCoeffsNear(c, c2);
}

TEST(SinCos, Pythagorean) {
    auto x = DAd<5>::variable<0>({2.3});
    auto [s, c] = sincos(x);
    DAd<5> r = square(s) + square(c);
    EXPECT_NEAR(r[0], 1.0, kTol);
    for (std::size_t k = 1; k < DAd<5>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

// =============================================================================
// Tan — Taylor series of tan(f)
// =============================================================================

TEST(Tan, ConstantTan) {
    DAd<3> a{0.5};
    DAd<3> r = tan(a);
    EXPECT_NEAR(r.value(), std::tan(0.5), kTol);
    for (std::size_t k = 1; k < DAd<3>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(Tan, TanOfVariable) {
    // tan(x) at x0=0: coeffs are 0, 1, 0, 1/3, 0, 2/15, ...
    auto x = DAd<5>::variable<0>({0.0});
    DAd<5> r = tan(x);
    EXPECT_NEAR(r[0], 0.0,       kTol);
    EXPECT_NEAR(r[1], 1.0,       kTol);
    EXPECT_NEAR(r[2], 0.0,       kTol);
    EXPECT_NEAR(r[3], 1.0/3.0,   kTol);
    EXPECT_NEAR(r[4], 0.0,       kTol);
    EXPECT_NEAR(r[5], 2.0/15.0,  kTol);
}

TEST(Tan, MatchesSinOverCos) {
    auto x = DAd<5>::variable<0>({0.8});
    DAd<5> r1 = tan(x);
    DAd<5> r2 = sin(x) / cos(x);
    ExpectCoeffsNear(r1, r2);
}

TEST(Tan, DerivativeCheck) {
    // d/dx tan(x) at x0 = 1/sec^2(x0) = 1 + tan^2(x0)
    constexpr double x0 = 0.7;
    auto x = DAd<3>::variable<0>({x0});
    DAd<3> r = tan(x);
    EXPECT_NEAR(r.value(), std::tan(x0), kTol);
    double expected_deriv = 1.0 + std::tan(x0) * std::tan(x0);
    EXPECT_NEAR(r.derivative({1}), expected_deriv, kTol);
}

TEST(Tan, Bivariate) {
    auto [x, y] = DAMd<3,2>::variables({0.5, 1.0});
    DAMd<3,2> r1 = tan(x + y);
    DAMd<3,2> r2 = sin(x + y) / cos(x + y);
    ExpectCoeffsNear(r1, r2);
}

TEST(Tan, OfExpression) {
    auto x = DAd<4>::variable<0>({0.3});
    DAd<4> r1 = tan(x * 2.0);
    DAd<4> r2 = sin(x * 2.0) / cos(x * 2.0);
    ExpectCoeffsNear(r1, r2);
}
