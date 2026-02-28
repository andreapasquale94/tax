#include "testUtils.hpp"

// =============================================================================
// Sinh
// =============================================================================

TEST(Sinh, Constant) {
    DA<3> a{1.5};
    DA<3> r = sinh(a);
    EXPECT_NEAR(r.value(), std::sinh(1.5), kTol);
    for (std::size_t k = 1; k < DA<3>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(Sinh, KnownSeries) {
    // sinh(x) at x0=0: coeffs are 0, 1, 0, 1/6, 0, 1/120
    auto x = DA<5>::variable(0.0);
    DA<5> r = sinh(x);
    EXPECT_NEAR(r[0], 0.0, kTol);
    EXPECT_NEAR(r[1], 1.0, kTol);
    EXPECT_NEAR(r[2], 0.0, kTol);
    EXPECT_NEAR(r[3], 1.0/6.0, kTol);
    EXPECT_NEAR(r[4], 0.0, kTol);
    EXPECT_NEAR(r[5], 1.0/120.0, kTol);
}

TEST(Sinh, DerivativeCheck) {
    constexpr double x0 = 1.0;
    auto x = DA<3>::variable(x0);
    DA<3> r = sinh(x);
    EXPECT_NEAR(r.value(), std::sinh(x0), kTol);
    EXPECT_NEAR(r.derivative({1}), std::cosh(x0), kTol);
}

TEST(Sinh, ViaExp) {
    // sinh(x) = (exp(x) - exp(-x)) / 2
    auto x = DA<5>::variable(1.0);
    DA<5> r1 = sinh(x);
    DA<5> r2 = (exp(x) - exp(-x)) * 0.5;
    ExpectCoeffsNear(r1, r2);
}

TEST(Sinh, Bivariate) {
    auto [x, y] = DAn<3,2>::variables({1.0, 2.0});
    DAn<3,2> r = sinh(x);
    EXPECT_NEAR(r.coeff({0,1}), 0.0, kTol);
    EXPECT_NEAR(r.coeff({0,0}), std::sinh(1.0), kTol);
}

// =============================================================================
// Cosh
// =============================================================================

TEST(Cosh, Constant) {
    DA<3> a{1.5};
    DA<3> r = cosh(a);
    EXPECT_NEAR(r.value(), std::cosh(1.5), kTol);
    for (std::size_t k = 1; k < DA<3>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(Cosh, KnownSeries) {
    // cosh(x) at x0=0: coeffs are 1, 0, 1/2, 0, 1/24, 0
    auto x = DA<5>::variable(0.0);
    DA<5> r = cosh(x);
    EXPECT_NEAR(r[0], 1.0, kTol);
    EXPECT_NEAR(r[1], 0.0, kTol);
    EXPECT_NEAR(r[2], 1.0/2.0, kTol);
    EXPECT_NEAR(r[3], 0.0, kTol);
    EXPECT_NEAR(r[4], 1.0/24.0, kTol);
    EXPECT_NEAR(r[5], 0.0, kTol);
}

TEST(Cosh, DerivativeCheck) {
    constexpr double x0 = 1.0;
    auto x = DA<3>::variable(x0);
    DA<3> r = cosh(x);
    EXPECT_NEAR(r.value(), std::cosh(x0), kTol);
    EXPECT_NEAR(r.derivative({1}), std::sinh(x0), kTol);
}

TEST(Cosh, ViaExp) {
    // cosh(x) = (exp(x) + exp(-x)) / 2
    auto x = DA<5>::variable(1.0);
    DA<5> r1 = cosh(x);
    DA<5> r2 = (exp(x) + exp(-x)) * 0.5;
    ExpectCoeffsNear(r1, r2);
}

// =============================================================================
// SinhCosh identity and dual output
// =============================================================================

TEST(SinhCoshIdentity, HyperbolicPythagorean) {
    // cosh^2(x) - sinh^2(x) = 1
    auto x = DA<5>::variable(1.5);
    DA<5> r = square(cosh(x)) - square(sinh(x));
    EXPECT_NEAR(r.value(), 1.0, kTol);
    for (std::size_t k = 1; k < DA<5>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(SinhCosh, MatchesSeparate) {
    auto x = DA<5>::variable(1.0);
    auto [sh, ch] = sinhcosh(x);
    DA<5> s = sinh(x);
    DA<5> c = cosh(x);
    ExpectCoeffsNear(sh, s);
    ExpectCoeffsNear(ch, c);
}

TEST(SinhCosh, Pythagorean) {
    auto x = DA<5>::variable(2.0);
    auto [sh, ch] = sinhcosh(x);
    DA<5> r = square(ch) - square(sh);
    EXPECT_NEAR(r.value(), 1.0, kTol);
    for (std::size_t k = 1; k < DA<5>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

// =============================================================================
// Tanh
// =============================================================================

TEST(Tanh, Constant) {
    DA<3> a{1.0};
    DA<3> r = tanh(a);
    EXPECT_NEAR(r.value(), std::tanh(1.0), kTol);
    for (std::size_t k = 1; k < DA<3>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(Tanh, AtZero) {
    // tanh(x) at x0=0: coeffs 0, 1, 0, -1/3, 0, 2/15
    auto x = DA<5>::variable(0.0);
    DA<5> r = tanh(x);
    EXPECT_NEAR(r[0], 0.0, kTol);
    EXPECT_NEAR(r[1], 1.0, kTol);
    EXPECT_NEAR(r[2], 0.0, kTol);
    EXPECT_NEAR(r[3], -1.0/3.0, kTol);
    EXPECT_NEAR(r[4], 0.0, kTol);
    EXPECT_NEAR(r[5], 2.0/15.0, kTol);
}

TEST(Tanh, MatchesSinhOverCosh) {
    auto x = DA<5>::variable(1.0);
    DA<5> r1 = tanh(x);
    DA<5> r2 = sinh(x) / cosh(x);
    ExpectCoeffsNear(r1, r2);
}

TEST(Tanh, DerivativeCheck) {
    // d/dx tanh(x) = 1 - tanh^2(x) = sech^2(x) = 1/cosh^2(x)
    constexpr double x0 = 0.5;
    auto x = DA<3>::variable(x0);
    DA<3> r = tanh(x);
    double expected_deriv = 1.0 / (std::cosh(x0) * std::cosh(x0));
    EXPECT_NEAR(r.value(), std::tanh(x0), kTol);
    EXPECT_NEAR(r.derivative({1}), expected_deriv, kTol);
}

TEST(Tanh, Bivariate) {
    auto [x, y] = DAn<3,2>::variables({0.5, 1.0});
    DAn<3,2> r1 = tanh(x * y);
    DAn<3,2> r2 = sinh(x * y) / cosh(x * y);
    ExpectCoeffsNear(r1, r2);
}

TEST(Tanh, OfExpression) {
    DA<4> a{0.5}, b{0.3};
    DA<4> r1 = tanh(a + b);
    EXPECT_NEAR(r1.value(), std::tanh(0.8), kTol);
}
