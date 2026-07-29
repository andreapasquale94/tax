#include <gtest/gtest.h>
#include <tax/tax.hpp>

// Concept self-checks (moved out of taylor_expansion.hpp so they are
// compiled once here instead of in every consumer translation unit).
static_assert(tax::TaylorPolynomial<tax::TE<3>>, "TE<3> must satisfy TaylorPolynomial");
static_assert(tax::DensePolynomial<tax::TE<3>>, "TE<3> must satisfy DensePolynomial");
static_assert(tax::TaylorPolynomial<tax::TE<3, 2>>, "TE<3,2> must satisfy TaylorPolynomial");
static_assert(tax::DensePolynomial<tax::TE<3, 2>>, "TE<3,2> must satisfy DensePolynomial");

TEST(TaylorExpansion, ZeroCtor) {
    tax::TE<3> z;
    EXPECT_EQ(z.value(), 0.0);
    for (std::size_t k = 0; k < z.nCoefficients; ++k) EXPECT_EQ(z[k], 0.0);
}

TEST(TaylorExpansion, ConstantCtor) {
    tax::TE<3> c{2.5};
    EXPECT_EQ(c.value(), 2.5);
    for (std::size_t k = 1; k < c.nCoefficients; ++k) EXPECT_EQ(c[k], 0.0);
}

TEST(TaylorExpansion, VariableFactoryUni) {
    auto x = tax::TE<5>::variable(1.0);
    EXPECT_EQ(x.value(), 1.0);
    EXPECT_EQ(x[1], 1.0);  // d/dx coefficient
    for (std::size_t k = 2; k < x.nCoefficients; ++k) EXPECT_EQ(x[k], 0.0);
}

TEST(TaylorExpansion, VariableFactoryMulti) {
    typename tax::TE<3, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<3, 2>::variable<0>(p);
    auto y = tax::TE<3, 2>::variable<1>(p);
    EXPECT_EQ(x.value(), 1.0);
    EXPECT_EQ(y.value(), 2.0);
    EXPECT_EQ(x.coeff(tax::MultiIndex<2>{1, 0}), 1.0);
    EXPECT_EQ(x.coeff(tax::MultiIndex<2>{0, 1}), 0.0);
    EXPECT_EQ(y.coeff(tax::MultiIndex<2>{1, 0}), 0.0);
    EXPECT_EQ(y.coeff(tax::MultiIndex<2>{0, 1}), 1.0);
}
