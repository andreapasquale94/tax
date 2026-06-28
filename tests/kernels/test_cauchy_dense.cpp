#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/expansion/detail/cauchy.hpp>

TEST(CauchyDense, AgreesWithOperator) {
    auto x = tax::TE<5>::variable(0.3);
    auto y = tax::TE<5>::variable(0.7);
    auto z = x * y;
    tax::Coeffs<double, 5, 1> direct{};
    tax::detail::kernels::cauchyProduct<double, 5, 1>(direct, x.coefficients(), y.coefficients());
    for (std::size_t k = 0; k < z.nCoefficients; ++k)
        EXPECT_DOUBLE_EQ(direct[k], z[k]);
}
