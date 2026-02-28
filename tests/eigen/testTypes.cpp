#include "testUtils.hpp"

#include <tax/eigen/types.hpp>

TEST(EigenTypes, VectorCommaInitializerFromVariables) {
    auto [x, y, z] = DAn<2,3>::variables({1.0, 2.0, 3.0});

    Eigen::Matrix<DAn<2,3>, 3, 1> v;
    v << x, y, z;

    EXPECT_NEAR(v(0).value(), 1.0, kTol);
    EXPECT_NEAR(v(1).value(), 2.0, kTol);
    EXPECT_NEAR(v(2).value(), 3.0, kTol);

    EXPECT_NEAR(v(0).coeff({1,0,0}), 1.0, kTol);
    EXPECT_NEAR(v(1).coeff({0,1,0}), 1.0, kTol);
    EXPECT_NEAR(v(2).coeff({0,0,1}), 1.0, kTol);
}

TEST(EigenTypes, VariablesFromEigenFixedVector) {
    Eigen::Vector3d x0;
    x0 << 1.0, 2.0, 3.0;

    auto [x, y, z] = DAn<2,3>::variables(x0);

    EXPECT_NEAR(x.value(), 1.0, kTol);
    EXPECT_NEAR(y.value(), 2.0, kTol);
    EXPECT_NEAR(z.value(), 3.0, kTol);

    EXPECT_NEAR(x.coeff({1,0,0}), 1.0, kTol);
    EXPECT_NEAR(y.coeff({0,1,0}), 1.0, kTol);
    EXPECT_NEAR(z.coeff({0,0,1}), 1.0, kTol);
}

TEST(EigenTypes, VariablesFromEigenDynamicVector) {
    Eigen::VectorXd x0(3);
    x0 << 4.0, 5.0, 6.0;

    auto [x, y, z] = DAn<2,3>::variables(x0);

    EXPECT_NEAR(x.value(), 4.0, kTol);
    EXPECT_NEAR(y.value(), 5.0, kTol);
    EXPECT_NEAR(z.value(), 6.0, kTol);
}