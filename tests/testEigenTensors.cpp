#include "testUtils.hpp"

#if __has_include(<unsupported/Eigen/CXX11/Tensor>)

#include <tax/eigen/tensors.hpp>

TEST(EigenTensors, DerivativeTensorGradientAndHessian) {
    auto [x, y] = DAn<2,2>::variables({1.0, 2.0});
    DAn<2,2> f = x*x + x*y + y*y;

    const auto g = tax::derivativeTensor<1>(f);
    EXPECT_NEAR(g(0), 4.0, kTol);
    EXPECT_NEAR(g(1), 5.0, kTol);

    const auto h = tax::derivativeTensor<2>(f);
    EXPECT_NEAR(h(0,0), 2.0, kTol);
    EXPECT_NEAR(h(0,1), 1.0, kTol);
    EXPECT_NEAR(h(1,0), 1.0, kTol);
    EXPECT_NEAR(h(1,1), 2.0, kTol);
}

TEST(EigenTensors, CoeffTensorOrder2Normalized) {
    auto [x, y] = DAn<2,2>::variables({0.0, 0.0});
    DAn<2,2> f = x*x + x*y + y*y;

    const auto c2 = tax::coeffTensor<2>(f);
    EXPECT_NEAR(c2(0,0), 1.0, kTol);  // coeff x^2
    EXPECT_NEAR(c2(1,1), 1.0, kTol);  // coeff y^2
    EXPECT_NEAR(c2(0,1), 0.5, kTol);  // normalized split of xy term
    EXPECT_NEAR(c2(1,0), 0.5, kTol);  // normalized split of xy term
}

#else

TEST(EigenTensors, HeaderUnavailable) { GTEST_SKIP() << "Eigen tensor headers not available"; }

#endif
