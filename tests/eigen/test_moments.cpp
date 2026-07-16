#include <gtest/gtest.h>

#include <tax/tax.hpp>

// All tests assume the expansion's formal variables are i.i.d. standard normal
// (x ~ N(0, I)), per tax::la::mean/covariance/skewnessTensor/kurtosisTensor's
// documented convention.

TEST( Moments, MeanAndVarianceOfXSquared )
{
    // Y = X^2, X ~ N(0,1) is chi-square(1): E[Y]=1, Var(Y)=2.
    auto x = tax::TE< 4 >::variable( 0.0 );
    Eigen::Matrix< tax::TE< 4 >, 1, 1 > F;
    F( 0 ) = x * x;

    const auto mu = tax::la::mean( F );
    EXPECT_NEAR( mu( 0 ), 1.0, 1e-12 );

    const auto C = tax::la::covariance( F );
    EXPECT_NEAR( C( 0, 0 ), 2.0, 1e-12 );
}

TEST( Moments, SkewnessOfChiSquare1 )
{
    // chi-square(1) third central moment is 8.
    auto x = tax::TE< 4 >::variable( 0.0 );
    Eigen::Matrix< tax::TE< 4 >, 1, 1 > F;
    F( 0 ) = x * x;

    const auto S = tax::la::skewnessTensor( F );
    ASSERT_EQ( S.dimension( 0 ), 1 );
    EXPECT_NEAR( S( 0, 0, 0 ), 8.0, 1e-9 );
}

TEST( Moments, KurtosisOfChiSquare1 )
{
    // chi-square(1): 4th central moment 60, excess kurtosis 12 (=> excess
    // tensor entry 12 * Var^2 = 12 * 4 = 48).
    auto x = tax::TE< 4 >::variable( 0.0 );
    Eigen::Matrix< tax::TE< 4 >, 1, 1 > F;
    F( 0 ) = x * x;

    const auto K = tax::la::kurtosisTensor( F );
    // Return type is a fixed-size (compile-time D x D x D x D) tensor.
    static_assert(
        std::is_same_v< decltype( K ),
                        const Eigen::TensorFixedSize< double, Eigen::Sizes< 1, 1, 1, 1 > > >,
        "kurtosisTensor must return a fixed-size rank-4 tensor" );
    EXPECT_NEAR( K( 0, 0, 0, 0 ), 60.0, 1e-8 );

    const auto Kexcess = tax::la::excessKurtosisTensor( F );
    EXPECT_NEAR( Kexcess( 0, 0, 0, 0 ), 48.0, 1e-8 );
}

TEST( Moments, OddFunctionHasZeroMeanAndSkew )
{
    auto x = tax::TE< 3 >::variable( 0.0 );
    Eigen::Matrix< tax::TE< 3 >, 1, 1 > F;
    F( 0 ) = x * x * x;  // odd function of a symmetric distribution

    const auto mu = tax::la::mean( F );
    EXPECT_NEAR( mu( 0 ), 0.0, 1e-12 );

    const auto S = tax::la::skewnessTensor( F );
    EXPECT_NEAR( S( 0, 0, 0 ), 0.0, 1e-9 );
}

TEST( Moments, IndependentSumOfSquaresVariance )
{
    // F = X^2 + Y^2, X,Y iid N(0,1): E[F]=2, Var(F)=Var(X^2)+Var(Y^2)=2+2=4.
    typename tax::TE< 4, 2 >::Input p{ 0.0, 0.0 };
    auto x = tax::TE< 4, 2 >::variable< 0 >( p );
    auto y = tax::TE< 4, 2 >::variable< 1 >( p );
    Eigen::Matrix< tax::TE< 4, 2 >, 1, 1 > F;
    F( 0 ) = x * x + y * y;

    const auto mu = tax::la::mean( F );
    EXPECT_NEAR( mu( 0 ), 2.0, 1e-12 );

    const auto C = tax::la::covariance( F );
    EXPECT_NEAR( C( 0, 0 ), 4.0, 1e-12 );
}

TEST( Moments, SkewnessTensorIsSymmetricWithCrossMoments )
{
    // F = [X, X^2], X ~ N(0,1). Centered: c0 = X, c1 = X^2 - 1.
    //   S_000 = E[X^3]           = 0
    //   S_001 = E[X^2 (X^2-1)]   = E[X^4] - E[X^2] = 3 - 1 = 2
    //   S_011 = E[X (X^2-1)^2]   = 0            (odd moment)
    //   S_111 = E[(X^2-1)^3]     = 8            (chi-square(1) skew)
    auto x = tax::TE< 6 >::variable( 0.0 );
    Eigen::Matrix< tax::TE< 6 >, 2, 1 > F;
    F( 0 ) = x;
    F( 1 ) = x * x;

    const auto S = tax::la::skewnessTensor( F );
    // Return type is a fixed-size (compile-time D x D x D) tensor.
    static_assert(
        std::is_same_v< decltype( S ),
                        const Eigen::TensorFixedSize< double, Eigen::Sizes< 2, 2, 2 > > >,
        "skewnessTensor must return a fixed-size rank-3 tensor" );
    ASSERT_EQ( S.dimension( 0 ), 2 );
    ASSERT_EQ( S.dimension( 1 ), 2 );
    ASSERT_EQ( S.dimension( 2 ), 2 );

    EXPECT_NEAR( S( 0, 0, 0 ), 0.0, 1e-8 );
    EXPECT_NEAR( S( 1, 1, 1 ), 8.0, 1e-8 );

    // S_001 == 2 and equal across all three permutations of (0,0,1).
    EXPECT_NEAR( S( 0, 0, 1 ), 2.0, 1e-8 );
    EXPECT_NEAR( S( 0, 1, 0 ), 2.0, 1e-8 );
    EXPECT_NEAR( S( 1, 0, 0 ), 2.0, 1e-8 );

    // S_011 == 0 across all three permutations of (0,1,1).
    EXPECT_NEAR( S( 0, 1, 1 ), 0.0, 1e-8 );
    EXPECT_NEAR( S( 1, 0, 1 ), 0.0, 1e-8 );
    EXPECT_NEAR( S( 1, 1, 0 ), 0.0, 1e-8 );
}

TEST( Moments, CrossCovarianceOfXAndXSquaredIsZero )
{
    // Cov(X, X^2) = E[X^3] - E[X]E[X^2] = 0 - 0 = 0 (odd moment of a
    // symmetric distribution).
    auto x = tax::TE< 4 >::variable( 0.0 );
    Eigen::Matrix< tax::TE< 4 >, 2, 1 > F;
    F( 0 ) = x;
    F( 1 ) = x * x;

    const auto C = tax::la::covariance( F );
    EXPECT_NEAR( C( 0, 0 ), 1.0, 1e-12 );
    EXPECT_NEAR( C( 1, 1 ), 2.0, 1e-12 );
    EXPECT_NEAR( C( 0, 1 ), 0.0, 1e-12 );
    EXPECT_NEAR( C( 1, 0 ), 0.0, 1e-12 );
}
