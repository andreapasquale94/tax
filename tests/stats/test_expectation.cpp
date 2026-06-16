#include <gtest/gtest.h>

#include <tax/stats.hpp>

using tax::stats::centralMoment;
using tax::stats::covariance;
using tax::stats::expectation;
using tax::stats::GaussianMoments;
using tax::stats::kurtosis;
using tax::stats::mean;
using tax::stats::moments;
using tax::stats::skewness;
using tax::stats::variance;

namespace
{

Eigen::Matrix< double, 1, 1 > cov1( double s2 )
{
    Eigen::Matrix< double, 1, 1 > s;
    s( 0, 0 ) = s2;
    return s;
}

}  // namespace

// Mean of a quadratic in one variable: E[z^2] = sigma^2; Var[z^2] = 2 sigma^4.
// The variance needs E[z^4] (order 2N), exercising the exact double contraction.
TEST( Expectation, UnivariateQuadratic )
{
    const double s2 = 2.5;
    GaussianMoments< double, 1 > dist( cov1( s2 ) );

    auto x = tax::TE< 2 >::variable( 0.0 );  // expansion at the mean (deviation)
    auto f = x * x;                          // f = z^2

    EXPECT_NEAR( expectation( f, dist ), s2, 1e-12 );
    EXPECT_NEAR( variance( f, dist ), 2.0 * s2 * s2, 1e-12 );
}

// Linear map F = A z + b with z ~ N(0, Sigma): E[F] = b, Cov[F] = A Sigma A^T.
TEST( Expectation, LinearMapPushforward )
{
    Eigen::Matrix< double, 2, 2 > S;
    S << 4.0, 1.0, 1.0, 3.0;
    GaussianMoments< double, 2 > dist( S );

    using TE = tax::TE< 3, 2 >;
    auto v = tax::la::variables< TE >( Eigen::Vector2d::Zero() );

    Eigen::Matrix< double, 2, 2 > A;
    A << 2.0, -1.0, 0.5, 3.0;
    Eigen::Vector2d b{ 7.0, -2.0 };

    Eigen::Matrix< TE, 2, 1 > F;
    F( 0 ) = b( 0 ) + A( 0, 0 ) * v( 0 ) + A( 0, 1 ) * v( 1 );
    F( 1 ) = b( 1 ) + A( 1, 0 ) * v( 0 ) + A( 1, 1 ) * v( 1 );

    auto mean = expectation( F, dist );
    EXPECT_NEAR( ( mean - b ).norm(), 0.0, 1e-12 );

    Eigen::Matrix< double, 2, 2 > expected = A * S * A.transpose();
    auto cov = covariance( F, dist );
    EXPECT_NEAR( ( cov - expected ).norm(), 0.0, 1e-12 );
    EXPECT_NEAR( cov( 0, 1 ), cov( 1, 0 ), 1e-14 );  // symmetric
}

// Bilinear scalar f = z0 z1: E[f] = S01, Var[f] = S00 S11 + 2 S01^2.
TEST( Expectation, BilinearVariance )
{
    Eigen::Matrix< double, 2, 2 > S;
    S << 2.0, 0.7, 0.7, 5.0;
    GaussianMoments< double, 2 > dist( S );

    using TE = tax::TE< 2, 2 >;
    auto v = tax::la::variables< TE >( Eigen::Vector2d::Zero() );
    auto f = v( 0 ) * v( 1 );

    // Var[z0 z1] = E[z0^2 z1^2] - E[z0 z1]^2 = (S00 S11 + 2 S01^2) - S01^2.
    EXPECT_NEAR( expectation( f, dist ), S( 0, 1 ), 1e-12 );
    EXPECT_NEAR( variance( f, dist ), S( 0, 0 ) * S( 1, 1 ) + S( 0, 1 ) * S( 0, 1 ), 1e-12 );
}

// A Gaussian linear form is symmetric and mesokurtic: skewness 0, excess kurtosis 0.
TEST( Expectation, StandardizedMomentsOfLinearGaussian )
{
    const double s2 = 1.7;
    GaussianMoments< double, 1 > dist( cov1( s2 ) );

    auto x = tax::TE< 4 >::variable( 0.0 );
    auto f = 3.0 * x + 5.0;  // affine: mean 5, variance 9 s2

    EXPECT_NEAR( expectation( f, dist ), 5.0, 1e-12 );
    EXPECT_NEAR( variance( f, dist ), 9.0 * s2, 1e-12 );
    EXPECT_NEAR( skewness( f, dist ), 0.0, 1e-10 );
    EXPECT_NEAR( kurtosis( f, dist, /*excess=*/true ), 0.0, 1e-10 );
    EXPECT_NEAR( kurtosis( f, dist ), 3.0, 1e-10 );
}

// centralMoment(.,2) reproduces the variance for a polynomial whose deviation
// stays within the truncation order.
TEST( Expectation, CentralMomentMatchesVariance )
{
    const double s2 = 2.0;
    GaussianMoments< double, 1 > dist( cov1( s2 ) );

    auto x = tax::TE< 4 >::variable( 0.0 );
    auto f = 2.0 * x;  // deviation is degree 1, so (f-mean)^2 is exact at N=4

    EXPECT_NEAR( centralMoment( f, dist, 2 ), variance( f, dist ), 1e-12 );
    EXPECT_NEAR( centralMoment( f, dist, 0 ), 1.0, 1e-12 );
}

// The one-shot moments() summary agrees with the individual functions.
TEST( Expectation, MomentsSummaryMatchesIndividual )
{
    const double s2 = 1.3;
    GaussianMoments< double, 1 > dist( cov1( s2 ) );

    auto x = tax::TE< 4 >::variable( 0.0 );
    auto f = 1.0 + 2.0 * x + 0.5 * ( x * x );  // mildly nonlinear

    auto m = moments( f, dist );
    EXPECT_NEAR( m.mean, mean( f, dist ), 1e-12 );
    EXPECT_NEAR( m.variance, variance( f, dist ), 1e-12 );
    EXPECT_NEAR( m.skewness, skewness( f, dist ), 1e-10 );
    EXPECT_NEAR( m.kurtosis, kurtosis( f, dist ), 1e-10 );
    EXPECT_NEAR( m.standardDeviation(), std::sqrt( variance( f, dist ) ), 1e-12 );
    EXPECT_NEAR( m.excessKurtosis(), kurtosis( f, dist, /*excess=*/true ), 1e-10 );
}

// mean() is a thin alias of expectation() for both scalar and vector forms.
TEST( Expectation, MeanAliasMatchesExpectation )
{
    Eigen::Matrix< double, 2, 2 > S;
    S << 2.0, 0.3, 0.3, 1.5;
    GaussianMoments< double, 2 > dist( S );

    using TE = tax::TE< 3, 2 >;
    auto v = tax::la::variables< TE >( Eigen::Vector2d::Zero() );

    Eigen::Matrix< TE, 2, 1 > F;
    F( 0 ) = 4.0 + v( 0 ) * v( 1 );
    F( 1 ) = -1.0 + v( 0 ) * v( 0 );

    auto me = mean( F, dist );
    auto ex = expectation( F, dist );
    EXPECT_NEAR( ( me - ex ).norm(), 0.0, 1e-14 );
    EXPECT_NEAR( mean( F( 0 ), dist ), expectation( F( 0 ), dist ), 1e-14 );
}
