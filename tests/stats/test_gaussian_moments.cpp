#include <gtest/gtest.h>

#include <tax/stats.hpp>

using tax::MultiIndex;
using tax::stats::GaussianMoments;

namespace
{

Eigen::Matrix< double, 1, 1 > cov1( double s2 )
{
    Eigen::Matrix< double, 1, 1 > s;
    s( 0, 0 ) = s2;
    return s;
}

}  // namespace

// Univariate N(0, sigma^2): E[z^{2k}] = (2k-1)!! * sigma^{2k}, odd moments zero.
TEST( GaussianMoments, UnivariateEvenAndOddMoments )
{
    const double s2 = 2.0;
    GaussianMoments< double, 1 > dist( cov1( s2 ) );
    const auto m = dist.momentTable( 6 );

    EXPECT_NEAR( m[0], 1.0, 1e-12 );                  // E[1]
    EXPECT_NEAR( m[1], 0.0, 1e-12 );                  // E[z]
    EXPECT_NEAR( m[2], s2, 1e-12 );                   // E[z^2] = sigma^2
    EXPECT_NEAR( m[3], 0.0, 1e-12 );                  // E[z^3]
    EXPECT_NEAR( m[4], 3.0 * s2 * s2, 1e-12 );        // E[z^4] = 3 sigma^4
    EXPECT_NEAR( m[5], 0.0, 1e-12 );                  // E[z^5]
    EXPECT_NEAR( m[6], 15.0 * s2 * s2 * s2, 1e-12 );  // E[z^6] = 15 sigma^6
}

// Bivariate Isserlis identities against a non-diagonal covariance.
TEST( GaussianMoments, BivariateIsserlis )
{
    Eigen::Matrix< double, 2, 2 > S;
    S << 4.0, 1.5, 1.5, 9.0;
    GaussianMoments< double, 2 > dist( S );

    auto E = [&]( int a, int b ) { return dist.rawMoment( MultiIndex< 2 >{ a, b } ); };

    EXPECT_NEAR( E( 1, 1 ), S( 0, 1 ), 1e-12 );                                   // E[z1 z2] = S01
    EXPECT_NEAR( E( 2, 0 ), S( 0, 0 ), 1e-12 );                                   // E[z1^2]  = S00
    EXPECT_NEAR( E( 1, 0 ), 0.0, 1e-12 );                                         // odd
    EXPECT_NEAR( E( 3, 1 ), 3.0 * S( 0, 0 ) * S( 0, 1 ), 1e-12 );                 // E[z1^3 z2]
    EXPECT_NEAR( E( 2, 2 ), S( 0, 0 ) * S( 1, 1 ) + 2.0 * S( 0, 1 ) * S( 0, 1 ),  // E[z1^2 z2^2]
                 1e-12 );
}

// diagonal() factory matches an explicit diagonal covariance.
TEST( GaussianMoments, DiagonalFactory )
{
    Eigen::Matrix< double, 2, 1 > var;
    var << 3.0, 5.0;
    auto dist = GaussianMoments< double, 2 >::diagonal( var );

    EXPECT_NEAR( dist.rawMoment( MultiIndex< 2 >{ 2, 0 } ), 3.0, 1e-12 );
    EXPECT_NEAR( dist.rawMoment( MultiIndex< 2 >{ 0, 2 } ), 5.0, 1e-12 );
    EXPECT_NEAR( dist.rawMoment( MultiIndex< 2 >{ 1, 1 } ), 0.0, 1e-12 );  // independent
    EXPECT_NEAR( dist.rawMoment( MultiIndex< 2 >{ 2, 2 } ), 3.0 * 5.0, 1e-12 );
}
