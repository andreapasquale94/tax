#include <dace/dace.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <tax/tax.hpp>

TEST( DaceUnivariate, Norm )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.cos();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::cos( x );

    EXPECT_NEAR( yr.norm( 0 ), y.coeffsNormInf(), 1e-10 );
    for ( int i = 1; i < N; ++i ) EXPECT_NEAR( yr.norm( i ), y.coeffsNorm( i ), 1e-10 );
}
