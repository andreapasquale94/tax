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

TEST( DaceUnivariate, NormEstimate )
{
    constexpr int N = 40;
    constexpr unsigned int nc = 45;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.cos();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::cos( x );

    for ( const auto type : { 0u, 1u, 2u, 4u } )
    {
        const auto yr_est = yr.estimNorm( 0u, type, nc );
        const auto y_est = y.coeffsNormEstimate( 0u, type, nc );
        ASSERT_EQ( yr_est.size(), y_est.size() );
        for ( std::size_t i = 0; i < yr_est.size(); ++i )
            EXPECT_NEAR( yr_est[i], y_est[i], 1e-10 ) << "type=" << type << ", i=" << i;
    }
}

TEST( DaceMultivariate, NormEstimateGroupedByVariable )
{
    constexpr int N = 12;
    constexpr unsigned int nc = 15;

    DACE::DA::init( N, 2 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );

    auto xr = dxr + 0.1;
    auto yr = dyr - 0.2;
    DACE::DA fr = ( xr * yr ).sin() + xr * xr + 0.5 * yr;

    auto [x, y] = tax::TEn< N, 2 >::variables( { 0.1, -0.2 } );
    tax::TEn< N, 2 > f = tax::sin( x * y ) + x * x + 0.5 * y;

    for ( const auto var : { 1u, 2u } )
    {
        for ( const auto type : { 0u, 1u, 2u } )
        {
            const auto fr_est = fr.estimNorm( var, type, nc );
            const auto f_est = f.coeffsNormEstimate( var, type, nc );
            ASSERT_EQ( fr_est.size(), f_est.size() );
            for ( std::size_t i = 0; i < fr_est.size(); ++i )
                EXPECT_NEAR( fr_est[i], f_est[i], 1e-8 )
                    << "var=" << var << ", type=" << type << ", i=" << i;
        }
    }
}
