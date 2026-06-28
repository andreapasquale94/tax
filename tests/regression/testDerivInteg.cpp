// tests/regression/testDerivInteg.cpp
//
// DACE-vs-tax parity for symbolic differentiation and integration of
// Taylor expansions: f.deriv<I>() / f.integ<I>() against
// DACE::DA::deriv / DACE::DA::integ.

#include <dace/dace.h>
#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "regressionUtils.hpp"

using tax_regression::expectCoeffsMatch;
using tax_regression::prepareInput;

TEST( DaceDerivInteg, UnivariateDeriv )
{
    constexpr int N = 20;
    DACE::DA::init( N, 1 );

    DACE::DA xr( 1 );
    DACE::DA f_ref  = prepareInput( xr ).sin();
    DACE::DA df_ref = f_ref.deriv( 1 );  // 1-based: derivative w.r.t. variable 1

    auto         x  = tax::TE< N >::variable( 0.0 );
    auto         f  = tax::sin( prepareInput( x ) );
    tax::TE< N > df = f.deriv< 0 >();

    EXPECT_TRUE( expectCoeffsMatch( df, df_ref ) );
}

TEST( DaceDerivInteg, UnivariateInteg )
{
    constexpr int N = 20;
    DACE::DA::init( N, 1 );

    DACE::DA xr( 1 );
    DACE::DA f_ref = prepareInput( xr ).cos();
    DACE::DA F_ref = f_ref.integ( 1 );

    auto         x = tax::TE< N >::variable( 0.0 );
    auto         f = tax::cos( prepareInput( x ) );
    tax::TE< N > F = f.integ< 0 >();

    EXPECT_TRUE( expectCoeffsMatch( F, F_ref ) );
}

TEST( DaceDerivInteg, MultivariateDerivX )
{
    constexpr int N = 6;
    constexpr int M = 2;
    DACE::DA::init( N, M );

    DACE::DA dxr( 1 ), dyr( 2 );
    DACE::DA fr  = ( prepareInput( dxr ) * prepareInput( dyr ) ).exp();
    DACE::DA dfr = fr.deriv( 1 );  // w.r.t. DACE variable 1 (i.e. x)

    const Eigen::Vector2d x0{ 0.0, 0.0 };
    auto                  v  = tax::variables< tax::TE< N, M > >( x0 );
    auto                  f  = tax::exp( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) );
    tax::TE< N, M >       df = f.deriv< 0 >();

    EXPECT_TRUE( expectCoeffsMatch( df, dfr ) );
}

TEST( DaceDerivInteg, MultivariateDerivY )
{
    constexpr int N = 6;
    constexpr int M = 2;
    DACE::DA::init( N, M );

    DACE::DA dxr( 1 ), dyr( 2 );
    DACE::DA fr  = ( prepareInput( dxr ) * prepareInput( dyr ) ).exp();
    DACE::DA dfr = fr.deriv( 2 );

    const Eigen::Vector2d x0{ 0.0, 0.0 };
    auto                  v  = tax::variables< tax::TE< N, M > >( x0 );
    auto                  f  = tax::exp( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) );
    tax::TE< N, M >       df = f.deriv< 1 >();

    EXPECT_TRUE( expectCoeffsMatch( df, dfr ) );
}

TEST( DaceDerivInteg, MultivariateInteg )
{
    constexpr int N = 6;
    constexpr int M = 2;
    DACE::DA::init( N, M );

    DACE::DA dxr( 1 ), dyr( 2 );
    DACE::DA fr = ( prepareInput( dxr ) + prepareInput( dyr ) ).cos();
    DACE::DA Fr = fr.integ( 1 );

    const Eigen::Vector2d x0{ 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    auto                  f = tax::cos( prepareInput( v( 0 ) ) + prepareInput( v( 1 ) ) );
    tax::TE< N, M >       F = f.integ< 0 >();

    EXPECT_TRUE( expectCoeffsMatch( F, Fr ) );
}
