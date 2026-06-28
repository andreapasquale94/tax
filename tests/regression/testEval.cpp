// tests/regression/testEval.cpp
//
// DACE-vs-tax parity for displacement evaluation:
//   tax::TE<N>::eval(dx)        vs  DACE::DA::eval(...)
//   tax::TE<N, M>::eval(dx)     vs  DACE::DA::eval(...)

#include <dace/dace.h>
#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <cmath>
#include <tax/tax.hpp>

#include "regressionUtils.hpp"

using tax_regression::prepareInput;

TEST( DaceEval, UnivariateAtDx )
{
    constexpr int N = 20;
    DACE::DA::init( N, 1 );

    DACE::DA xr( 1 );
    DACE::DA fr = prepareInput( xr ).sin();

    auto x = tax::TE< N >::variable( 0.0 );
    auto f = tax::sin( prepareInput( x ) );

    for ( const double dx : { -0.3, -0.1, 0.0, 0.1, 0.3 } )
    {
        const double ref = fr.eval( std::vector< double >{ dx } );
        const double got = f.eval( tax::la::VecNT< 1, double >( dx ) );
        EXPECT_NEAR( ref, got, 1e-12 ) << "dx=" << dx;
    }
}

TEST( DaceEval, MultivariateAtDx )
{
    constexpr int N = 6;
    constexpr int M = 2;
    DACE::DA::init( N, M );

    DACE::DA dxr( 1 ), dyr( 2 );
    DACE::DA fr = ( prepareInput( dxr ) + prepareInput( dyr ) ).exp();

    const Eigen::Vector2d x0{ 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    auto                  f = tax::exp( prepareInput( v( 0 ) ) + prepareInput( v( 1 ) ) );

    const Eigen::Vector2d displacements[] = {
        { -0.2, -0.2 }, { -0.1, 0.1 }, { 0.0, 0.0 }, { 0.1, -0.1 }, { 0.2, 0.2 }
    };
    for ( const auto& dx : displacements )
    {
        const double ref = fr.eval( std::vector< double >{ dx( 0 ), dx( 1 ) } );
        const double got = f.eval( dx );
        EXPECT_NEAR( ref, got, 1e-12 ) << "dx=" << dx.transpose();
    }
}
