// tests/regression/testEigenVectorCalc.cpp
//
// DACE-vs-tax parity for vector-calculus extractors built on the tax::la
// Eigen helpers:
//   tax::la::gradient(f)   vs  DACE getCoefficient at unit multi-indices
//   tax::la::hessian(f)    vs  DACE getCoefficient at 2nd-order multi-indices,
//                          with symmetry handling (factor 2 on diagonal)
//   tax::la::jacobian(F)   vs  the row-wise gradient of each component

#include <dace/dace.h>
#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "regressionUtils.hpp"

using tax_regression::prepareInput;

namespace
{
// DACE coefficient at a unit multi-index for variable i (0-based here; DACE
// indices in the multi-index vector are positional, so 0-based is correct).
double daceCoeff1( const DACE::DA& f, int M, int i )
{
    std::vector< unsigned int > idx( static_cast< std::size_t >( M ), 0u );
    idx[std::size_t( i )] = 1u;
    return f.getCoefficient( idx );
}
// Raw (un-scaled) DACE coefficient at second-order multi-index.
double daceCoeff2( const DACE::DA& f, int M, int i, int j )
{
    std::vector< unsigned int > idx( static_cast< std::size_t >( M ), 0u );
    idx[std::size_t( i )] += 1u;
    idx[std::size_t( j )] += 1u;
    return f.getCoefficient( idx );
}
}  // namespace

TEST( DaceVectorCalc, GradientMatchesDace )
{
    constexpr int N = 5;
    constexpr int M = 3;
    DACE::DA::init( N, M );

    DACE::DA d1( 1 ), d2( 2 ), d3( 3 );
    DACE::DA fr = ( prepareInput( d1 ) * prepareInput( d2 ) + prepareInput( d3 ) ).cos();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       f =
        tax::cos( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) + prepareInput( v( 2 ) ) );

    tax::la::VecNT< M, double > g = tax::la::gradient( f );
    for ( int i = 0; i < M; ++i )
    {
        const double ref = daceCoeff1( fr, M, i );  // = ∂f/∂x_i at x0
        EXPECT_NEAR( g( i ), ref, 1e-12 ) << "i=" << i;
    }
}

TEST( DaceVectorCalc, HessianMatchesDace )
{
    constexpr int N = 5;
    constexpr int M = 3;
    DACE::DA::init( N, M );

    DACE::DA d1( 1 ), d2( 2 ), d3( 3 );
    DACE::DA fr = prepareInput( d1 ) * prepareInput( d2 ) * prepareInput( d3 );

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       f =
        prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) * prepareInput( v( 2 ) );

    tax::la::MatNMT< M, M , double > H = tax::la::hessian( f );
    for ( int i = 0; i < M; ++i )
    {
        for ( int j = 0; j < M; ++j )
        {
            // DACE stores raw Taylor coefficients; ∂² f / ∂x_i ∂x_j
            //   = (i == j) ? 2 * c_{2 e_i} : c_{e_i + e_j}
            const double raw = daceCoeff2( fr, M, i, j );
            const double ref = ( i == j ) ? 2.0 * raw : raw;
            EXPECT_NEAR( H( i, j ), ref, 1e-12 ) << "i=" << i << " j=" << j;
        }
    }
}

TEST( DaceVectorCalc, JacobianMatchesDace )
{
    constexpr int N = 4;
    constexpr int M = 2;
    DACE::DA::init( N, M );

    DACE::DA d1( 1 ), d2( 2 );
    DACE::DA fr0 = ( prepareInput( d1 ) + prepareInput( d2 ) ).sin();
    DACE::DA fr1 = ( prepareInput( d1 ) * prepareInput( d2 ) ).cos();

    const Eigen::Vector2d x0{ 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    using TE                = tax::TE< N, M >;
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = tax::sin( prepareInput( v( 0 ) ) + prepareInput( v( 1 ) ) );
    F( 1 ) = tax::cos( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) );

    tax::la::MatNMT< 2, M , double > J       = tax::la::jacobian( F );
    const DACE::DA                refs[2] = { fr0, fr1 };
    for ( int row = 0; row < 2; ++row )
    {
        for ( int col = 0; col < M; ++col )
        {
            const double ref = daceCoeff1( refs[row], M, col );
            EXPECT_NEAR( J( row, col ), ref, 1e-12 ) << "row=" << row << " col=" << col;
        }
    }
}
