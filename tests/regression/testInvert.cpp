// tests/regression/testInvert.cpp
//
// DACE-vs-tax parity for polynomial map inversion:
//   tax::la::invert(F)                      (Eigen<TE,M,1>)
//   DACE::AlgebraicVector<DA>::invert() (vector of DA)

#include <dace/AlgebraicVector.h>
#include <dace/dace.h>
#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "regressionUtils.hpp"

using tax_regression::expectCoeffsMatch;
using tax_regression::prepareInput;

// Map is identity + small nonlinear perturbation, so the linear part is the
// identity (trivially invertible). Both sides are constructed the same way.
TEST( DaceInvert, IdentityPlusPerturbation )
{
    constexpr int N = 5;
    constexpr int M = 2;
    DACE::DA::init( N, M );

    DACE::DA                          d1( 1 ), d2( 2 );
    DACE::AlgebraicVector< DACE::DA > Fr( 2 );
    Fr[0] = d1 + 0.1 * ( prepareInput( d2 ) - 1.0 );
    Fr[1] = d2 + 0.1 * ( prepareInput( d1 ) - 1.0 );
    DACE::AlgebraicVector< DACE::DA > Fr_inv = Fr.invert();

    const Eigen::Vector2d x0{ 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    using TE                = tax::TE< N, M >;
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = v( 0 ) + 0.1 * ( prepareInput( v( 1 ) ) - 1.0 );
    F( 1 ) = v( 1 ) + 0.1 * ( prepareInput( v( 0 ) ) - 1.0 );
    tax::la::VecNT< 2, TE > F_inv = tax::la::invert( F );

    for ( int i = 0; i < 2; ++i )
    {
        EXPECT_TRUE( expectCoeffsMatch( F_inv( i ), Fr_inv[i] ) ) << "inverse component " << i;
    }
}
