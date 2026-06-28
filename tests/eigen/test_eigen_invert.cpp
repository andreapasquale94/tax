#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/la.hpp>

// The invert() algorithm:
//   - Drops the constant term of each component of the input map.
//   - Inverts the non-constant (perturbation) part via Picard iteration.
//   - Therefore the output is a map from perturbation-space to perturbation-space.
//
// For F(x) = x + 1 (input: v(0)+1 around x0=0), the non-constant part is just x,
// whose inverse is x itself => Finv.value() == 0 (expansion point of the inverse is 0).
//
// We test a more meaningful case: a non-trivial linear map.

TEST( EigenInvert, IdentityMap )
{
    // F = identity map: F_i(x) = x_i
    Eigen::Vector2d x0{ 0.0, 0.0 };
    auto v = tax::la::variables< tax::TE< 3, 2 > >( x0 );
    tax::la::VecNT< 2, tax::TE< 3, 2 > > F;
    F( 0 ) = v( 0 );
    F( 1 ) = v( 1 );
    auto Finv = tax::la::invert( F );
    // inverse of identity is identity
    EXPECT_NEAR( Finv( 0 ).value(), 0.0, 1e-12 );
    EXPECT_NEAR( Finv( 1 ).value(), 0.0, 1e-12 );
    EXPECT_NEAR( ( Finv( 0 ).coeff< 1, 0 >() ), 1.0, 1e-12 );
    EXPECT_NEAR( ( Finv( 1 ).coeff< 0, 1 >() ), 1.0, 1e-12 );
}

TEST( EigenInvert, LinearScaling )
{
    // F = 2*x (univariate). Inverse = x/2.
    tax::la::VecNT< 1, double > x0;
    x0 << 0.0;
    auto v = tax::la::variables< tax::TE< 4, 1 > >( x0 );
    tax::la::VecNT< 1, tax::TE< 4, 1 > > F;
    F( 0 ) = 2.0 * v( 0 );
    auto Finv = tax::la::invert( F );
    EXPECT_NEAR( Finv( 0 ).value(), 0.0, 1e-12 );
    // Linear coefficient of inverse should be 1/2
    EXPECT_NEAR( ( Finv( 0 ).coeff< 1 >() ), 0.5, 1e-12 );
}

TEST( EigenInvert, NonlinearMap )
{
    // F(x) = x + x^2 (univariate). Invert up to order 3.
    // Formal inverse: x - x^2 + 2x^3 - ...
    tax::la::VecNT< 1, double > x0;
    x0 << 0.0;
    auto v = tax::la::variables< tax::TE< 4, 1 > >( x0 );
    tax::la::VecNT< 1, tax::TE< 4, 1 > > F;
    F( 0 ) = v( 0 ) + v( 0 ) * v( 0 );
    auto Finv = tax::la::invert( F );
    // Finv(0): constant = 0, linear = 1, quadratic = -1, cubic = 2
    EXPECT_NEAR( Finv( 0 ).value(), 0.0, 1e-12 );
    EXPECT_NEAR( ( Finv( 0 ).coeff< 1 >() ), 1.0, 1e-10 );
    EXPECT_NEAR( ( Finv( 0 ).coeff< 2 >() ), -1.0, 1e-10 );
    EXPECT_NEAR( ( Finv( 0 ).coeff< 3 >() ), 2.0, 1e-10 );
}
