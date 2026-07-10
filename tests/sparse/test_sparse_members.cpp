// Sparse member surface (eval / deriv / integ / gradient / hessian), compound
// assignment, and Eigen (tax::la) integration — each checked against the dense
// specialisation on the same input.

#include <gtest/gtest.h>

#include "../testUtils.hpp"

namespace
{
using tax::test::ExpectCoeffsNear;

// f = exp(x)*sin(y) over an order-5 bivariate expansion at (x0, y0): a
// genuinely dense support, exercising the full multi-index machinery.
auto denseF( double x0, double y0 )
{
    using TE2 = tax::TE< 5, 2 >;
    typename TE2::Input p{ x0, y0 };
    auto x = TE2::variable< 0 >( p );
    auto y = TE2::variable< 1 >( p );
    return exp( x ) * sin( y ) + x * y;
}
}  // namespace

// ---- eval ----------------------------------------------------------------

TEST( SparseMembers, EvalMatchesDense )
{
    auto f = denseF( 0.3, 0.2 );
    auto sf = tax::sparse( f );
    typename tax::TE< 5, 2 >::Input dx{ 0.05, -0.03 };
    EXPECT_NEAR( sf.eval( dx ), f.eval( dx ), 1e-12 );
}

TEST( SparseMembers, EvalEigenMatchesDense )
{
    auto f = denseF( 0.3, 0.2 );
    auto sf = tax::sparse( f );
    Eigen::Vector2d dx{ 0.05, -0.03 };
    EXPECT_NEAR( sf.eval( dx ), f.eval( dx ), 1e-12 );
}

// ---- deriv / integ (polynomials) -----------------------------------------

TEST( SparseMembers, DerivMatchesDense )
{
    auto f = denseF( 0.3, 0.2 );
    auto sf = tax::sparse( f );
    ExpectCoeffsNear( sf.deriv< 0 >().dense(), f.deriv< 0 >() );
    ExpectCoeffsNear( sf.deriv< 1 >().dense(), f.deriv< 1 >() );
    ExpectCoeffsNear( sf.deriv( 0 ).dense(), f.deriv( 0 ) );
}

TEST( SparseMembers, IntegMatchesDense )
{
    auto f = denseF( 0.3, 0.2 );
    auto sf = tax::sparse( f );
    ExpectCoeffsNear( sf.integ< 0 >().dense(), f.integ< 0 >() );
    ExpectCoeffsNear( sf.integ< 1 >().dense(), f.integ< 1 >() );
    ExpectCoeffsNear( sf.integ( 1 ).dense(), f.integ( 1 ) );
}

TEST( SparseMembers, DerivIntegRoundTrip )
{
    auto f = denseF( 0.3, 0.2 );
    auto sf = tax::sparse( f );
    // d/dx of the x-integral recovers f (up to the top-degree truncation).
    ExpectCoeffsNear( sf.integ< 0 >().deriv< 0 >().dense(), f.integ< 0 >().deriv< 0 >() );
}

TEST( SparseMembers, DerivOutOfRangeThrows )
{
    auto sf = tax::sparse( denseF( 0.3, 0.2 ) );
    EXPECT_THROW( (void)sf.deriv( 2 ), std::out_of_range );
    EXPECT_THROW( (void)sf.integ( -1 ), std::out_of_range );
}

// ---- gradient / hessian --------------------------------------------------

TEST( SparseMembers, GradientMatchesDense )
{
    auto f = denseF( 0.3, 0.2 );
    auto sf = tax::sparse( f );
    EXPECT_TRUE( sf.gradient().isApprox( f.gradient(), 1e-12 ) );
}

TEST( SparseMembers, HessianMatchesDense )
{
    auto f = denseF( 0.3, 0.2 );
    auto sf = tax::sparse( f );
    EXPECT_TRUE( sf.hessian().isApprox( f.hessian(), 1e-12 ) );
}

// ---- compound assignment -------------------------------------------------

TEST( SparseMembers, CompoundAssignMatchesDense )
{
    auto x = tax::TE< 5 >::variable( 1.5 );
    auto y = tax::TE< 5 >::variable( 0.5 );

    auto sx = tax::sparse( x );
    auto sy = tax::sparse( y );

    sx += sy;
    x += y;
    ExpectCoeffsNear( sx.dense(), x );

    sx -= sy;
    x -= y;
    ExpectCoeffsNear( sx.dense(), x );

    sx *= sy;
    x *= y;
    ExpectCoeffsNear( sx.dense(), x );

    sx /= sy;
    x /= y;
    ExpectCoeffsNear( sx.dense(), x );
}

TEST( SparseMembers, CompoundAssignScalarMatchesDense )
{
    auto x = tax::TE< 5 >::variable( 1.5 );
    auto sx = tax::sparse( x );

    sx += 2.0;
    x += 2.0;
    ExpectCoeffsNear( sx.dense(), x );

    sx -= 0.5;
    x -= 0.5;
    ExpectCoeffsNear( sx.dense(), x );

    sx *= 3.0;
    x *= 3.0;
    ExpectCoeffsNear( sx.dense(), x );

    sx /= 4.0;
    x /= 4.0;
    ExpectCoeffsNear( sx.dense(), x );
}

// ---- Eigen (tax::la) integration on sparse scalars -----------------------

TEST( SparseLa, VariablesAndValue )
{
    using STE2 = tax::STE< 4, 2 >;
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< STE2 >( x0 );
    ASSERT_EQ( v.size(), 2 );
    EXPECT_NEAR( v( 0 ).value(), 1.0, 1e-14 );
    EXPECT_NEAR( v( 1 ).value(), 2.0, 1e-14 );

    // A vector-valued map, then value() of it.
    Eigen::Matrix< STE2, 2, 1 > F;
    F( 0 ) = v( 0 ) * v( 1 );
    F( 1 ) = v( 0 ) + v( 1 );
    auto val = tax::la::value( F );
    EXPECT_NEAR( val( 0 ), 2.0, 1e-14 );
    EXPECT_NEAR( val( 1 ), 3.0, 1e-14 );
}

TEST( SparseLa, GradientHessianJacobian )
{
    using STE2 = tax::STE< 4, 2 >;
    using DTE2 = tax::TE< 4, 2 >;
    Eigen::Vector2d x0{ 0.5, 0.3 };

    auto sv = tax::la::variables< STE2 >( x0 );
    auto dv = tax::la::variables< DTE2 >( x0 );

    auto sf = exp( sv( 0 ) ) * sv( 1 ) + sin( sv( 0 ) );
    auto df = exp( dv( 0 ) ) * dv( 1 ) + sin( dv( 0 ) );

    EXPECT_TRUE( tax::la::gradient( sf ).isApprox( tax::la::gradient( df ), 1e-12 ) );
    EXPECT_TRUE( tax::la::hessian( sf ).isApprox( tax::la::hessian( df ), 1e-12 ) );

    Eigen::Matrix< STE2, 2, 1 > sF;
    sF( 0 ) = sf;
    sF( 1 ) = sv( 0 ) * sv( 1 );
    Eigen::Matrix< DTE2, 2, 1 > dF;
    dF( 0 ) = df;
    dF( 1 ) = dv( 0 ) * dv( 1 );

    EXPECT_TRUE( tax::la::jacobian( sF ).isApprox( tax::la::jacobian( dF ), 1e-12 ) );
}

TEST( SparseLa, EvalVector )
{
    using STE2 = tax::STE< 4, 2 >;
    Eigen::Vector2d x0{ 0.5, 0.3 };
    auto sv = tax::la::variables< STE2 >( x0 );
    Eigen::Matrix< STE2, 2, 1 > F;
    F( 0 ) = sv( 0 ) * sv( 1 );
    F( 1 ) = sv( 0 ) + sv( 1 );

    Eigen::Vector2d dx{ 0.01, 0.02 };
    auto out = tax::la::eval( F, dx );
    EXPECT_NEAR( out( 0 ), ( 0.5 + 0.01 ) * ( 0.3 + 0.02 ), 1e-12 );
    EXPECT_NEAR( out( 1 ), ( 0.5 + 0.01 ) + ( 0.3 + 0.02 ), 1e-12 );
}

TEST( SparseLa, NormAndDotMatchDense )
{
    using STE2 = tax::STE< 4, 2 >;
    using DTE2 = tax::TE< 4, 2 >;
    Eigen::Vector2d x0{ 0.5, 0.3 };
    auto sv = tax::la::variables< STE2 >( x0 );
    auto dv = tax::la::variables< DTE2 >( x0 );

    Eigen::Matrix< STE2, 3, 1 > sV;
    sV << sv( 0 ), sv( 1 ), sv( 0 ) * sv( 1 );
    Eigen::Matrix< DTE2, 3, 1 > dV;
    dV << dv( 0 ), dv( 1 ), dv( 0 ) * dv( 1 );

    ExpectCoeffsNear( tax::la::norm( sV ).dense(), tax::la::norm( dV ) );
    ExpectCoeffsNear( tax::la::dot( sV, sV ).dense(), tax::la::dot( dV, dV ) );
}

TEST( SparseLa, InvertRoundTrip )
{
    // Formal inverse of a near-identity map; composing back must give identity.
    using STE2 = tax::STE< 5, 2 >;
    using DTE2 = tax::TE< 5, 2 >;
    Eigen::Vector2d x0{ 0.0, 0.0 };
    auto sv = tax::la::variables< STE2 >( x0 );
    auto dv = tax::la::variables< DTE2 >( x0 );

    Eigen::Matrix< STE2, 2, 1 > sF;
    sF( 0 ) = sv( 0 ) + 0.5 * sv( 1 ) * sv( 1 );
    sF( 1 ) = sv( 1 ) + 0.5 * sv( 0 ) * sv( 0 );
    Eigen::Matrix< DTE2, 2, 1 > dF;
    dF( 0 ) = dv( 0 ) + 0.5 * dv( 1 ) * dv( 1 );
    dF( 1 ) = dv( 1 ) + 0.5 * dv( 0 ) * dv( 0 );

    auto sInv = tax::la::invert( sF );
    auto dInv = tax::la::invert( dF );
    ExpectCoeffsNear( sInv( 0 ).dense(), dInv( 0 ) );
    ExpectCoeffsNear( sInv( 1 ).dense(), dInv( 1 ) );
}
