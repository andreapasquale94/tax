// Sparse math surface: every unary/binary/fused math function must agree,
// coefficient-for-coefficient, with the dense recurrence on the same input.

#include <gtest/gtest.h>

#include "../testUtils.hpp"

namespace
{

// Compare sparse(f)(...) materialised to dense against dense f(...).
using tax::test::ExpectCoeffsNear;

// A univariate order-6 expansion at a benign point (positive, |x0| suitable for
// the inverse-trig / log / sqrt domains used below).
auto denseX( double x0 ) { return tax::TE< 6 >::variable( x0 ); }

// A bivariate order-5 expansion f = x + y at (x0, y0).
auto denseXY( double x0, double y0 )
{
    using TE2 = tax::TE< 5, 2 >;
    typename TE2::Input p{ x0, y0 };
    return TE2::variable< 0 >( p ) + TE2::variable< 1 >( p );
}

}  // namespace

// ---- Unary transcendental / polynomial functions -------------------------

#define TAX_TEST_SPARSE_UNARY( NAME, X0 )                            \
    TEST( SparseMath, NAME##MatchesDense )                           \
    {                                                                \
        auto x = denseX( X0 );                                       \
        auto sx = tax::sparse( x );                                  \
        ExpectCoeffsNear( tax::NAME( sx ).dense(), tax::NAME( x ) ); \
    }

TAX_TEST_SPARSE_UNARY( square, 1.5 )
TAX_TEST_SPARSE_UNARY( cube, 1.5 )
TAX_TEST_SPARSE_UNARY( cbrt, 2.0 )
TAX_TEST_SPARSE_UNARY( exp, 0.3 )
TAX_TEST_SPARSE_UNARY( log, 2.0 )
TAX_TEST_SPARSE_UNARY( sin, 0.4 )
TAX_TEST_SPARSE_UNARY( cos, 0.4 )
TAX_TEST_SPARSE_UNARY( tan, 0.4 )
TAX_TEST_SPARSE_UNARY( asin, 0.3 )
TAX_TEST_SPARSE_UNARY( acos, 0.3 )
TAX_TEST_SPARSE_UNARY( atan, 0.3 )
TAX_TEST_SPARSE_UNARY( sinh, 0.4 )
TAX_TEST_SPARSE_UNARY( cosh, 0.4 )
TAX_TEST_SPARSE_UNARY( tanh, 0.4 )
TAX_TEST_SPARSE_UNARY( asinh, 0.5 )
TAX_TEST_SPARSE_UNARY( acosh, 1.5 )
TAX_TEST_SPARSE_UNARY( atanh, 0.3 )
TAX_TEST_SPARSE_UNARY( erf, 0.4 )

#undef TAX_TEST_SPARSE_UNARY

TEST( SparseMath, UnaryMultivarMatchesDense )
{
    auto f = denseXY( 0.3, 0.2 );
    auto sf = tax::sparse( f );
    ExpectCoeffsNear( tax::exp( sf ).dense(), tax::exp( f ) );
    ExpectCoeffsNear( tax::sin( sf ).dense(), tax::sin( f ) );
    ExpectCoeffsNear( tax::log( sf + 2.0 ).dense(), tax::log( f + 2.0 ) );
}

// ---- Power family --------------------------------------------------------

TEST( SparseMath, PowCompileTimeIntMatchesDense )
{
    auto x = denseX( 1.5 );
    auto sx = tax::sparse( x );
    ExpectCoeffsNear( tax::pow< 4 >( sx ).dense(), tax::pow< 4 >( x ) );
}

TEST( SparseMath, PowRealMatchesDense )
{
    auto x = denseX( 2.0 );
    auto sx = tax::sparse( x );
    ExpectCoeffsNear( tax::pow( sx, 0.5 ).dense(), tax::pow( x, 0.5 ) );
    ExpectCoeffsNear( tax::pow( sx, 2.5 ).dense(), tax::pow( x, 2.5 ) );
    ExpectCoeffsNear( tax::pow( sx, -1.5 ).dense(), tax::pow( x, -1.5 ) );
}

TEST( SparseMath, HalfPowMatchesDense )
{
    auto x = denseX( 3.0 );
    auto sx = tax::sparse( x );
    ExpectCoeffsNear( tax::halfPow< 3 >( sx ).dense(), tax::halfPow< 3 >( x ) );
    ExpectCoeffsNear( tax::halfPow< -3 >( sx ).dense(), tax::halfPow< -3 >( x ) );
}

TEST( SparseMath, InvSqrtPowMatchesDense )
{
    auto x = denseX( 3.0 );
    auto sx = tax::sparse( x );
    ExpectCoeffsNear( tax::invSqrtPow< 3 >( sx ).dense(), tax::invSqrtPow< 3 >( x ) );
}

TEST( SparseMath, PowRationalMatchesDense )
{
    auto x = denseX( 2.0 );
    auto sx = tax::sparse( x );
    ExpectCoeffsNear( ( tax::pow< 2, 5 >( sx ).dense() ), ( tax::pow< 2, 5 >( x ) ) );
    ExpectCoeffsNear( ( tax::pow< 6, 3 >( sx ).dense() ), ( tax::pow< 6, 3 >( x ) ) );
}

TEST( SparseMath, PowTaylorExponentMatchesDense )
{
    auto a = denseX( 2.0 );
    auto b = tax::TE< 6 >::variable( 0.5 );
    auto sa = tax::sparse( a ), sb = tax::sparse( b );
    ExpectCoeffsNear( tax::pow( sa, sb ).dense(), tax::pow( a, b ) );
    ExpectCoeffsNear( tax::pow( 2.0, sb ).dense(), tax::pow( 2.0, b ) );
}

// ---- atan2 ---------------------------------------------------------------

TEST( SparseMath, Atan2MatchesDense )
{
    auto y = tax::TE< 6 >::variable( 1.0 );
    auto x = denseX( 2.0 );
    auto sy = tax::sparse( y ), sx = tax::sparse( x );
    ExpectCoeffsNear( tax::atan2( sy, sx ).dense(), tax::atan2( y, x ) );
    ExpectCoeffsNear( tax::atan2( sy, 2.0 ).dense(), tax::atan2( y, 2.0 ) );
    ExpectCoeffsNear( tax::atan2( 1.0, sx ).dense(), tax::atan2( 1.0, x ) );
}

// ---- Fused ---------------------------------------------------------------

TEST( SparseMath, SinCosMatchesDense )
{
    auto x = denseX( 0.4 );
    auto sx = tax::sparse( x );
    auto [ss, sc] = tax::sinCos( sx );
    auto [ds, dc] = tax::sinCos( x );
    ExpectCoeffsNear( ss.dense(), ds );
    ExpectCoeffsNear( sc.dense(), dc );
}

TEST( SparseMath, SinhCoshMatchesDense )
{
    auto x = denseX( 0.4 );
    auto sx = tax::sparse( x );
    auto [ss, sc] = tax::sinhCosh( sx );
    auto [ds, dc] = tax::sinhCosh( x );
    ExpectCoeffsNear( ss.dense(), ds );
    ExpectCoeffsNear( sc.dense(), dc );
}

TEST( SparseMath, SqrtInvSqrtMatchesDense )
{
    auto x = denseX( 3.0 );
    auto sx = tax::sparse( x );
    auto [sr, si] = tax::sqrtInvSqrt( sx );
    auto [dr, di] = tax::sqrtInvSqrt( x );
    ExpectCoeffsNear( sr.dense(), dr );
    ExpectCoeffsNear( si.dense(), di );
}

// ---- Sparsity preservation ----------------------------------------------
// A function of one axis of a multivariate expansion must never populate
// monomials of the other axes — the native recurrence walks only the additive
// closure of the input's support, so cross-axis coefficients stay structurally
// absent (a dense round-trip would compute them all, then drop the zeros).

TEST( SparseMath, PreservesSparsityAcrossAxes )
{
    using TE2 = tax::TE< 5, 2 >;
    typename TE2::Input p{ 0.3, 0.7 };
    auto sx = tax::sparse( TE2::variable< 0 >( p ) );  // depends on x only

    // exp(x): support is {1, x, x^2, ..., x^5} — 6 monomials, no y appears.
    auto e = tax::exp( sx );
    EXPECT_EQ( e.nnz(), 6u );
    // No stored monomial may carry a y exponent.
    for ( auto k : e.support() )
    {
        auto alpha = tax::unflatIndex< 2 >( std::size_t( k ) );
        EXPECT_EQ( alpha[1], 0 ) << "exp(x) leaked a y term at flat " << k;
    }

    // sin/atan likewise stay on the x axis.
    EXPECT_EQ( tax::sin( sx ).nnz(), 6u );
    EXPECT_EQ( tax::atan( sx ).nnz(), 6u );
}

TEST( SparseMath, ExpSinCosMatchesDense )
{
    auto v = tax::TE< 6 >::variable( 0.2 );
    auto u = tax::TE< 6 >::variable( 0.5 );
    auto sv = tax::sparse( v ), su = tax::sparse( u );
    ExpectCoeffsNear( tax::expSin( sv, su ).dense(), tax::expSin( v, u ) );
    ExpectCoeffsNear( tax::expCos( sv, su ).dense(), tax::expCos( v, u ) );
    auto [ssin, scos] = tax::expSinCos( sv, su );
    auto [dsin, dcos] = tax::expSinCos( v, u );
    ExpectCoeffsNear( ssin.dense(), dsin );
    ExpectCoeffsNear( scos.dense(), dcos );
}
