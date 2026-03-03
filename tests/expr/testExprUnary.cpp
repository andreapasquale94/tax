#include "testUtils.hpp"

// =============================================================================
// UnaryExpr<OpNeg> — negation sign-flip propagation
// =============================================================================

TEST( NegAddTo, NegLeaf_AddedTo_Leaf )
{
    DA< 4 > a{ 7.0 }, b{ 3.0 };
    DA< 4 > r = a + ( -b );
    EXPECT_NEAR( r.value(), 4.0, kTol );
}

TEST( NegAddTo, NegLeaf_SubtractedFrom_Leaf )
{
    DA< 4 > a{ 3.0 }, b{ 2.0 };
    DA< 4 > r = a - ( -b );  // a + b
    EXPECT_NEAR( r.value(), 5.0, kTol );
}

TEST( NegAddTo, NegLeaf_MatchesDirect )
{
    auto x = DA< 5 >::variable< 0 >( { 2.0 } );
    DA< 5 > r1 = x + ( -x );
    DA< 5 > r2 = x - x;
    ExpectCoeffsNear( r1, r2 );
}

TEST( NegAddTo, NegNode_AddTo )
{
    auto x = DA< 4 >::variable< 0 >( { 1.0 } );
    DA< 4 > two{ 2.0 }, three{ 3.0 };
    DA< 4 > r1 = three + ( -( x + two ) );
    DA< 4 > r2 = three - ( x + two );
    ExpectCoeffsNear( r1, r2 );
}

TEST( NegAddTo, DoubleNeg )
{
    auto x = DA< 4 >::variable< 0 >( { 3.0 } );
    DA< 4 > r = -( -x );
    ExpectCoeffsNear( r, x );
}

TEST( NegAddTo, MulOfNegs )
{
    auto x = DA< 4 >::variable< 0 >( { 2.0 } );
    auto y = DA< 4 >::variable< 0 >( { 0.0 } );
    y[0] = 3.0;
    DA< 4 > r1 = ( -x ) * ( -y );
    DA< 4 > r2 = x * y;
    ExpectCoeffsNear( r1, r2 );
}

TEST( NegAddTo, NegBivariate )
{
    auto [x, y] = DAn< 2, 2 >::variables( { 1.0, 2.0 } );
    DAn< 2, 2 > r1 = -( x + y );
    DAn< 2, 2 > r2 = ( -x ) + ( -y );
    ExpectCoeffsNear( r1, r2 );
}

TEST( NegAddTo, NegValue )
{
    auto x = DA< 3 >::variable< 0 >( { 5.0 } );
    DA< 3 > r = -x;
    EXPECT_NEAR( r[0], -5.0, kTol );
    EXPECT_NEAR( r[1], -1.0, kTol );
}

// =============================================================================
// ScalarExpr — DA op scalar
// =============================================================================

TEST( ScalarExpr, AddScalarRight )
{
    auto x = DA< 3 >::variable< 0 >( { 1.0 } );
    DA< 3 > r = x + 5.0;
    EXPECT_NEAR( r[0], 6.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
}

TEST( ScalarExpr, SubScalarRight )
{
    auto x = DA< 3 >::variable< 0 >( { 3.0 } );
    DA< 3 > r = x - 1.0;
    EXPECT_NEAR( r[0], 2.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
}

TEST( ScalarExpr, MulScalarRight )
{
    auto x = DA< 3 >::variable< 0 >( { 2.0 } );
    DA< 3 > r = x * 4.0;
    EXPECT_NEAR( r[0], 8.0, kTol );
    EXPECT_NEAR( r[1], 4.0, kTol );
}

TEST( ScalarExpr, DivScalarRight )
{
    auto x = DA< 3 >::variable< 0 >( { 4.0 } );
    DA< 3 > r = x / 2.0;
    EXPECT_NEAR( r[0], 2.0, kTol );
    EXPECT_NEAR( r[1], 0.5, kTol );
}

TEST( ScalarExpr, AddScalarLeft )
{
    auto x = DA< 3 >::variable< 0 >( { 1.0 } );
    DA< 3 > r = 10.0 + x;
    EXPECT_NEAR( r[0], 11.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
}

TEST( ScalarExpr, SubScalarLeft )
{
    // s - e: negate e, then add s to constant term
    auto x = DA< 3 >::variable< 0 >( { 1.0 } );
    DA< 3 > r = 5.0 - x;
    EXPECT_NEAR( r[0], 4.0, kTol );   // 5 - 1
    EXPECT_NEAR( r[1], -1.0, kTol );  // -(1)
}

TEST( ScalarExpr, MulScalarLeft )
{
    auto x = DA< 3 >::variable< 0 >( { 2.0 } );
    DA< 3 > r1 = 3.0 * x;
    DA< 3 > r2 = x * 3.0;
    ExpectCoeffsNear( r1, r2 );
}

TEST( ScalarExpr, AddScalar_MatchesManual )
{
    // (x + 2) at x0=1 should give [3, 1, 0, ...]
    auto x = DA< 4 >::variable< 0 >( { 1.0 } );
    DA< 4 > r = x + 2.0;
    DA< 4 > expected{};
    expected[0] = 3.0;
    expected[1] = 1.0;
    ExpectCoeffsNear( r, expected );
}

TEST( ScalarExpr, ScalarChain )
{
    // (x * 2 + 3 - 1) at x0=0: constant = 2, coeff[1] = 2
    auto x = DA< 3 >::variable< 0 >( { 0.0 } );
    DA< 3 > r = x * 2.0 + 3.0 - 1.0;
    EXPECT_NEAR( r[0], 2.0, kTol );
    EXPECT_NEAR( r[1], 2.0, kTol );
}

// =============================================================================
// ScalarDivLExpr — s / DA
// =============================================================================

TEST( ScalarDivL, LeafDivisor_Constant )
{
    DA< 4 > c{ 4.0 };
    DA< 4 > r = 1.0 / c;
    EXPECT_NEAR( r.value(), 0.25, kTol );
    for ( std::size_t k = 1; k < DA< 4 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( ScalarDivL, LeafDivisor_Linear )
{
    // 2/(1+x): c[k] = 2*(-1)^k
    DA< 4 > one_plus_x{};
    one_plus_x[0] = 1.0;
    one_plus_x[1] = 1.0;
    DA< 4 > r = 2.0 / one_plus_x;
    EXPECT_NEAR( r[0], 2.0, kTol );
    EXPECT_NEAR( r[1], -2.0, kTol );
    EXPECT_NEAR( r[2], 2.0, kTol );
    EXPECT_NEAR( r[3], -2.0, kTol );
    EXPECT_NEAR( r[4], 2.0, kTol );
}

TEST( ScalarDivL, LeafDivisor_MatchesReciprocal )
{
    auto x = DA< 5 >::variable< 0 >( { 2.0 } );
    DA< 5 > r1 = 1.0 / x;
    DA< 5 > r2 = DA< 5 >{ 1.0 } / x;
    ExpectCoeffsNear( r1, r2 );
}

TEST( ScalarDivL, NodeDivisor_MatchesLeaf )
{
    auto x = DA< 4 >::variable< 0 >( { 2.0 } );
    DA< 4 > c{ 1.0 };
    DA< 4 > sum = x + c;
    DA< 4 > r1 = 1.0 / sum;
    DA< 4 > r2 = 1.0 / ( x + c );
    ExpectCoeffsNear( r1, r2 );
}

TEST( ScalarDivL, ScalarTimesReciprocal )
{
    auto x = DA< 4 >::variable< 0 >( { 2.0 } );
    DA< 4 > r1 = 3.0 / x;
    DA< 4 > r2 = 3.0 * ( 1.0 / x );
    ExpectCoeffsNear( r1, r2 );
}

TEST( ScalarDivL, Bivariate_Leaf )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 2.0 } );
    DAn< 3, 2 > r = 6.0 / x;
    EXPECT_NEAR( r.value(), 6.0, kTol );
    EXPECT_NEAR( r.derivative( { 1, 0 } ), -6.0, kTol );  // d(6/x)/dx|_{x=1} = -6
}

TEST( ScalarDivL, ProductWithOriginal_IsScalar )
{
    // (s/x)*x should equal s (as constant DA)
    auto x = DA< 4 >::variable< 0 >( { 3.0 } );
    DA< 4 > r = ( 2.0 / x ) * x;
    EXPECT_NEAR( r[0], 2.0, kTol );
    for ( std::size_t k = 1; k < DA< 4 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol ) << "k=" << k;
}
