#include "testUtils.hpp"

// =============================================================================
// BinExpr<OpAdd> — basic correctness
// =============================================================================

TEST( AddTo, TwoLeaves_Sum )
{
    DA< 4 > a{ 2.0 }, b{ 3.0 };
    DA< 4 > r = a + b;
    EXPECT_NEAR( r.value(), 5.0, kTol );
}

TEST( AddTo, FourLeaves_Sum )
{
    DA< 4 > a{ 1.0 }, b{ 2.0 }, c{ 3.0 }, d{ 4.0 };
    DA< 4 > r = a + b + c + d;
    EXPECT_NEAR( r.value(), 10.0, kTol );
}

TEST( AddTo, LeafSum_MatchesManual )
{
    auto x = DA< 5 >::variable< 0 >( { 1.0 } );
    auto y = DA< 5 >::variable< 0 >( { 0.0 } );
    y[0] = 2.0;
    y[1] = 3.0;
    DA< 5 > r1 = x + y;
    DA< 5 > r2 = y + x;
    ExpectCoeffsNear( r1, r2 );
}

TEST( AddTo, LinearExpansion )
{
    // x at x0=3: result is 3 + 1*δ. x+x should be 6 + 2δ
    auto x = DA< 4 >::variable< 0 >( { 3.0 } );
    DA< 4 > r = x + x;
    EXPECT_NEAR( r[0], 6.0, kTol );
    EXPECT_NEAR( r[1], 2.0, kTol );
    for ( std::size_t k = 2; k < DA< 4 >::ncoef; ++k ) EXPECT_EQ( r[k], 0.0 );
}

TEST( AddTo, MulLeaves_Plus_MulLeaves )
{
    auto x = DA< 4 >::variable< 0 >( { 1.0 } );
    DA< 4 > two{ 2.0 }, three{ 3.0 };
    DA< 4 > r = x * two + x * three;  // x*2 + x*3 = 5x = 5 + 5δ
    EXPECT_NEAR( r[0], 5.0, kTol );
    EXPECT_NEAR( r[1], 5.0, kTol );
}

TEST( AddTo, MulLeaves_Plus_MulLeaves_vs_Reference )
{
    auto x = DA< 5 >::variable< 0 >( { 2.0 } );
    auto y = DA< 5 >::variable< 0 >( { 0.0 } );
    y[0] = 3.0;
    DA< 5 > ref = ( x * x ) + ( y * y );
    DA< 5 > r = x * x + y * y;
    ExpectCoeffsNear( r, ref );
}

TEST( AddTo, Correctness_Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 0.0, 0.0 } );
    DAn< 3, 2 > r = x * x + y * y;
    EXPECT_NEAR( r.coeff( { 2, 0 } ), 1.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 2 } ), 1.0, kTol );
    EXPECT_NEAR( r.coeff( { 1, 1 } ), 0.0, kTol );
}

TEST( AddTo, LongAddChain_Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 2.0 } );
    DAn< 3, 2 > r = x + y + x + y;
    DAn< 3, 2 > ref = ( x + x ) + ( y + y );
    ExpectCoeffsNear( r, ref );
}

TEST( AddTo, SumExpr_FiveOperands )
{
    // a+b+c+d+e should produce SumExpr<A,B,C,D,E> and compute correctly
    DA< 2 > a{ 1.0 }, b{ 2.0 }, c{ 3.0 }, d{ 4.0 }, e{ 5.0 };
    DA< 2 > r = a + b + c + d + e;
    EXPECT_NEAR( r.value(), 15.0, kTol );
}

TEST( AddTo, SumExpr_WithVariables )
{
    // x + x + x at x0=2: value=6, coeff[1]=3
    auto x = DA< 3 >::variable< 0 >( { 2.0 } );
    DA< 3 > r = x + x + x;
    EXPECT_NEAR( r[0], 6.0, kTol );
    EXPECT_NEAR( r[1], 3.0, kTol );
}

TEST( AddTo, SumExpr_SubtractCancel )
{
    // (a+b) - (a+b) = 0 via subtraction chain
    auto x = DA< 3 >::variable< 0 >( { 1.0 } );
    DA< 3 > two{ 2.0 };
    DA< 3 > r = ( x + two ) - ( x + two );
    for ( std::size_t k = 0; k < DA< 3 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol ) << "k=" << k;
}

// =============================================================================
// BinExpr<OpSub>
// =============================================================================

TEST( SubTo, TwoLeaves_Diff )
{
    DA< 4 > a{ 5.0 }, b{ 3.0 };
    DA< 4 > r = a - b;
    EXPECT_NEAR( r.value(), 2.0, kTol );
}

TEST( SubTo, MulLeaves_Minus_MulLeaves )
{
    auto x = DA< 4 >::variable< 0 >( { 1.0 } );
    DA< 4 > two{ 2.0 }, one{ 1.0 };
    DA< 4 > r = x * two - x * one;  // x*(2-1) = x = 1+δ
    EXPECT_NEAR( r[0], 1.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
}

TEST( SubTo, MixedAddSubChain )
{
    auto x = DA< 4 >::variable< 0 >( { 0.0 } );
    DA< 4 > one{ 1.0 }, two{ 2.0 }, three{ 3.0 };
    DA< 4 > r = x + one - two + three;  // x + 2
    EXPECT_NEAR( r[0], 2.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
}

TEST( SubTo, SubtractSelf_IsZero )
{
    auto x = DA< 5 >::variable< 0 >( { 3.0 } );
    DA< 5 > r = x - x;
    for ( std::size_t k = 0; k < DA< 5 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol ) << "k=" << k;
}

TEST( SubTo, Bivariate_Subtraction )
{
    auto [x, y] = DAn< 2, 2 >::variables( { 1.0, 2.0 } );
    DAn< 2, 2 > r = x - y;  // (1-2) + δx - δy
    EXPECT_NEAR( r.coeff( { 0, 0 } ), -1.0, kTol );
    EXPECT_NEAR( r.coeff( { 1, 0 } ), 1.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 1 } ), -1.0, kTol );
}

// =============================================================================
// SumExpr — variadic flattening
// =============================================================================

TEST( SumExpr, LeftAssocParenthesisation )
{
    // ((a+b)+c)+d and a+(b+(c+d)) should give the same result
    DA< 2 > a{ 1.0 }, b{ 2.0 }, c{ 3.0 }, d{ 4.0 };
    DA< 2 > left = ( ( a + b ) + c ) + d;
    DA< 2 > right = a + ( b + ( c + d ) );
    ExpectCoeffsNear( left, right );
}

TEST( SumExpr, MixedLeafAndNode )
{
    // (a+b) + c where a,b,c are leaves: SumExpr<A,B,C>
    auto x = DA< 3 >::variable< 0 >( { 1.0 } );
    DA< 3 > c{ 5.0 };
    DA< 3 > r = x + x + c;
    EXPECT_NEAR( r[0], 7.0, kTol );  // 1 + 1 + 5
    EXPECT_NEAR( r[1], 2.0, kTol );  // 2 from two x
}

TEST( SumExpr, AddTo_ZeroTemps_AllLeaves )
{
    // SumExpr.addTo should work correctly when used in a larger expression
    DA< 3 > a{ 1.0 }, b{ 2.0 }, c{ 3.0 };
    DA< 3 > d{ 10.0 };
    // (a+b+c) gets used as right operand of another add
    DA< 3 > r = d + ( a + b + c );
    EXPECT_NEAR( r.value(), 16.0, kTol );
}
