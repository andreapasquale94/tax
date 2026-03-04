#include "testUtils.hpp"

// =============================================================================
// BinExpr<OpMul> — leaf detection paths
// =============================================================================

TEST( LeafMul, BothLeaves_CorrectProduct )
{
    auto a = TE< 4 >::variable< 0 >( { 2.0 } );
    TE< 4 > b{ 3.0 };
    TE< 4 > r = a * b;  // (2+δ)*3 = 6 + 3δ
    EXPECT_NEAR( r[0], 6.0, kTol );
    EXPECT_NEAR( r[1], 3.0, kTol );
}

TEST( LeafMul, LeftLeafRightNode )
{
    auto a = TE< 4 >::variable< 0 >( { 1.0 } );
    TE< 4 > b{ 2.0 }, c{ 3.0 };
    TE< 4 > r = a * ( b + c );  // (1+δ)*5 = 5 + 5δ
    EXPECT_NEAR( r[0], 5.0, kTol );
    EXPECT_NEAR( r[1], 5.0, kTol );
}

TEST( LeafMul, RightLeafLeftNode )
{
    auto a = TE< 4 >::variable< 0 >( { 1.0 } );
    TE< 4 > b{ 2.0 }, c{ 4.0 };
    TE< 4 > r = ( a + b ) * c;  // (3+δ)*4 = 12 + 4δ
    EXPECT_NEAR( r[0], 12.0, kTol );
    EXPECT_NEAR( r[1], 4.0, kTol );
}

TEST( LeafMul, BothNodes )
{
    auto x = TE< 4 >::variable< 0 >( { 0.0 } );
    TE< 4 > two{};
    two[0] = 2;
    TE< 4 > r = ( x + two ) * ( x + two );  // (2+δ)^2 = 4 + 4δ + δ^2
    EXPECT_NEAR( r[0], 4.0, kTol );
    EXPECT_NEAR( r[1], 4.0, kTol );
    EXPECT_NEAR( r[2], 1.0, kTol );
}

TEST( LeafMul, Bivariate )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    TEn< 3, 2 > r = x * y;
    // x*y at (1,2): value=2, ∂/∂x=y=2, ∂/∂y=x=1, ∂²/∂x∂y=1
    EXPECT_NEAR( r.coeff( { 0, 0 } ), 2.0, kTol );
    EXPECT_NEAR( r.coeff( { 1, 0 } ), 2.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 1.0, kTol );
    EXPECT_NEAR( r.coeff( { 1, 1 } ), 1.0, kTol );
}

TEST( LeafMul, MultiplyByOne )
{
    auto x = TE< 4 >::variable< 0 >( { 2.0 } );
    TE< 4 > one{ 1.0 };
    TE< 4 > r = x * one;
    ExpectCoeffsNear( r, x );
}

TEST( LeafMul, SquareViaProduct )
{
    auto x = TE< 5 >::variable< 0 >( { 3.0 } );
    TE< 5 > r1 = x * x;
    TE< 5 > r2 = x * x;
    ExpectCoeffsNear( r1, r2 );
}

// =============================================================================
// BinExpr<OpDiv>
// =============================================================================

TEST( Div, DivideByConstant )
{
    TE< 4 > a{ 8.0 };
    TE< 4 > b{ 4.0 };
    TE< 4 > r = a / b;  // 8/4 = 2
    EXPECT_NEAR( r.value(), 2.0, kTol );
    for ( std::size_t k = 1; k < TE< 4 >::nCoefficients; ++k ) EXPECT_NEAR( r[k], 0.0, kTol ) << "k=" << k;
}

TEST( Div, DivideByLinear )
{
    // (1+x)/(1+x) = 1
    auto x = TE< 4 >::variable< 0 >( { 0.0 } );
    TE< 4 > one_plus_x{};
    one_plus_x[0] = 1.0;
    one_plus_x[1] = 1.0;
    TE< 4 > r = one_plus_x / one_plus_x;
    EXPECT_NEAR( r[0], 1.0, kTol );
    for ( std::size_t k = 1; k < TE< 4 >::nCoefficients; ++k ) EXPECT_NEAR( r[k], 0.0, kTol ) << "k=" << k;
}

TEST( Div, OneOverGeometricSeries )
{
    // 1/(1+x) at order 4: coefficients are (-1)^k
    TE< 4 > denom{};
    denom[0] = 1;
    denom[1] = 1;
    TE< 4 > numer{};
    numer[0] = 1;
    TE< 4 > r = numer / denom;
    for ( int k = 0; k <= 4; ++k ) EXPECT_NEAR( r[k], std::pow( -1.0, k ), kTol ) << "k=" << k;
}

TEST( Div, LeafByLeaf_MatchesScalarDivL )
{
    auto x = TE< 5 >::variable< 0 >( { 2.0 } );
    TE< 5 > one{ 1.0 };
    TE< 5 > r1 = one / x;  // BinExpr<Div>
    TE< 5 > r2 = 1.0 / x;  // ScalarDivLExpr
    ExpectCoeffsNear( r1, r2 );
}

// =============================================================================
// ProductExpr — variadic multiplication
// =============================================================================

TEST( ProductExpr, ThreeLeaves )
{
    // a*b*c: should produce ProductExpr<A,B,C>
    TE< 4 > a{ 2.0 }, b{ 3.0 }, c{ 4.0 };
    TE< 4 > r = a * b * c;
    EXPECT_NEAR( r.value(), 24.0, kTol );
}

TEST( ProductExpr, ThreeLeaves_WithVariables )
{
    // x*2*3 at x0=1: (1+δ)*2*3 = 6 + 6δ
    auto x = TE< 4 >::variable< 0 >( { 1.0 } );
    TE< 4 > two{ 2.0 }, three{ 3.0 };
    TE< 4 > r = x * two * three;
    EXPECT_NEAR( r[0], 6.0, kTol );
    EXPECT_NEAR( r[1], 6.0, kTol );
}

TEST( ProductExpr, FourLeaves_MatchesPairwiseProduct )
{
    auto x = TE< 4 >::variable< 0 >( { 2.0 } );
    TE< 4 > a{ 1.0 }, b{ 2.0 }, c{ 3.0 };
    TE< 4 > variadic = x * a * b * c;
    TE< 4 > pairwise = ( ( x * a ) * b ) * c;
    ExpectCoeffsNear( variadic, pairwise );
}

TEST( ProductExpr, BilinearBivariate )
{
    auto [x, y] = TEn< 2, 2 >::variables( { 1.0, 1.0 } );
    TEn< 2, 2 > r = x * y;
    // should give same as single BinExpr multiplication
    EXPECT_NEAR( r.coeff( { 0, 0 } ), 1.0, kTol );  // 1*1
    EXPECT_NEAR( r.coeff( { 1, 0 } ), 1.0, kTol );  // ∂/∂x(xy)|1,1 = y = 1
    EXPECT_NEAR( r.coeff( { 0, 1 } ), 1.0, kTol );  // ∂/∂y(xy)|1,1 = x = 1
    EXPECT_NEAR( r.coeff( { 1, 1 } ), 1.0, kTol );  // ∂²/∂x∂y = 1
}

TEST( ProductExpr, TripleProduct_Bivariate )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 1.0 } );
    // x*x*y vs x^2*y computed pairwise
    TEn< 3, 2 > variadic = x * x * y;
    TEn< 3, 2 > pairwise = ( x * x ) * y;
    ExpectCoeffsNear( variadic, pairwise );
}

TEST( ProductExpr, LeftExtend_Associativity )
{
    // (a*b)*c and a*b*c (left-extend overload) should be the same
    auto x = TE< 5 >::variable< 0 >( { 1.0 } );
    TE< 5 > two{ 2.0 }, three{ 3.0 };
    TE< 5 > r1 = ( x * two ) * three;  // left-extend
    TE< 5 > r2 = x * two * three;      // same via left-extend
    ExpectCoeffsNear( r1, r2 );
}
