#include "testUtils.hpp"

// =============================================================================
// Aliasing safety: operator+= / operator-= must not corrupt data
// =============================================================================

TEST( Aliasing, PlusEqSelf_IsDoubling )
{
    auto x = TE< 4 >::variable< 0 >( { 3.0 } );
    TE< 4 > r = x;
    r += r;
    EXPECT_NEAR( r[0], 6.0, kTol );
    EXPECT_NEAR( r[1], 2.0, kTol );
}

TEST( Aliasing, MinusEqSelf_IsZero )
{
    auto x = TE< 4 >::variable< 0 >( { 3.0 } );
    TE< 4 > r = x;
    r -= r;
    for ( std::size_t k = 0; k < TE< 4 >::ncoef; ++k ) EXPECT_NEAR( r[k], 0.0, kTol );
}

TEST( Aliasing, PlusEqExpression_Safe )
{
    auto x = TE< 4 >::variable< 0 >( { 1.0 } );
    TE< 4 > r = x;
    r += x * x;                      // r = x + x^2
    EXPECT_NEAR( r[0], 2.0, kTol );  // (1+δ) + (1+δ)^2 at δ=0: 1+1
    EXPECT_NEAR( r[1], 3.0, kTol );  // 1 + 2
}

TEST( Aliasing, MinusEqExpression_Safe )
{
    auto x = TE< 4 >::variable< 0 >( { 1.0 } );
    TE< 4 > r = x * x;
    r -= x;
    EXPECT_NEAR( r[0], 0.0, kTol );  // 1 - 1
    EXPECT_NEAR( r[1], 1.0, kTol );  // 2 - 1
}

// =============================================================================
// Composed expressions — multiple operations
// =============================================================================

TEST( Compose, SumOfProducts_AllLeaves )
{
    // x^2 + y^2 + z^2 at origin: diagonal Hessian = 2I
    auto [x, y, z] = TEn< 2, 3 >::variables( { 0.0, 0.0, 0.0 } );
    TEn< 2, 3 > r = x * x + y * y + z * z;
    EXPECT_NEAR( r.coeff( { 2, 0, 0 } ), 1.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 2, 0 } ), 1.0, kTol );
    EXPECT_NEAR( r.coeff( { 0, 0, 2 } ), 1.0, kTol );
    EXPECT_NEAR( r.coeff( { 1, 1, 0 } ), 0.0, kTol );
}

TEST( Compose, NegatedProductInSum )
{
    // x*y - z*z: uses subTo for the right Mul
    auto [x, y, z] = TEn< 2, 3 >::variables( { 1.0, 1.0, 1.0 } );
    TEn< 2, 3 > r = x * y - z * z;
    EXPECT_NEAR( r.value(), 0.0, kTol );
    EXPECT_NEAR( r.derivative( { 1, 0, 0 } ), 1.0, kTol );
    EXPECT_NEAR( r.derivative( { 0, 0, 1 } ), -2.0, kTol );
}

TEST( Compose, AddChain_WithNeg )
{
    auto [x, y, z] = TEn< 2, 3 >::variables( { 1.0, 2.0, 3.0 } );
    TEn< 2, 3 > w{};
    w[0] = 4.0;
    TEn< 2, 3 > r1 = x + y - z + ( -w );
    TEn< 2, 3 > r2 = x + y - z - w;
    ExpectCoeffsNear( r1, r2 );
}

TEST( Compose, DivByLeaf_InLargerExpr )
{
    auto x = TE< 4 >::variable< 0 >( { 2.0 } );
    TE< 4 > y{ 4.0 };
    TE< 4 > r = x + 1.0 / y;
    EXPECT_NEAR( r[0], 2.25, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
}

// =============================================================================
// Gradient and Hessian verification
// =============================================================================

TEST( Compose, GradientOfQuadratic )
{
    // f(x,y) = x^2 + x*y + y^2 at (1,2)
    // ∂f/∂x = 2x + y = 4,  ∂f/∂y = x + 2y = 5
    auto [x, y] = TEn< 2, 2 >::variables( { 1.0, 2.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;
    EXPECT_NEAR( f.value(), 7.0, kTol );  // 1+2+4
    EXPECT_NEAR( f.derivative( { 1, 0 } ), 4.0, kTol );
    EXPECT_NEAR( f.derivative( { 0, 1 } ), 5.0, kTol );
}

TEST( Compose, HessianOfQuadratic )
{
    // f(x,y) = x^2 + x*y + y^2
    // H = [[2,1],[1,2]]
    auto [x, y] = TEn< 2, 2 >::variables( { 1.0, 2.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;
    EXPECT_NEAR( f.derivative( { 2, 0 } ), 2.0, kTol );
    EXPECT_NEAR( f.derivative( { 1, 1 } ), 1.0, kTol );
    EXPECT_NEAR( f.derivative( { 0, 2 } ), 2.0, kTol );
}

TEST( Compose, CubicDerivatives )
{
    // f(x) = x^3 at x0=2
    // f(2)=8, f'(2)=12, f''(2)=12, f'''(2)=6
    auto x = TE< 3 >::variable< 0 >( { 2.0 } );
    TE< 3 > f = cube( x );
    EXPECT_NEAR( f.value(), 8.0, kTol );
    EXPECT_NEAR( f.derivative( { 1 } ), 12.0, kTol );
    EXPECT_NEAR( f.derivative( { 2 } ), 12.0, kTol );
    EXPECT_NEAR( f.derivative( { 3 } ), 6.0, kTol );
}

// =============================================================================
// Mixed chains of different expression types
// =============================================================================

TEST( Compose, MulPlusDivMinusSqrt )
{
    // f(x) = x*x + 1/x - sqrt(x) at x0=4
    // f(4) = 16 + 0.25 - 2 = 14.25
    auto x = TE< 3 >::variable< 0 >( { 4.0 } );
    TE< 3 > f = x * x + 1.0 / x - sqrt( x );
    EXPECT_NEAR( f.value(), 14.25, kTol );
}

TEST( Compose, ScalarArithmeticChain )
{
    // (2*x + 3) * (x - 1) at x0=2
    // = (4+3)*(2-1) = 7
    auto x = TE< 4 >::variable< 0 >( { 2.0 } );
    TE< 4 > f = ( 2.0 * x + 3.0 ) * ( x - 1.0 );
    EXPECT_NEAR( f.value(), 7.0, kTol );
    // f'(x) = 2*(x-1) + (2x+3) = 4x+1 → f'(2) = 9
    EXPECT_NEAR( f.derivative( { 1 } ), 9.0, kTol );
}

TEST( Compose, NestedSquareAndCube )
{
    // square(cube(x)) vs x^6 (via x*x*x*x*x*x)
    auto x = TE< 6 >::variable< 0 >( { 1.0 } );
    TE< 6 > r1 = square( cube( x ) );
    TE< 6 > r2 = x * x * x * x * x * x;
    ExpectCoeffsNear( r1, r2 );
}

TEST( Compose, SqrtOfProduct )
{
    // sqrt(x*y) at (4,9) = 6
    auto [x, y] = TEn< 2, 2 >::variables( { 4.0, 9.0 } );
    TEn< 2, 2 > r = sqrt( x * y );
    EXPECT_NEAR( r.value(), 6.0, kTol );
}

TEST( Compose, NegatedDivision )
{
    // -(a/b) should equal (-a)/b
    auto x = TE< 4 >::variable< 0 >( { 2.0 } );
    TE< 4 > y{ 3.0 };
    TE< 4 > r1 = -( x / y );
    TE< 4 > r2 = ( -x ) / y;
    ExpectCoeffsNear( r1, r2 );
}

TEST( Compose, TrivariateMixedExpr )
{
    // f(x,y,z) = x*y + y*z + z*x at (1,2,3)
    // value = 2+6+3 = 11
    auto [x, y, z] = TEn< 2, 3 >::variables( { 1.0, 2.0, 3.0 } );
    TEn< 2, 3 > f = x * y + y * z + z * x;
    EXPECT_NEAR( f.value(), 11.0, kTol );
    // ∂f/∂x = y + z = 5
    EXPECT_NEAR( f.derivative( { 1, 0, 0 } ), 5.0, kTol );
    // ∂f/∂y = x + z = 4
    EXPECT_NEAR( f.derivative( { 0, 1, 0 } ), 4.0, kTol );
    // ∂f/∂z = y + x = 3
    EXPECT_NEAR( f.derivative( { 0, 0, 1 } ), 3.0, kTol );
}
