#include "testUtils.hpp"

// =============================================================================
// fma(x, y, z) = x * y + z
// =============================================================================

TEST( FMA, Constants )
{
    DA< 4 > a{ 2.0 }, b{ 3.0 }, c{ 5.0 };
    DA< 4 > r = fma( a, b, c );  // 2*3 + 5 = 11
    EXPECT_NEAR( r.value(), 11.0, kTol );
    for ( std::size_t k = 1; k < DA< 4 >::ncoef; ++k )
        EXPECT_NEAR( r[k], 0.0, kTol ) << "k=" << k;
}

TEST( FMA, VariableTimesConstantPlusConstant )
{
    // fma(x, 3, 5) = 3x + 5, at x0=2: value=11, dx=3
    auto x = DA< 4 >::variable< 0 >( { 2.0 } );
    DA< 4 > three{ 3.0 }, five{ 5.0 };
    DA< 4 > r = fma( x, three, five );
    EXPECT_NEAR( r[0], 11.0, kTol );
    EXPECT_NEAR( r[1], 3.0, kTol );
}

TEST( FMA, TwoVariables )
{
    // fma(x, x, 0) = x^2, at x0=3: value=9, dx=6, d2x=1
    auto x = DA< 4 >::variable< 0 >( { 3.0 } );
    DA< 4 > zero{ 0.0 };
    DA< 4 > r = fma( x, x, zero );
    DA< 4 > ref = x * x;
    ExpectCoeffsNear( r, ref );
}

TEST( FMA, MatchesManualMulAdd )
{
    auto x = DA< 5 >::variable< 0 >( { 1.0 } );
    DA< 5 > a{ 2.0 }, b{ 3.0 };
    DA< 5 > r1 = fma( x, a, b );
    DA< 5 > r2 = x * a + b;
    ExpectCoeffsNear( r1, r2 );
}

TEST( FMA, AllVariables )
{
    // fma(x, x, x) = x^2 + x, at x0=2: value=6, dx=5, d2x=1
    auto x = DA< 4 >::variable< 0 >( { 2.0 } );
    DA< 4 > r = fma( x, x, x );
    DA< 4 > ref = x * x + x;
    ExpectCoeffsNear( r, ref );
}

TEST( FMA, Bivariate )
{
    auto [x, y] = DAn< 3, 2 >::variables( { 1.0, 2.0 } );
    // fma(x, y, x) = x*y + x at (1,2): value=4, dx=3, dy=1, dxdy=1
    DAn< 3, 2 > r = fma( x, y, x );
    DAn< 3, 2 > ref = x * y + x;
    ExpectCoeffsNear( r, ref );
}

TEST( FMA, WithExpressions )
{
    // fma on non-leaf expressions: fma(x+1, x+2, x) vs (x+1)*(x+2)+x
    auto x = DA< 4 >::variable< 0 >( { 0.0 } );
    DA< 4 > one{ 1.0 }, two{ 2.0 };
    DA< 4 > r = fma( x + one, x + two, x );
    DA< 4 > ref = ( x + one ) * ( x + two ) + x;
    ExpectCoeffsNear( r, ref );
}
