#include "testUtils.hpp"

// =============================================================================
// deriv() – univariate
// =============================================================================

TEST( TTEDeriv, Univariate_Constant_IsZero )
{
    // d/dx (5) = 0
    TE< 3 > f{ 5.0 };
    auto df = f.deriv< 0 >();
    for ( std::size_t k = 0; k < TE< 3 >::nCoefficients; ++k )
        EXPECT_NEAR( df[k], 0.0, kTol ) << "k=" << k;
}

TEST( TTEDeriv, Univariate_Linear )
{
    // f = 3 + 2*dx  →  df/dx = 2
    TE< 3 >::Data c{ 3.0, 2.0, 0.0, 0.0 };
    TE< 3 > f{ c };
    auto df = f.deriv< 0 >();
    EXPECT_NEAR( df[0], 2.0, kTol );
    for ( std::size_t k = 1; k < TE< 3 >::nCoefficients; ++k )
        EXPECT_NEAR( df[k], 0.0, kTol ) << "k=" << k;
}

TEST( TTEDeriv, Univariate_Polynomial )
{
    // f = 1 + 2*dx + 3*dx^2 + 4*dx^3  →  df/dx = 2 + 6*dx + 12*dx^2
    TE< 3 >::Data c{ 1.0, 2.0, 3.0, 4.0 };
    TE< 3 > f{ c };
    auto df = f.deriv< 0 >();
    EXPECT_NEAR( df[0], 2.0, kTol );
    EXPECT_NEAR( df[1], 6.0, kTol );
    EXPECT_NEAR( df[2], 12.0, kTol );
}

TEST( TTEDeriv, Univariate_Variable )
{
    // x = x0 + 1*dx  →  d/dx x = 1
    auto x = TE< 4 >::variable( 3.0 );
    auto dx = x.deriv< 0 >();
    EXPECT_NEAR( dx[0], 1.0, kTol );
    for ( std::size_t k = 1; k < TE< 4 >::nCoefficients; ++k )
        EXPECT_NEAR( dx[k], 0.0, kTol ) << "k=" << k;
}

TEST( TTEDeriv, Univariate_RuntimeIndex )
{
    // Same as compile-time version
    TE< 3 >::Data c{ 1.0, 2.0, 3.0, 4.0 };
    TE< 3 > f{ c };
    auto df_ct = f.deriv< 0 >();
    auto df_rt = f.deriv( 0 );
    ExpectCoeffsNear( df_ct, df_rt );
}

TEST( TTEDeriv, Univariate_RuntimeIndex_OutOfRange )
{
    TE< 3 > f{ 1.0 };
    EXPECT_THROW( f.deriv( 1 ), std::out_of_range );
    EXPECT_THROW( f.deriv( -1 ), std::out_of_range );
}

// =============================================================================
// deriv() – multivariate
// =============================================================================

TEST( TTEDeriv, Bivariate_wrtX )
{
    // f = x^2 + x*y + y^2  (expanded at origin)
    // df/dx = 2*x + y
    auto [x, y] = TEn< 2, 2 >::variables( { 0.0, 0.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;
    auto dfdx = f.deriv< 0 >();

    // Coefficients: const=0, x-coeff=2, y-coeff=1, x^2=0, xy=0, y^2=0
    EXPECT_NEAR( dfdx.coeff( { 0, 0 } ), 0.0, kTol );
    EXPECT_NEAR( dfdx.coeff( { 1, 0 } ), 2.0, kTol );
    EXPECT_NEAR( dfdx.coeff( { 0, 1 } ), 1.0, kTol );
    // All degree-2 terms zero
    EXPECT_NEAR( dfdx.coeff( { 2, 0 } ), 0.0, kTol );
    EXPECT_NEAR( dfdx.coeff( { 1, 1 } ), 0.0, kTol );
    EXPECT_NEAR( dfdx.coeff( { 0, 2 } ), 0.0, kTol );
}

TEST( TTEDeriv, Bivariate_wrtY )
{
    // f = x^2 + x*y + y^2
    // df/dy = x + 2*y
    auto [x, y] = TEn< 2, 2 >::variables( { 0.0, 0.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;
    auto dfdy = f.deriv< 1 >();

    EXPECT_NEAR( dfdy.coeff( { 0, 0 } ), 0.0, kTol );
    EXPECT_NEAR( dfdy.coeff( { 1, 0 } ), 1.0, kTol );
    EXPECT_NEAR( dfdy.coeff( { 0, 1 } ), 2.0, kTol );
}

TEST( TTEDeriv, Bivariate_RuntimeIndex )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 0.0, 0.0 } );
    TEn< 3, 2 > f = x * x + x * y + y * y;
    ExpectCoeffsNear( f.deriv< 0 >(), f.deriv( 0 ) );
    ExpectCoeffsNear( f.deriv< 1 >(), f.deriv( 1 ) );
}

TEST( TTEDeriv, Bivariate_DerivOfDeriv )
{
    // f = x^3 + x^2*y + x*y^2 + y^3 (at origin, order 3)
    auto [x, y] = TEn< 3, 2 >::variables( { 0.0, 0.0 } );
    TEn< 3, 2 > f = x * x * x + x * x * y + x * y * y + y * y * y;

    // d^2f/dx^2 = 6*x + 2*y
    auto d2fdx2 = f.deriv< 0 >().deriv< 0 >();
    EXPECT_NEAR( d2fdx2.coeff( { 1, 0 } ), 6.0, kTol );
    EXPECT_NEAR( d2fdx2.coeff( { 0, 1 } ), 2.0, kTol );
}

// =============================================================================
// integ() – univariate
// =============================================================================

TEST( TTEInteg, Univariate_Constant )
{
    // integral of 2 dx = 2*x
    TE< 3 >::Data c{ 2.0, 0.0, 0.0, 0.0 };
    TE< 3 > f{ c };
    auto F = f.integ< 0 >();
    EXPECT_NEAR( F[0], 0.0, kTol );  // constant of integration = 0
    EXPECT_NEAR( F[1], 2.0, kTol );  // 2*x term
    EXPECT_NEAR( F[2], 0.0, kTol );
    EXPECT_NEAR( F[3], 0.0, kTol );
}

TEST( TTEInteg, Univariate_Linear )
{
    // integral of (1 + 2*x) dx = x + x^2
    TE< 3 >::Data c{ 1.0, 2.0, 0.0, 0.0 };
    TE< 3 > f{ c };
    auto F = f.integ< 0 >();
    EXPECT_NEAR( F[0], 0.0, kTol );  // no constant of integration
    EXPECT_NEAR( F[1], 1.0, kTol );  // x
    EXPECT_NEAR( F[2], 1.0, kTol );  // x^2 = 2/2
    EXPECT_NEAR( F[3], 0.0, kTol );
}

TEST( TTEInteg, Univariate_Polynomial )
{
    // integral of (1 + 2*x + 3*x^2) dx = x + x^2 + x^3  (truncated to order 3)
    TE< 3 >::Data c{ 1.0, 2.0, 3.0, 0.0 };
    TE< 3 > f{ c };
    auto F = f.integ< 0 >();
    EXPECT_NEAR( F[0], 0.0, kTol );
    EXPECT_NEAR( F[1], 1.0, kTol );
    EXPECT_NEAR( F[2], 1.0, kTol );
    EXPECT_NEAR( F[3], 1.0, kTol );
}

TEST( TTEInteg, Univariate_TopOrderDropped )
{
    // The highest-order term can't be integrated (would exceed order N)
    TE< 2 >::Data c{ 1.0, 2.0, 3.0 };  // 1 + 2x + 3x^2, order N=2
    TE< 2 > f{ c };
    auto F = f.integ< 0 >();
    EXPECT_NEAR( F[0], 0.0, kTol );
    EXPECT_NEAR( F[1], 1.0, kTol );  // integral of 1 = x
    EXPECT_NEAR( F[2], 1.0, kTol );  // integral of 2x = x^2
    // 3x^2 term is dropped (would give x^3, order 3 > N=2)
}

TEST( TTEInteg, Univariate_RuntimeIndex )
{
    TE< 4 >::Data c{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    TE< 4 > f{ c };
    ExpectCoeffsNear( f.integ< 0 >(), f.integ( 0 ) );
}

TEST( TTEInteg, Univariate_RuntimeIndex_OutOfRange )
{
    TE< 3 > f{ 1.0 };
    EXPECT_THROW( f.integ( 1 ), std::out_of_range );
    EXPECT_THROW( f.integ( -1 ), std::out_of_range );
}

// =============================================================================
// integ() – multivariate
// =============================================================================

TEST( TTEInteg, Bivariate_wrtX )
{
    // f = 1 + x + y  →  integral_x f = x + x^2/2 + x*y
    auto [x, y] = TEn< 2, 2 >::variables( { 0.0, 0.0 } );
    TEn< 2, 2 > f = TEn< 2, 2 >::constant( 1.0 ) + x + y;
    auto Fx = f.integ< 0 >();

    EXPECT_NEAR( Fx.coeff( { 0, 0 } ), 0.0, kTol );  // no const of integration
    EXPECT_NEAR( Fx.coeff( { 1, 0 } ), 1.0, kTol );  // integral(1) w.r.t x = x
    EXPECT_NEAR( Fx.coeff( { 0, 1 } ), 0.0, kTol );  // no y in integral_x of const+x+y? wait
    // integral_x (1 + x + y) = x + x^2/2 + x*y
    EXPECT_NEAR( Fx.coeff( { 2, 0 } ), 0.5, kTol );  // x^2/2
    EXPECT_NEAR( Fx.coeff( { 1, 1 } ), 1.0, kTol );  // x*y
}

TEST( TTEInteg, Bivariate_wrtY )
{
    // f = 1 + x + y  →  integral_y f = y + x*y + y^2/2
    auto [x, y] = TEn< 2, 2 >::variables( { 0.0, 0.0 } );
    TEn< 2, 2 > f = TEn< 2, 2 >::constant( 1.0 ) + x + y;
    auto Fy = f.integ< 1 >();

    EXPECT_NEAR( Fy.coeff( { 0, 0 } ), 0.0, kTol );
    EXPECT_NEAR( Fy.coeff( { 0, 1 } ), 1.0, kTol );  // integral(1) w.r.t y = y
    EXPECT_NEAR( Fy.coeff( { 1, 1 } ), 1.0, kTol );  // x*y
    EXPECT_NEAR( Fy.coeff( { 0, 2 } ), 0.5, kTol );  // y^2/2
}

TEST( TTEInteg, Bivariate_RuntimeIndex )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 0.0, 0.0 } );
    TEn< 3, 2 > f = x * x + x * y + y * y;
    ExpectCoeffsNear( f.integ< 0 >(), f.integ( 0 ) );
    ExpectCoeffsNear( f.integ< 1 >(), f.integ( 1 ) );
}

// =============================================================================
// deriv/integ roundtrip: integ(deriv(f)) recovers f up to constant
// =============================================================================

TEST( TTEDerivInteg, Roundtrip_Univariate )
{
    // For f with f[0] = 0: integ(deriv(f)) == f
    TE< 4 >::Data c{ 0.0, 2.0, 3.0, 4.0, 5.0 };
    TE< 4 > f{ c };
    auto g = f.deriv< 0 >().integ< 0 >();
    // g should recover f (constant term stays 0 since integ sets it to 0)
    for ( std::size_t k = 0; k < TE< 4 >::nCoefficients; ++k )
        EXPECT_NEAR( g[k], f[k], kTol ) << "k=" << k;
}

TEST( TTEDerivInteg, Roundtrip_Univariate_WithConstant )
{
    // integ(deriv(f)) = f - f[0] (constant term lost by deriv, not restored by integ)
    TE< 4 >::Data c{ 7.0, 2.0, 3.0, 4.0, 5.0 };
    TE< 4 > f{ c };
    auto g = f.deriv< 0 >().integ< 0 >();
    EXPECT_NEAR( g[0], 0.0, kTol );  // constant lost
    for ( std::size_t k = 1; k < TE< 4 >::nCoefficients; ++k )
        EXPECT_NEAR( g[k], f[k], kTol ) << "k=" << k;
}

TEST( TTEDerivInteg, Roundtrip_Univariate_DerivOfInteg )
{
    // deriv(integ(f)) == f except top-order term is dropped
    TE< 3 >::Data c{ 1.0, 2.0, 3.0, 4.0 };
    TE< 3 > f{ c };
    auto g = f.integ< 0 >().deriv< 0 >();
    // The top-order coefficient (degree N=3) was dropped by integ, so g recovers f[0..N-1]
    EXPECT_NEAR( g[0], f[0], kTol );
    EXPECT_NEAR( g[1], f[1], kTol );
    EXPECT_NEAR( g[2], f[2], kTol );
    EXPECT_NEAR( g[3], 0.0, kTol );  // f[3] was dropped
}
