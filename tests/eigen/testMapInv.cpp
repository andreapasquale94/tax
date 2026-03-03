#include <tax/eigen/map_inv.hpp>

#include "../testUtils.hpp"

// =============================================================================
// Component-wise univariate inv on DAVec
// =============================================================================

TEST( MapInv, DAVec_ComponentWise )
{
    // Two independent univariate functions: f_0(x)=x+x^2, f_1(x)=2x
    DAVec< 5, 2 > f;
    f( 0 )    = DA< 5 >{};
    f( 0 )[1] = 1.0;
    f( 0 )[2] = 1.0;
    f( 1 )    = DA< 5 >{};
    f( 1 )[1] = 2.0;

    DAVec< 5, 2 > g = inv( f );

    // g_0 = inv(x+x^2): Catalan-like
    EXPECT_NEAR( g( 0 )[1], 1.0, kTol );
    EXPECT_NEAR( g( 0 )[2], -1.0, kTol );
    EXPECT_NEAR( g( 0 )[3], 2.0, kTol );

    // g_1 = inv(2x) = x/2
    EXPECT_NEAR( g( 1 )[1], 0.5, kTol );
    for ( int k = 2; k <= 5; ++k ) EXPECT_NEAR( g( 1 )[k], 0.0, kTol );
}

// =============================================================================
// Multivariate map inversion (DAnVec)
// =============================================================================

TEST( MapInv, Identity2D )
{
    // f(x,y) = (x, y) — identity map. Inverse is identity.
    auto [x, y] = DAn< 4, 2 >::variables( { 0.0, 0.0 } );
    DAnVec< 4, 2 > f;
    f( 0 ) = x;
    f( 1 ) = y;

    DAnVec< 4, 2 > g = inv( f );

    // g should be identity: g_0 = x, g_1 = y
    EXPECT_NEAR( g( 0 ).coeff( { 1, 0 } ), 1.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 1, 0 } ), 0.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 0, 1 } ), 1.0, kTol );

    // All higher-order terms should be zero
    for ( std::size_t k = 3; k < DAn< 4, 2 >::ncoef; ++k )
    {
        EXPECT_NEAR( g( 0 )[k], 0.0, kTol ) << "g(0)[" << k << "]";
        EXPECT_NEAR( g( 1 )[k], 0.0, kTol ) << "g(1)[" << k << "]";
    }
}

TEST( MapInv, LinearMap2D )
{
    // f(x,y) = (2x + y, x + 3y).  J = [[2,1],[1,3]], J^{-1} = (1/5)[[3,-1],[-1,2]]
    auto [x, y] = DAn< 4, 2 >::variables( { 0.0, 0.0 } );
    DAnVec< 4, 2 > f;
    f( 0 ) = 2.0 * x + y;
    f( 1 ) = x + 3.0 * y;

    DAnVec< 4, 2 > g = inv( f );

    // g_0(u,v) = (3u - v) / 5
    EXPECT_NEAR( g( 0 ).coeff( { 1, 0 } ), 3.0 / 5.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 0, 1 } ), -1.0 / 5.0, kTol );

    // g_1(u,v) = (-u + 2v) / 5
    EXPECT_NEAR( g( 1 ).coeff( { 1, 0 } ), -1.0 / 5.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 0, 1 } ), 2.0 / 5.0, kTol );

    // No higher-order terms for a linear map
    for ( std::size_t k = 3; k < DAn< 4, 2 >::ncoef; ++k )
    {
        EXPECT_NEAR( g( 0 )[k], 0.0, kTol ) << "g(0)[" << k << "]";
        EXPECT_NEAR( g( 1 )[k], 0.0, kTol ) << "g(1)[" << k << "]";
    }
}

TEST( MapInv, QuadraticMap2D )
{
    // f(x,y) = (x + x*y, y + x^2)
    // Jacobian at origin: J = [[1,0],[0,1]] = I → J^{-1} = I
    // Nonlinear: N = (x*y, x^2)
    auto [x, y] = DAn< 4, 2 >::variables( { 0.0, 0.0 } );
    DAnVec< 4, 2 > f;
    f( 0 ) = x + x * y;
    f( 1 ) = y + x * x;

    DAnVec< 4, 2 > g = inv( f );

    // Linear part: g ≈ (u, v) + O(2)
    EXPECT_NEAR( g( 0 ).coeff( { 1, 0 } ), 1.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 1, 0 } ), 0.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 0, 1 } ), 1.0, kTol );

    const auto fog = detail::composeMap( f, g );
    const auto gof = detail::composeMap( g, f );

    // Series inversion is exact in coefficient space up to truncation order N.
    ExpectCoeffsNear( fog( 0 ), x, 1e-9 );
    ExpectCoeffsNear( fog( 1 ), y, 1e-9 );
    ExpectCoeffsNear( gof( 0 ), x, 1e-9 );
    ExpectCoeffsNear( gof( 1 ), y, 1e-9 );
}

TEST( MapInv, NonIdentityJacobian )
{
    // f(x,y) = (2x + x^2, 3y + y^2)  — decoupled nonlinear map
    // J = [[2,0],[0,3]]
    auto [x, y] = DAn< 5, 2 >::variables( { 0.0, 0.0 } );
    DAnVec< 5, 2 > f;
    f( 0 ) = 2.0 * x + x * x;
    f( 1 ) = 3.0 * y + y * y;

    DAnVec< 5, 2 > g = inv( f );

    // Since the map is decoupled, g_0 should depend only on u and g_1 only on v.
    // g_0(u) = inv of 2x + x^2 as a function of u
    // g_1(v) = inv of 3y + y^2 as a function of v

    // Verify cross-terms are zero
    EXPECT_NEAR( g( 0 ).coeff( { 0, 1 } ), 0.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 0, 2 } ), 0.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 1, 0 } ), 0.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 2, 0 } ), 0.0, kTol );

    // Verify round-trip
    for ( double u : { -0.2, 0.0, 0.15 } )
    {
        for ( double v : { -0.2, 0.0, 0.15 } )
        {
            double g0  = g( 0 ).eval( { u, v } );
            double g1  = g( 1 ).eval( { u, v } );
            double fg0 = f( 0 ).eval( { g0, g1 } );
            double fg1 = f( 1 ).eval( { g0, g1 } );
            EXPECT_NEAR( fg0, u, 1e-5 ) << "  u=" << u << " v=" << v;
            EXPECT_NEAR( fg1, v, 1e-5 ) << "  u=" << u << " v=" << v;
        }
    }
}

TEST( MapInv, RoundTrip3D )
{
    // f(x,y,z) = (x + y*z, y + x*z, z + x*y)
    // J = I at origin, nonlinear cross terms
    auto [x, y, z] = DAn< 4, 3 >::variables( { 0.0, 0.0, 0.0 } );
    DAnVec< 4, 3 > f;
    f( 0 ) = x + y * z;
    f( 1 ) = y + x * z;
    f( 2 ) = z + x * y;

    DAnVec< 4, 3 > g = inv( f );

    const auto fog = detail::composeMap( f, g );
    const auto gof = detail::composeMap( g, f );

    // Check both compositions against identity map in coefficient space.
    ExpectCoeffsNear( fog( 0 ), x, 1e-9 );
    ExpectCoeffsNear( fog( 1 ), y, 1e-9 );
    ExpectCoeffsNear( fog( 2 ), z, 1e-9 );
    ExpectCoeffsNear( gof( 0 ), x, 1e-9 );
    ExpectCoeffsNear( gof( 1 ), y, 1e-9 );
    ExpectCoeffsNear( gof( 2 ), z, 1e-9 );
}

TEST( MapInv, CoupledQuadratic_Coefficients )
{
    // f(x,y) = (x + x^2 + x*y, y + y^2)
    // J = I, so inverse linear part is identity.
    // Degree-2 terms of g can be verified analytically.
    //
    // f(g(u,v)) = u → at degree 2:
    //   g_0^{(2,0)} + 1 = 0  →  g_0^{(2,0)} = -1    (from x^2 term in f_0)
    //   g_0^{(1,1)} + 1 = 0  →  g_0^{(1,1)} = -1    (from x*y term in f_0, using linear g)
    //   g_0^{(0,2)} = 0
    //
    // f(g(u,v)) = v → at degree 2:
    //   g_1^{(0,2)} + 1 = 0  →  g_1^{(0,2)} = -1    (from y^2 term in f_1)
    //   g_1^{(2,0)} = 0, g_1^{(1,1)} = 0
    auto [x, y] = DAn< 4, 2 >::variables( { 0.0, 0.0 } );
    DAnVec< 4, 2 > f;
    f( 0 ) = x + x * x + x * y;
    f( 1 ) = y + y * y;

    DAnVec< 4, 2 > g = inv( f );

    // Degree-2 coefficients
    EXPECT_NEAR( g( 0 ).coeff( { 2, 0 } ), -1.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 1, 1 } ), -1.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 0, 2 } ), 0.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 2, 0 } ), 0.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 1, 1 } ), 0.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 0, 2 } ), -1.0, kTol );
}

TEST( MapInv, CoupledQuadraticNonIdentityJacobian_Coefficients )
{
    // f(x,y) = (2x + y + x*y, x + 3y + y^2)
    // J = [[2,1],[1,3]], J^{-1} = (1/5)[[3,-1],[-1,2]]
    //
    // Using g = L + Q + O(3) with L = J^{-1}(u,v), degree-2 terms satisfy:
    //   J Q + N(L) = 0  =>  Q = -J^{-1}N(L)
    // with N = (x*y, y^2).
    auto [x, y] = DAn< 4, 2 >::variables( { 0.0, 0.0 } );
    DAnVec< 4, 2 > f;
    f( 0 ) = 2.0 * x + y + x * y;
    f( 1 ) = x + 3.0 * y + y * y;

    DAnVec< 4, 2 > g = inv( f );

    // Linear part: J^{-1}
    EXPECT_NEAR( g( 0 ).coeff( { 1, 0 } ), 3.0 / 5.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 0, 1 } ), -1.0 / 5.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 1, 0 } ), -1.0 / 5.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 0, 1 } ), 2.0 / 5.0, kTol );

    // Degree-2 terms from analytic expansion.
    EXPECT_NEAR( g( 0 ).coeff( { 2, 0 } ), 2.0 / 25.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 1, 1 } ), -1.0 / 5.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 0, 2 } ), 2.0 / 25.0, kTol );

    EXPECT_NEAR( g( 1 ).coeff( { 2, 0 } ), -1.0 / 25.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 1, 1 } ), 3.0 / 25.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 0, 2 } ), -2.0 / 25.0, kTol );
}

TEST( MapInv, BidirectionalCompositionIsIdentity2D )
{
    // Coupled nonlinear map with invertible non-identity Jacobian.
    auto [x, y] = DAn< 5, 2 >::variables( { 0.0, 0.0 } );
    DAnVec< 5, 2 > f;
    f( 0 ) = 1.5 * x + 0.4 * y + 0.2 * x * x - 0.1 * x * y;
    f( 1 ) = -0.3 * x + 1.2 * y + 0.15 * x * y + 0.25 * y * y;

    DAnVec< 5, 2 > g = inv( f );

    const auto fog = detail::composeMap( f, g );
    const auto gof = detail::composeMap( g, f );

    // f ∘ g = id and g ∘ f = id (up to truncation order).
    ExpectCoeffsNear( fog( 0 ), x, 1e-9 );
    ExpectCoeffsNear( fog( 1 ), y, 1e-9 );
    ExpectCoeffsNear( gof( 0 ), x, 1e-9 );
    ExpectCoeffsNear( gof( 1 ), y, 1e-9 );
}

TEST( MapInv, TallLinearMap_LeftInverse )
{
    // f : R^2 -> R^3, f(x,y) = (x, y, x+y)
    auto [x, y] = DAn< 4, 2 >::variables( { 0.0, 0.0 } );
    Eigen::Matrix< DAn< 4, 2 >, 3, 1 > f;
    f( 0 ) = x;
    f( 1 ) = y;
    f( 2 ) = x + y;

    // g : R^3 -> R^2 should satisfy g(f(x,y)) = (x,y).
    auto g = inv( f );

    // J = [[1,0],[0,1],[1,1]], G = (J^T J)^(-1) J^T = (1/3)[[2,-1,1],[-1,2,1]]
    EXPECT_NEAR( g( 0 ).coeff( { 1, 0, 0 } ), 2.0 / 3.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 0, 1, 0 } ), -1.0 / 3.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 0, 0, 1 } ), 1.0 / 3.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 1, 0, 0 } ), -1.0 / 3.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 0, 1, 0 } ), 2.0 / 3.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 0, 0, 1 } ), 1.0 / 3.0, kTol );

    for ( double a : { -0.3, 0.0, 0.2 } )
    {
        for ( double b : { -0.1, 0.0, 0.25 } )
        {
            const double u = f( 0 ).eval( { a, b } );
            const double v = f( 1 ).eval( { a, b } );
            const double w = f( 2 ).eval( { a, b } );

            const double ga = g( 0 ).eval( { u, v, w } );
            const double gb = g( 1 ).eval( { u, v, w } );
            EXPECT_NEAR( ga, a, 1e-10 );
            EXPECT_NEAR( gb, b, 1e-10 );
        }
    }
}

TEST( MapInv, WideLinearMap_RightInverse )
{
    // f : R^3 -> R^2, f(x,y,z) = (x+z, y+z)
    auto [x, y, z] = DAn< 4, 3 >::variables( { 0.0, 0.0, 0.0 } );
    Eigen::Matrix< DAn< 4, 3 >, 2, 1 > f;
    f( 0 ) = x + z;
    f( 1 ) = y + z;

    // g : R^2 -> R^3 should satisfy f(g(u,v)) = (u,v).
    auto g = inv( f );

    // J = [[1,0,1],[0,1,1]], G = J^T (J J^T)^(-1)
    //   = [[ 2/3,-1/3],[-1/3, 2/3],[ 1/3, 1/3]]
    EXPECT_NEAR( g( 0 ).coeff( { 1, 0 } ), 2.0 / 3.0, kTol );
    EXPECT_NEAR( g( 0 ).coeff( { 0, 1 } ), -1.0 / 3.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 1, 0 } ), -1.0 / 3.0, kTol );
    EXPECT_NEAR( g( 1 ).coeff( { 0, 1 } ), 2.0 / 3.0, kTol );
    EXPECT_NEAR( g( 2 ).coeff( { 1, 0 } ), 1.0 / 3.0, kTol );
    EXPECT_NEAR( g( 2 ).coeff( { 0, 1 } ), 1.0 / 3.0, kTol );

    for ( double u : { -0.2, 0.0, 0.35 } )
    {
        for ( double v : { -0.25, 0.0, 0.15 } )
        {
            const double gx = g( 0 ).eval( { u, v } );
            const double gy = g( 1 ).eval( { u, v } );
            const double gz = g( 2 ).eval( { u, v } );

            const double fu = f( 0 ).eval( { gx, gy, gz } );
            const double fv = f( 1 ).eval( { gx, gy, gz } );
            EXPECT_NEAR( fu, u, 1e-10 );
            EXPECT_NEAR( fv, v, 1e-10 );
        }
    }
}
