#include <gtest/gtest.h>

#include <cmath>
#include <tax/tax.hpp>

TEST( ChebyshevInterp, IdentityRecoversT1 )
{
    auto f = tax::chebyshevInterpolate< 6 >( []( double x ) { return x; } );
    EXPECT_NEAR( f[1], 1.0, 1e-12 );
    EXPECT_NEAR( f[0], 0.0, 1e-12 );
    EXPECT_NEAR( f[2], 0.0, 1e-12 );
}

TEST( ChebyshevInterp, QuadraticIsExact )
{
    // x^2 = (T_0 + T_2)/2; interpolation of a degree-2 poly at >=3 nodes is exact.
    auto f = tax::chebyshevInterpolate< 4 >( []( double x ) { return x * x; } );
    EXPECT_NEAR( f[0], 0.5, 1e-12 );
    EXPECT_NEAR( f[2], 0.5, 1e-12 );
    for ( double p : { -0.7, 0.0, 0.3, 0.9 } ) EXPECT_NEAR( f.eval( p ), p * p, 1e-12 );
}

TEST( ChebyshevInterp, ApproximatesExpUniformly )
{
    auto f = tax::chebyshevInterpolate< 14 >( []( double x ) { return std::exp( x ); } );
    // Near-best uniform approximation over the whole interval, not just near 0.
    for ( double x = -1.0; x <= 1.0; x += 0.05 ) EXPECT_NEAR( f.eval( x ), std::exp( x ), 1e-10 );
}

TEST( ChebyshevInterp, ApproximatesRunge )
{
    // The classic 1/(1+25x^2): Chebyshev nodes avoid the equispaced blow-up.
    // Poles at +-i/5 make convergence geometric but slow, so accuracy is bounded
    // by the degree (unlike equispaced interpolation, which diverges entirely).
    auto f =
        tax::chebyshevInterpolate< 80 >( []( double x ) { return 1.0 / ( 1.0 + 25.0 * x * x ); } );
    for ( double x = -1.0; x <= 1.0; x += 0.05 )
        EXPECT_NEAR( f.eval( x ), 1.0 / ( 1.0 + 25.0 * x * x ), 1e-6 );
}
