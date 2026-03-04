#include "testUtils.hpp"

using InDA = TEn< 3, 2 >;          // order 3, 2 spatial vars (x, y)
using OutDA = TruncatedTaylorExpansionT< InDA, 4, 1 >;  // order 4 in time, coefficients are spatial DAs

// =============================================================================
// Basic construction and arithmetic
// =============================================================================

TEST( Nested, ConstantConstruction )
{
    InDA s{ 5.0 };
    OutDA f{ s };
    EXPECT_NEAR( f.value().value(), 5.0, kTol );
}

TEST( Nested, Addition )
{
    InDA a{ 2.0 }, b{ 3.0 };
    OutDA fa{ a }, fb{ b };
    OutDA fc = fa + fb;
    EXPECT_NEAR( fc.value().value(), 5.0, kTol );
}

TEST( Nested, ScalarMultiplication )
{
    InDA a{ 2.0 };
    OutDA fa{ a };
    OutDA fb = fa * InDA{ 3.0 };
    EXPECT_NEAR( fb.value().value(), 6.0, kTol );
}

// =============================================================================
// Variable in time with spatial coefficients
// =============================================================================

TEST( Nested, TimeVariable )
{
    // f(t) = t, where coefficients are spatial DAs
    // c[0] = InDA{0}, c[1] = InDA{1}
    OutDA t = OutDA::variable( InDA{ 0.0 } );
    EXPECT_NEAR( t[0].value(), 0.0, kTol );
    EXPECT_NEAR( t[1].value(), 1.0, kTol );
}

TEST( Nested, TimeVariableWithSpatialExpansionPoint )
{
    // Expand around t0 where t0 is a spatial polynomial
    auto [x, y] = InDA::variables( { 1.0, 2.0 } );
    InDA t0 = x + y;  // t0 = x + y (a spatial polynomial)

    OutDA t = OutDA::variable( t0 );
    // t[0] should be the spatial polynomial x + y
    EXPECT_NEAR( t[0].value(), 3.0, kTol );                 // x+y at (1,2) = 3
    EXPECT_NEAR( t[0].derivative( { 1, 0 } ), 1.0, kTol );  // d/dx(x+y) = 1
    EXPECT_NEAR( t[0].derivative( { 0, 1 } ), 1.0, kTol );  // d/dy(x+y) = 1
    // t[1] = InDA{1} (constant)
    EXPECT_NEAR( t[1].value(), 1.0, kTol );
}

// =============================================================================
// Polynomial evaluation in time (Horner)
// =============================================================================

TEST( Nested, EvalInTime )
{
    // f(t) = a + b*t where a, b are spatial DAs
    auto [x, y] = InDA::variables( { 1.0, 2.0 } );
    InDA a = x * y;  // a = x*y
    InDA b = x + y;  // b = x+y

    OutDA::coeff_array coeffs{};
    coeffs[0] = a;
    coeffs[1] = b;
    OutDA f{ coeffs };

    // Evaluate at dt = 0.5 (a scalar displacement in time)
    InDA result = f.eval( InDA{ 0.5 } );
    // f(0.5) = x*y + 0.5*(x+y) at (1,2) = 2 + 0.5*3 = 3.5
    EXPECT_NEAR( result.value(), 3.5, kTol );
    // d/dx f(0.5) = y + 0.5 at (1,2) = 2.5
    EXPECT_NEAR( result.derivative( { 1, 0 } ), 2.5, kTol );
}

// =============================================================================
// Arithmetic on time variable
// =============================================================================

TEST( Nested, SquareInTime )
{
    OutDA t = OutDA::variable( InDA{ 0.0 } );
    OutDA t2 = t * t;
    // t^2: c[0]=0, c[1]=0, c[2]=1
    EXPECT_NEAR( t2[0].value(), 0.0, kTol );
    EXPECT_NEAR( t2[1].value(), 0.0, kTol );
    EXPECT_NEAR( t2[2].value(), 1.0, kTol );
}

TEST( Nested, PolynomialInTime )
{
    // f(t) = 1 + 2t + 3t^2, constant spatial coefficients
    OutDA t = OutDA::variable( InDA{ 0.0 } );
    OutDA f = InDA{ 1.0 } + InDA{ 2.0 } * t + InDA{ 3.0 } * t * t;

    EXPECT_NEAR( f[0].value(), 1.0, kTol );
    EXPECT_NEAR( f[1].value(), 2.0, kTol );
    EXPECT_NEAR( f[2].value(), 3.0, kTol );
}

// =============================================================================
// Transcendental functions on nested DA
// =============================================================================

TEST( Nested, ExpInTime )
{
    // exp(t) where t is a time variable with spatial DA coefficients
    OutDA t = OutDA::variable( InDA{ 0.0 } );
    OutDA f = exp( t );

    // exp(t) at t=0: coeffs are 1, 1, 1/2, 1/6, 1/24
    EXPECT_NEAR( f[0].value(), 1.0, kTol );
    EXPECT_NEAR( f[1].value(), 1.0, kTol );
    EXPECT_NEAR( f[2].value(), 0.5, kTol );
    EXPECT_NEAR( f[3].value(), 1.0 / 6.0, kTol );
    EXPECT_NEAR( f[4].value(), 1.0 / 24.0, kTol );
}

TEST( Nested, SinInTime )
{
    OutDA t = OutDA::variable( InDA{ 0.0 } );
    OutDA f = sin( t );

    // sin(t) at t=0: coeffs are 0, 1, 0, -1/6, 0
    EXPECT_NEAR( f[0].value(), 0.0, kTol );
    EXPECT_NEAR( f[1].value(), 1.0, kTol );
    EXPECT_NEAR( f[2].value(), 0.0, kTol );
    EXPECT_NEAR( f[3].value(), -1.0 / 6.0, kTol );
    EXPECT_NEAR( f[4].value(), 0.0, kTol );
}

TEST( Nested, ExpWithSpatialExpansionPoint )
{
    // exp(x + t) expanded at (x0=1, t0=0)
    auto [x, y] = InDA::variables( { 1.0, 0.0 } );
    OutDA t = OutDA::variable( x );  // t0 = x (spatial DA at x0=1)
    OutDA f = exp( t );

    // f[0] = exp(x) as a spatial polynomial at x0=1
    EXPECT_NEAR( f[0].value(), std::exp( 1.0 ), kTol );
    EXPECT_NEAR( f[0].derivative( { 1, 0 } ), std::exp( 1.0 ), kTol );  // d/dx exp(x) = exp(x)

    // f[1] = exp(x) (since d/dt exp(x+t)|_{t=0} = exp(x))
    EXPECT_NEAR( f[1].value(), std::exp( 1.0 ), kTol );
}

// =============================================================================
// Mixed spatial-time function: f(x, t) = sin(x) * t
// =============================================================================

TEST( Nested, MixedSpatialTime )
{
    auto [x, y] = InDA::variables( { 0.5, 0.0 } );
    OutDA t = OutDA::variable( InDA{ 0.0 } );

    // f = sin(x) * t: each time coeff multiplied by sin(x)
    InDA sinx = sin( x );
    OutDA f = OutDA{ sinx } * t;

    // f[0] = 0 (no constant term in t)
    EXPECT_NEAR( f[0].value(), 0.0, kTol );
    // f[1] = sin(x) at x0=0.5
    EXPECT_NEAR( f[1].value(), std::sin( 0.5 ), kTol );
    EXPECT_NEAR( f[1].derivative( { 1, 0 } ), std::cos( 0.5 ), kTol );
}
