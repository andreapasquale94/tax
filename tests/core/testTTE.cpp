#include "testUtils.hpp"

// =============================================================================
// Constructors
// =============================================================================

TEST( TTEConstruct, DefaultIsZero )
{
    TE< 4 > a;
    for ( std::size_t k = 0; k < TE< 4 >::nCoefficients; ++k ) EXPECT_EQ( a[k], 0.0 ) << "k=" << k;
}

TEST( TTEConstruct, ValueCtorSetsConstant )
{
    TE< 4 > a{ 3.14 };
    EXPECT_NEAR( a[0], 3.14, kTol );
    for ( std::size_t k = 1; k < TE< 4 >::nCoefficients; ++k ) EXPECT_EQ( a[k], 0.0 ) << "k=" << k;
}

TEST( TTEConstruct, CoeffArrayCtor )
{
    TE< 3 >::Data c{ 1, 2, 3, 4 };
    TE< 3 > a{ c };
    for ( std::size_t k = 0; k < TE< 3 >::nCoefficients; ++k )
        EXPECT_NEAR( a[k], double( k + 1 ), kTol ) << "k=" << k;
}

TEST( TTEConstruct, FromExpression )
{
    auto x = TE< 4 >::variable< 0 >( { 1.0 } );
    TE< 4 > r = x + x;  // construct from expression
    EXPECT_NEAR( r[0], 2.0, kTol );
    EXPECT_NEAR( r[1], 2.0, kTol );
    for ( std::size_t k = 2; k < TE< 4 >::nCoefficients; ++k ) EXPECT_EQ( r[k], 0.0 );
}

// =============================================================================
// Variable factories
// =============================================================================

TEST( TTEVariable, Univariate_Variable0 )
{
    // variable<0>({x0}) = x0 + 1*delta
    auto x = TE< 4 >::variable< 0 >( { 3.0 } );
    EXPECT_NEAR( x[0], 3.0, kTol );  // expansion point
    EXPECT_NEAR( x[1], 1.0, kTol );  // linear coefficient = 1
    for ( std::size_t k = 2; k < TE< 4 >::nCoefficients; ++k ) EXPECT_EQ( x[k], 0.0 ) << "k=" << k;
}

TEST( TTEVariable, Bivariate_Variable0 )
{
    auto x = TEn< 3, 2 >::variable< 0 >( { 2.0, 5.0 } );
    EXPECT_NEAR( x.coeff( { 0, 0 } ), 2.0, kTol );  // expansion point for x
    EXPECT_NEAR( x.coeff( { 1, 0 } ), 1.0, kTol );  // dx/dx = 1
    EXPECT_NEAR( x.coeff( { 0, 1 } ), 0.0, kTol );  // dx/dy = 0
}

TEST( TTEVariable, Bivariate_Variable1 )
{
    auto y = TEn< 3, 2 >::variable< 1 >( { 2.0, 5.0 } );
    EXPECT_NEAR( y.coeff( { 0, 0 } ), 5.0, kTol );  // expansion point for y
    EXPECT_NEAR( y.coeff( { 1, 0 } ), 0.0, kTol );  // dy/dx = 0
    EXPECT_NEAR( y.coeff( { 0, 1 } ), 1.0, kTol );  // dy/dy = 1
}

TEST( TTEVariable, Variables_StructuredBinding )
{
    auto [x, y, z] = TEn< 2, 3 >::variables( { 1.0, 2.0, 3.0 } );
    EXPECT_NEAR( x.value(), 1.0, kTol );
    EXPECT_NEAR( y.value(), 2.0, kTol );
    EXPECT_NEAR( z.value(), 3.0, kTol );
    // Each variable has coefficient 1 for its own direction only
    EXPECT_NEAR( x.coeff( { 1, 0, 0 } ), 1.0, kTol );
    EXPECT_NEAR( x.coeff( { 0, 1, 0 } ), 0.0, kTol );
    EXPECT_NEAR( y.coeff( { 1, 0, 0 } ), 0.0, kTol );
    EXPECT_NEAR( y.coeff( { 0, 1, 0 } ), 1.0, kTol );
    EXPECT_NEAR( z.coeff( { 0, 0, 1 } ), 1.0, kTol );
}

TEST( TTEVariable, Variables_SplattedStructuredBinding )
{
    auto [x, y, z] = TEn< 2, 3 >::variables( 1.0, 2.0f, 3 );
    EXPECT_NEAR( x.value(), 1.0, kTol );
    EXPECT_NEAR( y.value(), 2.0, kTol );
    EXPECT_NEAR( z.value(), 3.0, kTol );
    EXPECT_NEAR( x.coeff( { 1, 0, 0 } ), 1.0, kTol );
    EXPECT_NEAR( y.coeff( { 0, 1, 0 } ), 1.0, kTol );
    EXPECT_NEAR( z.coeff( { 0, 0, 1 } ), 1.0, kTol );
}

TEST( TTEVariable, Constant )
{
    auto c = TE< 5 >::constant( 7.0 );
    EXPECT_NEAR( c.value(), 7.0, kTol );
    for ( std::size_t k = 1; k < TE< 5 >::nCoefficients; ++k ) EXPECT_EQ( c[k], 0.0 );
}

// =============================================================================
// Element access
// =============================================================================

TEST( TTEAccess, ReadWrite )
{
    TE< 3 > a;
    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = -1.0;
    a[3] = 0.5;
    EXPECT_NEAR( a[0], 1.0, kTol );
    EXPECT_NEAR( a[1], 2.0, kTol );
    EXPECT_NEAR( a[2], -1.0, kTol );
    EXPECT_NEAR( a[3], 0.5, kTol );
}

TEST( TTEAccess, Value )
{
    TE< 3 > a{ 5.0 };
    EXPECT_NEAR( a.value(), 5.0, kTol );
}

TEST( TTEAccess, Coeff_Alpha )
{
    auto x = TE< 4 >::variable< 0 >( { 2.0 } );
    EXPECT_NEAR( x.coeff( { 0 } ), 2.0, kTol );
    EXPECT_NEAR( x.coeff( { 1 } ), 1.0, kTol );
    EXPECT_NEAR( x.coeff( { 2 } ), 0.0, kTol );
}

TEST( TTEAccess, Coeff_TemplateUnivariate )
{
    auto x = TE< 4 >::variable< 0 >( { 2.0 } );
    EXPECT_NEAR( x.coeff< 0 >(), 2.0, kTol );
    EXPECT_NEAR( x.coeff< 1 >(), 1.0, kTol );
    EXPECT_NEAR( x.coeff< 2 >(), 0.0, kTol );
}

TEST( TTEAccess, Coeff_TemplateBivariate )
{
    auto [x, y] = TEn< 2, 2 >::variables( { 0.0, 0.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;
    EXPECT_NEAR( ( f.coeff< 2, 0 >() ), 1.0, kTol );
    EXPECT_NEAR( ( f.coeff< 1, 1 >() ), 1.0, kTol );
    EXPECT_NEAR( ( f.coeff< 0, 2 >() ), 1.0, kTol );
}

TEST( TTEAccess, Derivative_FactorialFactor )
{
    // (1+x)^2 = 1 + 2x + x^2: derivative at x=0:
    //   d/dx = 2 → derivative({1}) = c[1]*1! = 2
    //   d^2/dx^2 = 2 → derivative({2}) = c[2]*2! = 1*2 = 2
    TE< 3 > a{};
    a[0] = 1;
    a[1] = 2;
    a[2] = 1;  // coefficients of (1+x)^2
    EXPECT_NEAR( a.derivative( { 0 } ), 1.0, kTol );
    EXPECT_NEAR( a.derivative( { 1 } ), 2.0, kTol );
    EXPECT_NEAR( a.derivative( { 2 } ), 2.0, kTol );
}

TEST( TTEAccess, Derivative_TemplateUnivariate )
{
    TE< 3 > a{};
    a[0] = 1;
    a[1] = 2;
    a[2] = 1;
    EXPECT_NEAR( a.derivative< 0 >(), 1.0, kTol );
    EXPECT_NEAR( a.derivative< 1 >(), 2.0, kTol );
    EXPECT_NEAR( a.derivative< 2 >(), 2.0, kTol );
}

TEST( TTEAccess, Derivative_Bivariate )
{
    // f(x,y) = x^2 + x*y + y^2 at (0,0): coefficients are exact
    // d^2f/dx^2 = 2 → coeff({2,0}) = 1, derivative({2,0}) = 1*2! = 2
    auto [x, y] = TEn< 2, 2 >::variables( { 0.0, 0.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;
    EXPECT_NEAR( f.derivative( { 2, 0 } ), 2.0, kTol );
    EXPECT_NEAR( f.derivative( { 0, 2 } ), 2.0, kTol );
    EXPECT_NEAR( f.derivative( { 1, 1 } ), 1.0, kTol );
}

TEST( TTEAccess, Derivative_TemplateBivariate )
{
    auto [x, y] = TEn< 2, 2 >::variables( { 0.0, 0.0 } );
    TEn< 2, 2 > f = x * x + x * y + y * y;
    EXPECT_NEAR( ( f.derivative< 2, 0 >() ), 2.0, kTol );
    EXPECT_NEAR( ( f.derivative< 0, 2 >() ), 2.0, kTol );
    EXPECT_NEAR( ( f.derivative< 1, 1 >() ), 1.0, kTol );
}

TEST( TTEAccess, Coeffs_ReturnsArray )
{
    TE< 2 > a{};
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    const auto& c = a.coeffs();
    EXPECT_NEAR( c[0], 1.0, kTol );
    EXPECT_NEAR( c[1], 2.0, kTol );
    EXPECT_NEAR( c[2], 3.0, kTol );
}

TEST( TTENorm, MaxNormMatchesNormInf )
{
    TE< 3 > a{};
    a[0] = 1.0;
    a[1] = -2.5;
    a[2] = 0.25;
    a[3] = -4.0;

    EXPECT_NEAR( a.coeffsNormInf(), 4.0, kTol );
}

TEST( TTENorm, L1Norm )
{
    TE< 3 > a{};
    a[0] = 1.0;
    a[1] = -2.5;
    a[2] = 0.25;
    a[3] = -4.0;

    EXPECT_NEAR( a.coeffsNorm( 1 ), 7.75, kTol );
}

TEST( TTENorm, LPNorm )
{
    TE< 3 > a{};
    a[0] = 1.0;
    a[1] = -2.5;
    a[2] = 0.25;
    a[3] = -4.0;

    const double expected = std::sqrt( 1.0 + 2.5 * 2.5 + 0.25 * 0.25 + 4.0 * 4.0 );
    EXPECT_NEAR( a.coeffsNorm( 2 ), expected, kTol );
}

TEST( TTENorm, CompileTimeNormType )
{
    auto x = TE< 3 >::variable( 1.0 );
    TE< 3 > y = 2.0 * x + 1.0;

    EXPECT_NEAR( y.coeffsNormInf(), 3.0, kTol );
    EXPECT_NEAR( y.coeffsNorm< 1 >(), 5.0, kTol );
    EXPECT_NEAR( y.coeffsNorm< 2 >(), std::sqrt( 13.0 ), kTol );
}

TEST( TTENorm, MultivariateNorm )
{
    TEn< 2, 2 > a{};
    a[0] = 1.0;
    a[1] = -2.0;
    a[2] = 3.0;
    a[3] = 0.0;
    a[4] = -4.0;
    a[5] = 0.5;

    EXPECT_NEAR( a.coeffsNormInf(), 4.0, kTol );
    EXPECT_NEAR( a.coeffsNorm( 1 ), 10.5, kTol );
}

TEST( TTENorm, RuntimeOrderMustBePositive )
{
    TE< 3 > a{};
    a[0] = 1.0;
    EXPECT_THROW( (void)a.coeffsNorm( 0 ), std::invalid_argument );
}

TEST( TTENorm, NormEstimateUnivariateExponentialFit )
{
    constexpr int N = 5;
    TE< N > a{};
    for ( int k = 0; k <= N; ++k ) a[std::size_t( k )] = std::exp( 1.0 - 0.4 * double( k ) );

    const auto est = a.coeffsNormEstimate( 0, 1, 8 );
    ASSERT_EQ( est.size(), 9u );
    for ( int k = 0; k <= 8; ++k )
        EXPECT_NEAR( est[std::size_t( k )], std::exp( 1.0 - 0.4 * double( k ) ), 1e-12 );
}

TEST( TTENorm, NormEstimateGroupedByVariable )
{
    TEn< 2, 2 > a{};
    a[0] = 1.0;   // (0,0): exponent x=0
    a[1] = 0.1;   // (1,0): exponent x=1
    a[2] = 1.0;   // (0,1): exponent x=0
    a[3] = 0.01;  // (2,0): exponent x=2
    a[4] = 0.1;   // (1,1): exponent x=1
    a[5] = 1.0;   // (0,2): exponent x=0

    const auto est = a.coeffsNormEstimate( 1, 0, 2 );
    ASSERT_EQ( est.size(), 3u );
    EXPECT_NEAR( est[0], 1.0, 1e-12 );
    EXPECT_NEAR( est[1], 0.1, 1e-12 );
    EXPECT_NEAR( est[2], 0.01, 1e-12 );
}

TEST( TTENorm, NormEstimateRejectsOutOfRangeVariableIndex )
{
    TEn< 2, 2 > a{};
    EXPECT_THROW( (void)a.coeffsNormEstimate( 3, 0, 2 ), std::invalid_argument );
}

// =============================================================================
// In-place operators
// =============================================================================

TEST( TTEInPlace, PlusEqTTE )
{
    TE< 3 > a{ 2.0 }, b{ 3.0 };
    a += b;
    EXPECT_NEAR( a.value(), 5.0, kTol );
}

TEST( TTEInPlace, MinusEqTTE )
{
    TE< 3 > a{ 5.0 }, b{ 3.0 };
    a -= b;
    EXPECT_NEAR( a.value(), 2.0, kTol );
}

TEST( TTEInPlace, PlusEqExpression )
{
    auto x = TE< 4 >::variable< 0 >( { 1.0 } );
    TE< 4 > r{ 2.0 };
    r += x;  // 2 + (1+δ)
    EXPECT_NEAR( r[0], 3.0, kTol );
    EXPECT_NEAR( r[1], 1.0, kTol );
}

TEST( TTEInPlace, MinusEqExpression )
{
    auto x = TE< 4 >::variable< 0 >( { 1.0 } );
    TE< 4 > r{ 2.0 };
    r -= x;  // 2 - (1+δ) = 1 - δ
    EXPECT_NEAR( r[0], 1.0, kTol );
    EXPECT_NEAR( r[1], -1.0, kTol );
}

TEST( TTEInPlace, TimesEqScalar )
{
    auto x = TE< 3 >::variable< 0 >( { 2.0 } );
    TE< 3 > r = x;
    r *= 3.0;  // 3*(2+δ) = 6 + 3δ
    EXPECT_NEAR( r[0], 6.0, kTol );
    EXPECT_NEAR( r[1], 3.0, kTol );
}

TEST( TTEInPlace, DivEqScalar )
{
    auto x = TE< 3 >::variable< 0 >( { 4.0 } );
    TE< 3 > r = x;
    r /= 2.0;  // (4+δ)/2 = 2 + 0.5δ
    EXPECT_NEAR( r[0], 2.0, kTol );
    EXPECT_NEAR( r[1], 0.5, kTol );
}

TEST( TTEInPlace, ChainedPlusEq )
{
    TE< 2 > a{ 1.0 }, b{ 2.0 }, c{ 3.0 };
    a += b;
    a += c;
    EXPECT_NEAR( a.value(), 6.0, kTol );
}

// =============================================================================
// eval — polynomial evaluation
// =============================================================================

TEST( TTEEval, UnivariateConstant )
{
    TE< 3 > a{ 5.0 };
    EXPECT_NEAR( a.eval( 0.1 ), 5.0, kTol );
    EXPECT_NEAR( a.eval( 0.0 ), 5.0, kTol );
}

TEST( TTEEval, UnivariateLinear )
{
    // f = 2 + 3*dx  (at x0=2, slope 3)
    auto x = TE< 3 >::variable( 2.0 );
    TE< 3 > f = 3.0 * x;  // 6 + 3*dx
    EXPECT_NEAR( f.eval( 0.0 ), 6.0, kTol );
    EXPECT_NEAR( f.eval( 1.0 ), 9.0, kTol );
    EXPECT_NEAR( f.eval( -0.5 ), 4.5, kTol );
}

TEST( TTEEval, UnivariateSinAtZero )
{
    // sin(x) expanded at x0=0, evaluate at dx to get sin(dx)
    auto x = TE< 9 >::variable( 0.0 );
    TE< 9 > f = sin( x );
    EXPECT_NEAR( f.eval( 0.3 ), std::sin( 0.3 ), 1e-12 );
    EXPECT_NEAR( f.eval( 0.5 ), std::sin( 0.5 ), 1e-10 );
}

TEST( TTEEval, UnivariateExpAtOne )
{
    // exp(x) expanded at x0=1, evaluate at dx=0.5 to get exp(1.5)
    auto x = TE< 12 >::variable( 1.0 );
    TE< 12 > f = exp( x );
    EXPECT_NEAR( f.eval( 0.5 ), std::exp( 1.5 ), 1e-12 );
}

TEST( TTEEval, UnivariateAtZeroDisplacement )
{
    auto x = TE< 5 >::variable( 1.0 );
    TE< 5 > f = sin( x );
    EXPECT_NEAR( f.eval( 0.0 ), std::sin( 1.0 ), kTol );
}

TEST( TTEEval, UnivariateViaPointType )
{
    // eval(Input) delegates to eval(T) for M=1
    auto x = TE< 9 >::variable( 0.0 );
    TE< 9 > f = cos( x );
    TE< 9 >::Input dx{ 0.3 };
    EXPECT_NEAR( f.eval( dx ), std::cos( 0.3 ), 1e-11 );
}

TEST( TTEEval, MultivariateConstant )
{
    TEn< 3, 2 > a{ 5.0 };
    EXPECT_NEAR( a.eval( { 0.1, 0.2 } ), 5.0, kTol );
}

TEST( TTEEval, MultivariateLinear )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    TEn< 3, 2 > f = x + 2.0 * y;  // (1+dx) + 2*(2+dy) = 5 + dx + 2*dy
    EXPECT_NEAR( f.eval( { 0.0, 0.0 } ), 5.0, kTol );
    EXPECT_NEAR( f.eval( { 1.0, 0.0 } ), 6.0, kTol );
    EXPECT_NEAR( f.eval( { 0.0, 1.0 } ), 7.0, kTol );
    EXPECT_NEAR( f.eval( { 0.5, 0.3 } ), 5.0 + 0.5 + 0.6, kTol );
}

TEST( TTEEval, MultivariateQuadratic )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 0.0, 0.0 } );
    TEn< 3, 2 > f = x * y;  // dx*dy
    EXPECT_NEAR( f.eval( { 2.0, 3.0 } ), 6.0, kTol );
    EXPECT_NEAR( f.eval( { 0.0, 5.0 } ), 0.0, kTol );
}

TEST( TTEEval, MultivariateSin )
{
    auto [x, y] = TEn< 8, 2 >::variables( { 0.0, 0.0 } );
    TEn< 8, 2 > f = sin( x + y );
    // sin(dx + dy) at small displacements
    EXPECT_NEAR( f.eval( { 0.1, 0.2 } ), std::sin( 0.3 ), 1e-10 );
}

TEST( TTEEval, MultivariateEigenVector )
{
    auto [x, y] = TEn< 3, 2 >::variables( { 1.0, 2.0 } );
    TEn< 3, 2 > f = x + 2.0 * y;  // 5 + dx + 2*dy
    Eigen::Vector2d dx( 0.5, 0.3 );
    EXPECT_NEAR( tax::eval( f, dx ), 5.0 + 0.5 + 0.6, kTol );
}
