#include <gtest/gtest.h>

#include <cmath>
#include <tax/basis/chebyshev_traits.hpp>
#include <tax/tax.hpp>

#include "../testUtils.hpp"

using namespace tax;

// =============================================================================
// Basis transform round-trip tests
// =============================================================================

TEST( ChebyshevTransform, UnivariateRoundTrip )
{
    // Start with monomial coefficients for p(x) = 1 + 2x + 3x^2
    using Data = std::array< double, 4 >;  // N=3, M=1
    Data mono = { 1.0, 2.0, 3.0, 0.0 };
    Data cheb{};
    Data back{};

    detail::monomialToChebyshev< double, 3, 1 >( cheb, mono );
    detail::chebyshevToMonomial< double, 3, 1 >( back, cheb );

    for ( int i = 0; i < 4; ++i )
        EXPECT_NEAR( back[i], mono[i], 1e-12 ) << "  index=" << i;
}

TEST( ChebyshevTransform, KnownConversion )
{
    // x^2 = 0.5*T_0 + 0*T_1 + 0.5*T_2  (since T_2(x) = 2x^2 - 1)
    using Data = std::array< double, 3 >;  // N=2, M=1
    Data mono = { 0.0, 0.0, 1.0 };  // x^2
    Data cheb{};

    detail::monomialToChebyshev< double, 2, 1 >( cheb, mono );

    EXPECT_NEAR( cheb[0], 0.5, 1e-12 );   // T_0 coefficient
    EXPECT_NEAR( cheb[1], 0.0, 1e-12 );   // T_1 coefficient
    EXPECT_NEAR( cheb[2], 0.5, 1e-12 );   // T_2 coefficient
}

TEST( ChebyshevTransform, BivariateRoundTrip )
{
    // 2-variable, order 2: p(x,y) = 1 + x + y + xy
    using Data = std::array< double, detail::numMonomials( 2, 2 ) >;
    Data mono{};
    mono[0] = 1.0;  // constant
    // Find flat indices for x, y, xy
    mono[detail::flatIndex< 2 >( { 1, 0 } )] = 1.0;  // x
    mono[detail::flatIndex< 2 >( { 0, 1 } )] = 1.0;  // y
    mono[detail::flatIndex< 2 >( { 1, 1 } )] = 1.0;  // xy

    Data cheb{}, back{};
    detail::monomialToChebyshev< double, 2, 2 >( cheb, mono );
    detail::chebyshevToMonomial< double, 2, 2 >( back, cheb );

    for ( std::size_t i = 0; i < mono.size(); ++i )
        EXPECT_NEAR( back[i], mono[i], 1e-11 ) << "  index=" << i;
}

// =============================================================================
// Chebyshev variable construction
// =============================================================================

TEST( ChebyshevBasic, VariableConstruction )
{
    // CE<3>::variable(0.0) should be T_1(x) = x
    // Chebyshev coefficients: [0, 1, 0, 0]
    auto x = CE< 3 >::variable( 0.0 );
    EXPECT_NEAR( x[0], 0.0, 1e-12 );
    EXPECT_NEAR( x[1], 1.0, 1e-12 );
    EXPECT_NEAR( x[2], 0.0, 1e-12 );
    EXPECT_NEAR( x[3], 0.0, 1e-12 );
}

TEST( ChebyshevBasic, VariableWithOffset )
{
    // CE<3>::variable(0.5) should have constant=0.5, T_1 coeff=1
    auto x = CE< 3 >::variable( 0.5 );
    EXPECT_NEAR( x[0], 0.5, 1e-12 );
    EXPECT_NEAR( x[1], 1.0, 1e-12 );
}

TEST( ChebyshevBasic, ConstantPolynomial )
{
    CE< 3 > c( 5.0 );
    EXPECT_NEAR( c[0], 5.0, 1e-12 );
    for ( int i = 1; i <= 3; ++i ) EXPECT_NEAR( c[i], 0.0, 1e-12 );
}

// =============================================================================
// Clenshaw evaluation
// =============================================================================

TEST( ChebyshevEval, ClenshawBasic )
{
    // Evaluate T_1(x) = x at x = 0.5
    auto x = CE< 3 >::variable( 0.0 );
    EXPECT_NEAR( x.eval( 0.5 ), 0.5, 1e-12 );
}

TEST( ChebyshevEval, ClenshawQuadratic )
{
    // p(x) = 0.5*T_0 + 0.5*T_2 = 0.5 + 0.5*(2x^2 - 1) = x^2
    CE< 3 > p( typename CE< 3 >::Data{ 0.5, 0.0, 0.5, 0.0 } );

    EXPECT_NEAR( p.eval( 0.0 ), 0.0, 1e-12 );
    EXPECT_NEAR( p.eval( 0.5 ), 0.25, 1e-12 );
    EXPECT_NEAR( p.eval( 1.0 ), 1.0, 1e-12 );
    EXPECT_NEAR( p.eval( -1.0 ), 1.0, 1e-12 );
}

TEST( ChebyshevEval, ClenshawHigherOrder )
{
    // T_3(x) = 4x^3 - 3x, evaluate at x = 0.5: 4*(1/8) - 3*(1/2) = -1
    CE< 5 > t3( typename CE< 5 >::Data{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 } );
    EXPECT_NEAR( t3.eval( 0.5 ), 4.0 * 0.125 - 1.5, 1e-12 );
}

// =============================================================================
// Chebyshev multiplication
// =============================================================================

TEST( ChebyshevMul, T1TimesT1 )
{
    // T_1 * T_1 = 0.5*(T_0 + T_2)
    auto x = CE< 3 >::variable( 0.0 );
    CE< 3 > x2 = x;
    x2 *= x;

    EXPECT_NEAR( x2[0], 0.5, 1e-12 );   // T_0 coeff
    EXPECT_NEAR( x2[1], 0.0, 1e-12 );   // T_1 coeff
    EXPECT_NEAR( x2[2], 0.5, 1e-12 );   // T_2 coeff
    EXPECT_NEAR( x2[3], 0.0, 1e-12 );   // T_3 coeff
}

TEST( ChebyshevMul, T1TimesT2 )
{
    // T_1 * T_2 = 0.5*(T_1 + T_3)
    CE< 4 > t1( typename CE< 4 >::Data{ 0.0, 1.0, 0.0, 0.0, 0.0 } );
    CE< 4 > t2( typename CE< 4 >::Data{ 0.0, 0.0, 1.0, 0.0, 0.0 } );
    CE< 4 > prod = t1;
    prod *= t2;

    EXPECT_NEAR( prod[0], 0.0, 1e-12 );
    EXPECT_NEAR( prod[1], 0.5, 1e-12 );   // T_1
    EXPECT_NEAR( prod[2], 0.0, 1e-12 );
    EXPECT_NEAR( prod[3], 0.5, 1e-12 );   // T_3
    EXPECT_NEAR( prod[4], 0.0, 1e-12 );
}

// =============================================================================
// Chebyshev expression templates (via monomial conversion)
// =============================================================================

TEST( ChebyshevExpr, Addition )
{
    auto x = CE< 3 >::variable( 0.0 );
    CE< 3 > two_x = x + x;

    EXPECT_NEAR( two_x[0], 0.0, 1e-12 );
    EXPECT_NEAR( two_x[1], 2.0, 1e-12 );
    EXPECT_NEAR( two_x[2], 0.0, 1e-12 );
}

TEST( ChebyshevExpr, ScalarAdd )
{
    auto x = CE< 3 >::variable( 0.0 );
    CE< 3 > xp1 = x + 1.0;

    EXPECT_NEAR( xp1[0], 1.0, 1e-12 );
    EXPECT_NEAR( xp1[1], 1.0, 1e-12 );
}

TEST( ChebyshevExpr, Multiplication )
{
    // x * x via expression templates should give same as in-place
    auto x = CE< 5 >::variable( 0.0 );
    CE< 5 > x2_expr = x * x;

    // Expected: 0.5*T_0 + 0.5*T_2
    EXPECT_NEAR( x2_expr[0], 0.5, 1e-10 );
    EXPECT_NEAR( x2_expr[1], 0.0, 1e-10 );
    EXPECT_NEAR( x2_expr[2], 0.5, 1e-10 );
}

TEST( ChebyshevExpr, SinFunction )
{
    // sin(x) for x in [-1,1] — higher order for better accuracy
    auto x = CE< 11 >::variable( 0.0 );
    CE< 11 > s = sin( x );

    for ( double t : { -0.9, -0.5, 0.0, 0.3, 0.7, 1.0 } )
    {
        EXPECT_NEAR( s.eval( t ), std::sin( t ), 1e-8 ) << "  at t=" << t;
    }
}

TEST( ChebyshevExpr, ExpFunction )
{
    // exp(x) for x in [-1,1]
    auto x = CE< 11 >::variable( 0.0 );
    CE< 11 > e = exp( x );

    for ( double t : { -1.0, -0.5, 0.0, 0.5, 1.0 } )
    {
        EXPECT_NEAR( e.eval( t ), std::exp( t ), 1e-8 ) << "  at t=" << t;
    }
}

TEST( ChebyshevExpr, CosFunction )
{
    auto x = CE< 11 >::variable( 0.0 );
    CE< 11 > c = cos( x );

    for ( double t : { -1.0, -0.5, 0.0, 0.5, 1.0 } )
    {
        EXPECT_NEAR( c.eval( t ), std::cos( t ), 1e-8 ) << "  at t=" << t;
    }
}

TEST( ChebyshevExpr, CompositeExpression )
{
    // sin(x) * cos(x) = 0.5 * sin(2x)
    auto x = CE< 13 >::variable( 0.0 );
    CE< 13 > f = sin( x ) * cos( x );

    for ( double t : { -0.8, -0.3, 0.0, 0.4, 0.9 } )
    {
        EXPECT_NEAR( f.eval( t ), 0.5 * std::sin( 2.0 * t ), 1e-7 ) << "  at t=" << t;
    }
}

// =============================================================================
// Chebyshev derivative
// =============================================================================

TEST( ChebyshevDeriv, DerivOfT2 )
{
    // T_2(x) = 2x^2 - 1, d/dx T_2 = 4x
    // In Chebyshev: derivative of T_2 has coeff c'_1 = 4 (T_1 = x)
    CE< 3 > t2( typename CE< 3 >::Data{ 0.0, 0.0, 1.0, 0.0 } );
    auto dt2 = t2.deriv< 0 >();

    // 4*T_1 means coeff[1] = 4
    EXPECT_NEAR( dt2[0], 0.0, 1e-12 );
    EXPECT_NEAR( dt2[1], 4.0, 1e-12 );
    EXPECT_NEAR( dt2[2], 0.0, 1e-12 );
}

TEST( ChebyshevDeriv, DerivOfT3 )
{
    // T_3(x) = 4x^3 - 3x, d/dx T_3 = 12x^2 - 3
    // In Chebyshev: 12x^2 - 3 = 12*(T_0+T_2)/2 - 3 = 6*T_0 + 6*T_2 - 3 = 3*T_0 + 6*T_2
    CE< 4 > t3( typename CE< 4 >::Data{ 0.0, 0.0, 0.0, 1.0, 0.0 } );
    auto dt3 = t3.deriv< 0 >();

    // Verify via evaluation at sample points
    for ( double t : { -0.5, 0.0, 0.5, 0.8 } )
    {
        EXPECT_NEAR( dt3.eval( t ), 12.0 * t * t - 3.0, 1e-10 ) << "  at t=" << t;
    }
}

// =============================================================================
// Chebyshev integration
// =============================================================================

TEST( ChebyshevInteg, IntegOfT1 )
{
    // integral T_1(x) dx = T_2(x)/4 + C
    // Using formula: C_k = (c_{k-1} - c_{k+1}) / (2k)
    // Input: c_0=0, c_1=1, c_2=0, c_3=0
    // C_1 = (c_0 - c_2)/(2*1) = 0
    // C_2 = (c_1 - c_3)/(2*2) = 1/4
    CE< 4 > t1( typename CE< 4 >::Data{ 0.0, 1.0, 0.0, 0.0, 0.0 } );
    auto It1 = t1.integ< 0 >();

    EXPECT_NEAR( It1[0], 0.0, 1e-12 );     // constant of integration
    EXPECT_NEAR( It1[1], 0.0, 1e-12 );     // C_1
    EXPECT_NEAR( It1[2], 0.25, 1e-12 );    // C_2 = 1/4
}

TEST( ChebyshevInteg, DerivIntegRoundTrip )
{
    // Start with a Chebyshev polynomial, integrate then differentiate: should recover original
    // (up to constant term)
    CE< 5 > p( typename CE< 5 >::Data{ 0.0, 1.0, 0.5, 0.2, 0.0, 0.0 } );  // no constant term
    auto Ip = p.integ< 0 >();
    auto dIp = Ip.deriv< 0 >();

    // Should recover p (except possibly constant term roundoff)
    for ( int i = 1; i < 5; ++i )
        EXPECT_NEAR( dIp[i], p[i], 1e-10 ) << "  coeff=" << i;
}

// =============================================================================
// Taylor <-> Chebyshev consistency
// =============================================================================

TEST( ChebyshevConsistency, SinMatchesTaylor )
{
    // Compute sin(x) around x=0 in Taylor and Chebyshev, verify they agree at sample points
    auto xt = TE< 7 >::variable( 0.0 );
    TE< 7 > st = sin( xt );

    auto xc = CE< 7 >::variable( 0.0 );
    CE< 7 > sc = sin( xc );

    for ( double t : { -0.5, 0.0, 0.25, 0.5 } )
    {
        double val_taylor = st.eval( t );
        double val_cheby = sc.eval( t );
        EXPECT_NEAR( val_taylor, val_cheby, 1e-8 ) << "  at t=" << t;
    }
}
