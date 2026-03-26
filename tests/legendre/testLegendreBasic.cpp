#include <gtest/gtest.h>

#include <cmath>
#include <tax/basis/legendre_traits.hpp>
#include <tax/tax.hpp>

#include "../testUtils.hpp"

using namespace tax;

// =============================================================================
// Basis transform round-trip tests
// =============================================================================

TEST( LegendreTransform, UnivariateRoundTrip )
{
    // Start with monomial coefficients for p(x) = 1 + 2x + 3x^2
    using Data = std::array< double, 4 >;  // N=3, M=1
    Data mono = { 1.0, 2.0, 3.0, 0.0 };
    Data leg{};
    Data back{};

    detail::monomialToLegendre< double, 3, 1 >( leg, mono );
    detail::legendreToMonomial< double, 3, 1 >( back, leg );

    for ( int i = 0; i < 4; ++i )
        EXPECT_NEAR( back[i], mono[i], 1e-12 ) << "  index=" << i;
}

TEST( LegendreTransform, KnownConversion )
{
    // x^2 = (1/3)*P_0 + (2/3)*P_2  since P_2(x) = (3x^2 - 1)/2
    // So x^2 = (2/3)*(3x^2-1)/2 + 1/3 = x^2 - 1/3 + 1/3 = x^2 ✓
    using Data = std::array< double, 3 >;  // N=2, M=1
    Data mono = { 0.0, 0.0, 1.0 };  // x^2
    Data leg{};

    detail::monomialToLegendre< double, 2, 1 >( leg, mono );

    EXPECT_NEAR( leg[0], 1.0 / 3.0, 1e-12 );  // P_0 coefficient
    EXPECT_NEAR( leg[1], 0.0, 1e-12 );          // P_1 coefficient
    EXPECT_NEAR( leg[2], 2.0 / 3.0, 1e-12 );   // P_2 coefficient
}

TEST( LegendreTransform, BivariateRoundTrip )
{
    using Data = std::array< double, detail::numMonomials( 2, 2 ) >;
    Data mono{};
    mono[0] = 1.0;
    mono[detail::flatIndex< 2 >( { 1, 0 } )] = 1.0;
    mono[detail::flatIndex< 2 >( { 0, 1 } )] = 1.0;
    mono[detail::flatIndex< 2 >( { 1, 1 } )] = 1.0;

    Data leg{}, back{};
    detail::monomialToLegendre< double, 2, 2 >( leg, mono );
    detail::legendreToMonomial< double, 2, 2 >( back, leg );

    for ( std::size_t i = 0; i < mono.size(); ++i )
        EXPECT_NEAR( back[i], mono[i], 1e-11 ) << "  index=" << i;
}

// =============================================================================
// Legendre variable construction
// =============================================================================

TEST( LegendreBasic, VariableConstruction )
{
    // LE<3>::variable(0.0) should be P_1(x) = x
    // Legendre coefficients: [0, 1, 0, 0]
    auto x = LE< 3 >::variable( 0.0 );
    EXPECT_NEAR( x[0], 0.0, 1e-12 );
    EXPECT_NEAR( x[1], 1.0, 1e-12 );
    EXPECT_NEAR( x[2], 0.0, 1e-12 );
    EXPECT_NEAR( x[3], 0.0, 1e-12 );
}

TEST( LegendreBasic, VariableWithOffset )
{
    auto x = LE< 3 >::variable( 0.5 );
    EXPECT_NEAR( x[0], 0.5, 1e-12 );
    EXPECT_NEAR( x[1], 1.0, 1e-12 );
}

TEST( LegendreBasic, ConstantPolynomial )
{
    LE< 3 > c( 5.0 );
    EXPECT_NEAR( c[0], 5.0, 1e-12 );
    for ( int i = 1; i <= 3; ++i ) EXPECT_NEAR( c[i], 0.0, 1e-12 );
}

// =============================================================================
// Clenshaw evaluation
// =============================================================================

TEST( LegendreEval, ClenshawBasic )
{
    auto x = LE< 3 >::variable( 0.0 );
    EXPECT_NEAR( x.eval( 0.5 ), 0.5, 1e-12 );
}

TEST( LegendreEval, ClenshawQuadratic )
{
    // p(x) = P_2(x) = (3x^2 - 1)/2
    LE< 3 > p2( typename LE< 3 >::Data{ 0.0, 0.0, 1.0, 0.0 } );

    EXPECT_NEAR( p2.eval( 0.0 ), -0.5, 1e-12 );
    EXPECT_NEAR( p2.eval( 1.0 ), 1.0, 1e-12 );
    EXPECT_NEAR( p2.eval( -1.0 ), 1.0, 1e-12 );
    EXPECT_NEAR( p2.eval( 0.5 ), ( 3.0 * 0.25 - 1.0 ) / 2.0, 1e-12 );
}

TEST( LegendreEval, ClenshawHigherOrder )
{
    // P_3(x) = (5x^3 - 3x)/2, evaluate at x = 0.5: (5*0.125 - 1.5)/2 = -0.4375
    LE< 5 > p3( typename LE< 5 >::Data{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 } );
    EXPECT_NEAR( p3.eval( 0.5 ), ( 5.0 * 0.125 - 1.5 ) / 2.0, 1e-12 );
}

// =============================================================================
// Legendre multiplication
// =============================================================================

TEST( LegendreMul, P1TimesP1 )
{
    // P_1 * P_1 = x^2 = (1/3)*P_0 + (2/3)*P_2
    auto x = LE< 3 >::variable( 0.0 );
    LE< 3 > x2 = x;
    x2 *= x;

    EXPECT_NEAR( x2[0], 1.0 / 3.0, 1e-12 );
    EXPECT_NEAR( x2[1], 0.0, 1e-12 );
    EXPECT_NEAR( x2[2], 2.0 / 3.0, 1e-12 );
    EXPECT_NEAR( x2[3], 0.0, 1e-12 );
}

// =============================================================================
// Legendre expression templates (via monomial conversion)
// =============================================================================

TEST( LegendreExpr, Addition )
{
    auto x = LE< 3 >::variable( 0.0 );
    LE< 3 > two_x = x + x;

    EXPECT_NEAR( two_x[0], 0.0, 1e-12 );
    EXPECT_NEAR( two_x[1], 2.0, 1e-12 );
    EXPECT_NEAR( two_x[2], 0.0, 1e-12 );
}

TEST( LegendreExpr, ScalarAdd )
{
    auto x = LE< 3 >::variable( 0.0 );
    LE< 3 > xp1 = x + 1.0;

    EXPECT_NEAR( xp1[0], 1.0, 1e-12 );
    EXPECT_NEAR( xp1[1], 1.0, 1e-12 );
}

TEST( LegendreExpr, SinFunction )
{
    auto x = LE< 11 >::variable( 0.0 );
    LE< 11 > s = sin( x );

    for ( double t : { -0.9, -0.5, 0.0, 0.3, 0.7, 1.0 } )
    {
        EXPECT_NEAR( s.eval( t ), std::sin( t ), 1e-8 ) << "  at t=" << t;
    }
}

TEST( LegendreExpr, ExpFunction )
{
    auto x = LE< 11 >::variable( 0.0 );
    LE< 11 > e = exp( x );

    for ( double t : { -1.0, -0.5, 0.0, 0.5, 1.0 } )
    {
        EXPECT_NEAR( e.eval( t ), std::exp( t ), 1e-8 ) << "  at t=" << t;
    }
}

TEST( LegendreExpr, CompositeExpression )
{
    auto x = LE< 13 >::variable( 0.0 );
    LE< 13 > f = sin( x ) * cos( x );

    for ( double t : { -0.8, -0.3, 0.0, 0.4, 0.9 } )
    {
        EXPECT_NEAR( f.eval( t ), 0.5 * std::sin( 2.0 * t ), 1e-7 ) << "  at t=" << t;
    }
}

// =============================================================================
// Legendre derivative
// =============================================================================

TEST( LegendreDeriv, DerivOfP2 )
{
    // P_2(x) = (3x^2-1)/2, d/dx P_2 = 3x = 3*P_1
    LE< 3 > p2( typename LE< 3 >::Data{ 0.0, 0.0, 1.0, 0.0 } );
    auto dp2 = p2.deriv< 0 >();

    EXPECT_NEAR( dp2[0], 0.0, 1e-12 );
    EXPECT_NEAR( dp2[1], 3.0, 1e-12 );
    EXPECT_NEAR( dp2[2], 0.0, 1e-12 );
}

TEST( LegendreDeriv, DerivOfP3 )
{
    // P_3(x) = (5x^3-3x)/2, d/dx P_3 = (15x^2-3)/2
    LE< 4 > p3( typename LE< 4 >::Data{ 0.0, 0.0, 0.0, 1.0, 0.0 } );
    auto dp3 = p3.deriv< 0 >();

    for ( double t : { -0.5, 0.0, 0.5, 0.8 } )
    {
        EXPECT_NEAR( dp3.eval( t ), ( 15.0 * t * t - 3.0 ) / 2.0, 1e-10 ) << "  at t=" << t;
    }
}

// =============================================================================
// Legendre integration
// =============================================================================

TEST( LegendreInteg, DerivIntegRoundTrip )
{
    LE< 5 > p( typename LE< 5 >::Data{ 0.0, 1.0, 0.5, 0.2, 0.0, 0.0 } );
    auto Ip = p.integ< 0 >();
    auto dIp = Ip.deriv< 0 >();

    for ( int i = 1; i < 5; ++i )
        EXPECT_NEAR( dIp[i], p[i], 1e-10 ) << "  coeff=" << i;
}

// =============================================================================
// Taylor <-> Legendre consistency
// =============================================================================

TEST( LegendreConsistency, SinMatchesTaylor )
{
    auto xt = TE< 7 >::variable( 0.0 );
    TE< 7 > st = sin( xt );

    auto xl = LE< 7 >::variable( 0.0 );
    LE< 7 > sl = sin( xl );

    for ( double t : { -0.5, 0.0, 0.25, 0.5 } )
    {
        double val_taylor = st.eval( t );
        double val_legendre = sl.eval( t );
        EXPECT_NEAR( val_taylor, val_legendre, 1e-8 ) << "  at t=" << t;
    }
}

TEST( LegendreConsistency, ExpMatchesTaylor )
{
    auto xt = TE< 7 >::variable( 0.0 );
    TE< 7 > et = exp( xt );

    auto xl = LE< 7 >::variable( 0.0 );
    LE< 7 > el = exp( xl );

    for ( double t : { -0.5, 0.0, 0.25, 0.5 } )
    {
        double val_taylor = et.eval( t );
        double val_legendre = el.eval( t );
        EXPECT_NEAR( val_taylor, val_legendre, 1e-8 ) << "  at t=" << t;
    }
}

// =============================================================================
// Multivariate Legendre evaluation
// =============================================================================

TEST( LegendreEval, MultivariateBasic )
{
    auto [x, y] = LEn< 3, 2 >::variables( 0.0, 0.0 );
    LEn< 3, 2 > f = x * y;

    // x*y at (0.3, 0.7) should be 0.21
    EXPECT_NEAR( f.eval( { 0.3, 0.7 } ), 0.21, 1e-10 );
}
