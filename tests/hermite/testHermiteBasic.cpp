#include <gtest/gtest.h>

#include <cmath>
#include <tax/basis/hermite_traits.hpp>
#include <tax/tax.hpp>

#include "../testUtils.hpp"

using namespace tax;

// =============================================================================
// Basis transform round-trip tests
// =============================================================================

TEST( HermiteTransform, UnivariateRoundTrip )
{
    using Data = std::array< double, 4 >;  // N=3, M=1
    Data mono = { 1.0, 2.0, 3.0, 0.0 };
    Data herm{};
    Data back{};

    detail::monomialToHermite< double, 3, 1 >( herm, mono );
    detail::hermiteToMonomial< double, 3, 1 >( back, herm );

    for ( int i = 0; i < 4; ++i )
        EXPECT_NEAR( back[i], mono[i], 1e-12 ) << "  index=" << i;
}

TEST( HermiteTransform, KnownConversion )
{
    // x^2 = He_0 + He_2 since He_2(x) = x^2 - 1, so x^2 = He_2 + 1 = 1*He_0 + 0*He_1 + 1*He_2
    using Data = std::array< double, 3 >;  // N=2, M=1
    Data mono = { 0.0, 0.0, 1.0 };  // x^2
    Data herm{};

    detail::monomialToHermite< double, 2, 1 >( herm, mono );

    EXPECT_NEAR( herm[0], 1.0, 1e-12 );  // He_0 coefficient
    EXPECT_NEAR( herm[1], 0.0, 1e-12 );  // He_1 coefficient
    EXPECT_NEAR( herm[2], 1.0, 1e-12 );  // He_2 coefficient
}

TEST( HermiteTransform, BivariateRoundTrip )
{
    using Data = std::array< double, detail::numMonomials( 2, 2 ) >;
    Data mono{};
    mono[0] = 1.0;
    mono[detail::flatIndex< 2 >( { 1, 0 } )] = 1.0;
    mono[detail::flatIndex< 2 >( { 0, 1 } )] = 1.0;
    mono[detail::flatIndex< 2 >( { 1, 1 } )] = 1.0;

    Data herm{}, back{};
    detail::monomialToHermite< double, 2, 2 >( herm, mono );
    detail::hermiteToMonomial< double, 2, 2 >( back, herm );

    for ( std::size_t i = 0; i < mono.size(); ++i )
        EXPECT_NEAR( back[i], mono[i], 1e-11 ) << "  index=" << i;
}

// =============================================================================
// Hermite variable construction
// =============================================================================

TEST( HermiteBasic, VariableConstruction )
{
    // HE<3>::variable(0.0) should be He_1(x) = x
    // Hermite coefficients: [0, 1, 0, 0]
    auto x = HE< 3 >::variable( 0.0 );
    EXPECT_NEAR( x[0], 0.0, 1e-12 );
    EXPECT_NEAR( x[1], 1.0, 1e-12 );
    EXPECT_NEAR( x[2], 0.0, 1e-12 );
    EXPECT_NEAR( x[3], 0.0, 1e-12 );
}

TEST( HermiteBasic, VariableWithOffset )
{
    auto x = HE< 3 >::variable( 0.5 );
    EXPECT_NEAR( x[0], 0.5, 1e-12 );
    EXPECT_NEAR( x[1], 1.0, 1e-12 );
}

TEST( HermiteBasic, ConstantPolynomial )
{
    HE< 3 > c( 5.0 );
    EXPECT_NEAR( c[0], 5.0, 1e-12 );
    for ( int i = 1; i <= 3; ++i ) EXPECT_NEAR( c[i], 0.0, 1e-12 );
}

// =============================================================================
// Clenshaw evaluation
// =============================================================================

TEST( HermiteEval, ClenshawBasic )
{
    auto x = HE< 3 >::variable( 0.0 );
    EXPECT_NEAR( x.eval( 0.5 ), 0.5, 1e-12 );
}

TEST( HermiteEval, ClenshawQuadratic )
{
    // He_2(x) = x^2 - 1
    HE< 3 > he2( typename HE< 3 >::Data{ 0.0, 0.0, 1.0, 0.0 } );

    EXPECT_NEAR( he2.eval( 0.0 ), -1.0, 1e-12 );
    EXPECT_NEAR( he2.eval( 1.0 ), 0.0, 1e-12 );
    EXPECT_NEAR( he2.eval( -1.0 ), 0.0, 1e-12 );
    EXPECT_NEAR( he2.eval( 2.0 ), 3.0, 1e-12 );
}

TEST( HermiteEval, ClenshawHigherOrder )
{
    // He_3(x) = x^3 - 3x, evaluate at x = 2: 8 - 6 = 2
    HE< 5 > he3( typename HE< 5 >::Data{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 } );
    EXPECT_NEAR( he3.eval( 2.0 ), 8.0 - 6.0, 1e-12 );
}

// =============================================================================
// Hermite multiplication
// =============================================================================

TEST( HermiteMul, He1TimesHe1 )
{
    // He_1 * He_1 = x^2 = He_0 + He_2
    auto x = HE< 3 >::variable( 0.0 );
    HE< 3 > x2 = x;
    x2 *= x;

    EXPECT_NEAR( x2[0], 1.0, 1e-12 );
    EXPECT_NEAR( x2[1], 0.0, 1e-12 );
    EXPECT_NEAR( x2[2], 1.0, 1e-12 );
    EXPECT_NEAR( x2[3], 0.0, 1e-12 );
}

// =============================================================================
// Hermite expression templates (via monomial conversion)
// =============================================================================

TEST( HermiteExpr, Addition )
{
    auto x = HE< 3 >::variable( 0.0 );
    HE< 3 > two_x = x + x;

    EXPECT_NEAR( two_x[0], 0.0, 1e-12 );
    EXPECT_NEAR( two_x[1], 2.0, 1e-12 );
    EXPECT_NEAR( two_x[2], 0.0, 1e-12 );
}

TEST( HermiteExpr, ScalarAdd )
{
    auto x = HE< 3 >::variable( 0.0 );
    HE< 3 > xp1 = x + 1.0;

    EXPECT_NEAR( xp1[0], 1.0, 1e-12 );
    EXPECT_NEAR( xp1[1], 1.0, 1e-12 );
}

TEST( HermiteExpr, SinFunction )
{
    auto x = HE< 11 >::variable( 0.0 );
    HE< 11 > s = sin( x );

    for ( double t : { -0.9, -0.5, 0.0, 0.3, 0.7, 1.0 } )
    {
        EXPECT_NEAR( s.eval( t ), std::sin( t ), 1e-8 ) << "  at t=" << t;
    }
}

TEST( HermiteExpr, ExpFunction )
{
    auto x = HE< 11 >::variable( 0.0 );
    HE< 11 > e = exp( x );

    for ( double t : { -1.0, -0.5, 0.0, 0.5, 1.0 } )
    {
        EXPECT_NEAR( e.eval( t ), std::exp( t ), 1e-8 ) << "  at t=" << t;
    }
}

TEST( HermiteExpr, CompositeExpression )
{
    auto x = HE< 13 >::variable( 0.0 );
    HE< 13 > f = sin( x ) * cos( x );

    for ( double t : { -0.8, -0.3, 0.0, 0.4, 0.9 } )
    {
        EXPECT_NEAR( f.eval( t ), 0.5 * std::sin( 2.0 * t ), 1e-7 ) << "  at t=" << t;
    }
}

// =============================================================================
// Hermite derivative
// =============================================================================

TEST( HermiteDeriv, DerivOfHe2 )
{
    // He_2(x) = x^2 - 1, d/dx He_2 = 2x = 2*He_1
    HE< 3 > he2( typename HE< 3 >::Data{ 0.0, 0.0, 1.0, 0.0 } );
    auto dhe2 = he2.deriv< 0 >();

    EXPECT_NEAR( dhe2[0], 0.0, 1e-12 );
    EXPECT_NEAR( dhe2[1], 2.0, 1e-12 );
    EXPECT_NEAR( dhe2[2], 0.0, 1e-12 );
}

TEST( HermiteDeriv, DerivOfHe3 )
{
    // He_3(x) = x^3 - 3x, d/dx He_3 = 3x^2 - 3 = 3*He_2 (since He_2 = x^2-1, 3*He_2 = 3x^2-3)
    HE< 4 > he3( typename HE< 4 >::Data{ 0.0, 0.0, 0.0, 1.0, 0.0 } );
    auto dhe3 = he3.deriv< 0 >();

    for ( double t : { -0.5, 0.0, 0.5, 0.8 } )
    {
        EXPECT_NEAR( dhe3.eval( t ), 3.0 * t * t - 3.0, 1e-10 ) << "  at t=" << t;
    }
}

// =============================================================================
// Hermite integration
// =============================================================================

TEST( HermiteInteg, DerivIntegRoundTrip )
{
    HE< 5 > p( typename HE< 5 >::Data{ 0.0, 1.0, 0.5, 0.2, 0.0, 0.0 } );
    auto Ip = p.integ< 0 >();
    auto dIp = Ip.deriv< 0 >();

    for ( int i = 1; i < 5; ++i )
        EXPECT_NEAR( dIp[i], p[i], 1e-10 ) << "  coeff=" << i;
}

// =============================================================================
// Taylor <-> Hermite consistency
// =============================================================================

TEST( HermiteConsistency, SinMatchesTaylor )
{
    auto xt = TE< 7 >::variable( 0.0 );
    TE< 7 > st = sin( xt );

    auto xh = HE< 7 >::variable( 0.0 );
    HE< 7 > sh = sin( xh );

    for ( double t : { -0.5, 0.0, 0.25, 0.5 } )
    {
        double val_taylor = st.eval( t );
        double val_hermite = sh.eval( t );
        EXPECT_NEAR( val_taylor, val_hermite, 1e-8 ) << "  at t=" << t;
    }
}

TEST( HermiteConsistency, ExpMatchesTaylor )
{
    auto xt = TE< 7 >::variable( 0.0 );
    TE< 7 > et = exp( xt );

    auto xh = HE< 7 >::variable( 0.0 );
    HE< 7 > eh = exp( xh );

    for ( double t : { -0.5, 0.0, 0.25, 0.5 } )
    {
        double val_taylor = et.eval( t );
        double val_hermite = eh.eval( t );
        EXPECT_NEAR( val_taylor, val_hermite, 1e-8 ) << "  at t=" << t;
    }
}

// =============================================================================
// Multivariate Hermite evaluation
// =============================================================================

TEST( HermiteEval, MultivariateBasic )
{
    auto [x, y] = HEn< 3, 2 >::variables( 0.0, 0.0 );
    HEn< 3, 2 > f = x * y;

    EXPECT_NEAR( f.eval( { 0.3, 0.7 } ), 0.21, 1e-10 );
}
