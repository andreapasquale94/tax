#include <gtest/gtest.h>

#include <tax/tax.hpp>

#include "../testUtils.hpp"

// --- Hermite (probabilists') ---

TEST( BasisConversion, HermiteKnownUnivariateSquare )
{
    // f = x^2 = He_2(x) + He_0(x)   [He_2 = x^2 - 1]
    auto x = tax::TE< 2 >::variable( 0.0 );
    auto f = x * x;
    auto h = tax::toHermite( f );
    EXPECT_NEAR( ( h.coeff< 0 >() ), 1.0, 1e-12 );
    EXPECT_NEAR( ( h.coeff< 1 >() ), 0.0, 1e-12 );
    EXPECT_NEAR( ( h.coeff< 2 >() ), 1.0, 1e-12 );
}

TEST( BasisConversion, HermiteKnownUnivariateCube )
{
    // f = x^3 = He_3(x) + 3*He_1(x)   [He_3 = x^3 - 3x]
    auto x = tax::TE< 3 >::variable( 0.0 );
    auto f = x * x * x;
    auto h = tax::toHermite( f );
    EXPECT_NEAR( ( h.coeff< 0 >() ), 0.0, 1e-12 );
    EXPECT_NEAR( ( h.coeff< 1 >() ), 3.0, 1e-12 );
    EXPECT_NEAR( ( h.coeff< 2 >() ), 0.0, 1e-12 );
    EXPECT_NEAR( ( h.coeff< 3 >() ), 1.0, 1e-12 );
}

TEST( BasisConversion, HermiteFromBasisElementIsTextbookPolynomial )
{
    // He_3(x) = x^3 - 3x
    tax::HermiteCoefficients< double, 3, 1 > h{};
    h.data[3] = 1.0;
    auto f = tax::fromHermite( h );
    EXPECT_NEAR( ( f.coeff< 0 >() ), 0.0, 1e-12 );
    EXPECT_NEAR( ( f.coeff< 1 >() ), -3.0, 1e-12 );
    EXPECT_NEAR( ( f.coeff< 2 >() ), 0.0, 1e-12 );
    EXPECT_NEAR( ( f.coeff< 3 >() ), 1.0, 1e-12 );
}

TEST( BasisConversion, HermiteRoundTripUnivariate )
{
    auto x = tax::TE< 5 >::variable( 0.3 );
    auto f = tax::exp( x );
    auto h = tax::toHermite( f );
    auto back = tax::fromHermite( h );
    tax::test::ExpectCoeffsNear( back, f, 1e-9 );
}

TEST( BasisConversion, HermiteRoundTripMultivariate )
{
    typename tax::TE< 4, 2 >::Input p{ 0.1, -0.2 };
    auto x = tax::TE< 4, 2 >::variable< 0 >( p );
    auto y = tax::TE< 4, 2 >::variable< 1 >( p );
    auto f = tax::sin( x ) * tax::cos( y ) + x * x * y;
    auto h = tax::toHermite( f );
    auto back = tax::fromHermite( h );
    tax::test::ExpectCoeffsNear( back, f, 1e-9 );
}

TEST( BasisConversion, HermiteMultivariateProductFactorizes )
{
    // f = x^2 * y^2 = (He_0(x)+He_2(x)) * (He_0(y)+He_2(y)): all four cross
    // terms have unit coefficient, everything else is zero.
    typename tax::TE< 4, 2 >::Input p{ 0.0, 0.0 };
    auto x = tax::TE< 4, 2 >::variable< 0 >( p );
    auto y = tax::TE< 4, 2 >::variable< 1 >( p );
    auto f = x * x * y * y;
    auto h = tax::toHermite( f );
    EXPECT_NEAR( ( h.coeff< 0, 0 >() ), 1.0, 1e-12 );
    EXPECT_NEAR( ( h.coeff< 0, 2 >() ), 1.0, 1e-12 );
    EXPECT_NEAR( ( h.coeff< 2, 0 >() ), 1.0, 1e-12 );
    EXPECT_NEAR( ( h.coeff< 2, 2 >() ), 1.0, 1e-12 );
    EXPECT_NEAR( ( h.coeff< 1, 1 >() ), 0.0, 1e-12 );
}

// --- Chebyshev (first kind) ---

TEST( BasisConversion, ChebyshevKnownUnivariateSquare )
{
    // f = x^2 = 0.5*T_0 + 0.5*T_2   [T_2 = 2x^2 - 1]
    auto x = tax::TE< 2 >::variable( 0.0 );
    auto f = x * x;
    auto c = tax::toChebyshev( f );
    EXPECT_NEAR( ( c.coeff< 0 >() ), 0.5, 1e-12 );
    EXPECT_NEAR( ( c.coeff< 1 >() ), 0.0, 1e-12 );
    EXPECT_NEAR( ( c.coeff< 2 >() ), 0.5, 1e-12 );
}

TEST( BasisConversion, ChebyshevKnownUnivariateCube )
{
    // f = x^3 = 0.75*T_1 + 0.25*T_3   [T_3 = 4x^3 - 3x]
    auto x = tax::TE< 3 >::variable( 0.0 );
    auto f = x * x * x;
    auto c = tax::toChebyshev( f );
    EXPECT_NEAR( ( c.coeff< 0 >() ), 0.0, 1e-12 );
    EXPECT_NEAR( ( c.coeff< 1 >() ), 0.75, 1e-12 );
    EXPECT_NEAR( ( c.coeff< 2 >() ), 0.0, 1e-12 );
    EXPECT_NEAR( ( c.coeff< 3 >() ), 0.25, 1e-12 );
}

TEST( BasisConversion, ChebyshevFromBasisElementIsTextbookPolynomial )
{
    // T_3(x) = 4x^3 - 3x
    tax::ChebyshevCoefficients< double, 3, 1 > c{};
    c.data[3] = 1.0;
    auto f = tax::fromChebyshev( c );
    EXPECT_NEAR( ( f.coeff< 0 >() ), 0.0, 1e-12 );
    EXPECT_NEAR( ( f.coeff< 1 >() ), -3.0, 1e-12 );
    EXPECT_NEAR( ( f.coeff< 2 >() ), 0.0, 1e-12 );
    EXPECT_NEAR( ( f.coeff< 3 >() ), 4.0, 1e-12 );
}

TEST( BasisConversion, ChebyshevRoundTripUnivariate )
{
    auto x = tax::TE< 5 >::variable( 0.3 );
    auto f = tax::exp( x );
    auto c = tax::toChebyshev( f );
    auto back = tax::fromChebyshev( c );
    tax::test::ExpectCoeffsNear( back, f, 1e-9 );
}

TEST( BasisConversion, ChebyshevRoundTripMultivariate )
{
    typename tax::TE< 4, 2 >::Input p{ 0.1, -0.2 };
    auto x = tax::TE< 4, 2 >::variable< 0 >( p );
    auto y = tax::TE< 4, 2 >::variable< 1 >( p );
    auto f = tax::sin( x ) * tax::cos( y ) + x * x * y;
    auto c = tax::toChebyshev( f );
    auto back = tax::fromChebyshev( c );
    tax::test::ExpectCoeffsNear( back, f, 1e-9 );
}
