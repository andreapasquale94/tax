#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/tax.hpp>

using tax::ChebyshevSeries;
using tax::TaylorSeries;

TEST( SeriesConvert, MonomialSquareToChebyshev )
{
    // x^2 in monomial basis -> (T_0 + T_2)/2 in Chebyshev basis.
    auto x = TaylorSeries< 4 >::variable();
    auto f = x * x;
    auto cheb = tax::toChebyshev( f );
    EXPECT_NEAR( cheb[0], 0.5, 1e-13 );
    EXPECT_NEAR( cheb[2], 0.5, 1e-13 );
}

TEST( SeriesConvert, RoundTripTaylor )
{
    std::array< double, 6 > raw{ 1.0, -2.0, 0.5, 3.0, -1.5, 0.25 };
    TaylorSeries< 5 > f{ raw };
    auto back = tax::toTaylor( tax::toChebyshev( f ) );
    for ( int k = 0; k <= 5; ++k )
        EXPECT_NEAR( back[std::size_t( k )], raw[std::size_t( k )], 1e-10 );
}

TEST( SeriesConvert, EvalAgreesAcrossBases )
{
    // exp built via the Taylor recurrences, then moved into the Chebyshev basis:
    // the two polynomials must evaluate identically (exact basis change).
    auto x = TaylorSeries< 8 >::variable();
    auto f = exp( x );
    auto cheb = tax::toChebyshev( f );
    for ( double p : { -0.5, -0.1, 0.2, 0.7 } ) EXPECT_NEAR( f.eval( p ), cheb.eval( p ), 1e-11 );
}

TEST( SeriesConvert, ChebyshevToTaylorEval )
{
    std::array< double, 5 > c{ 0.3, 1.2, -0.4, 0.8, 0.1 };
    ChebyshevSeries< 4 > g{ c };
    auto tay = tax::toTaylor( g );
    for ( double p : { -0.8, 0.0, 0.45 } ) EXPECT_NEAR( g.eval( p ), tay.eval( p ), 1e-11 );
}
