#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/tax.hpp>

// The new basis-generic carrier over TaylorBasis must reproduce the legacy
// TaylorExpansion bit-for-bit (same monomial layout, same kernels) — this is
// the "folding onto the Basis abstraction" made checkable.

TEST( SeriesUnify, TaylorBasisMatchesLegacyUnivariate )
{
    constexpr int N = 8;
    using NewE = tax::Expansion< double, tax::TaylorBasis, tax::IsotropicScheme< N, 1 > >;

    auto xn = NewE::variable();
    auto fn = exp( xn ) * sin( xn ) + 2.0 * xn;

    auto xl = tax::TaylorExpansion< double, tax::IsotropicScheme< N, 1 > >::variable( 0.0 );
    auto fl = exp( xl ) * sin( xl ) + 2.0 * xl;

    for ( std::size_t k = 0; k < NewE::nCoefficients; ++k )
        EXPECT_NEAR( fn[k], fl[k], 1e-14 ) << "coeff " << k;
}

TEST( SeriesUnify, TaylorBasisMatchesLegacyMultivariate )
{
    constexpr int N = 4, M = 2;
    using NewE = tax::Expansion< double, tax::TaylorBasis, tax::IsotropicScheme< N, M > >;
    using Leg = tax::TaylorExpansion< double, tax::IsotropicScheme< N, M > >;

    auto xn = NewE::variable< 0 >();
    auto yn = NewE::variable< 1 >();
    auto fn = exp( xn * yn ) + xn * xn - 3.0 * yn;

    std::array< double, 2 > p{ 0.0, 0.0 };
    auto xl = Leg::variable< 0 >( p );
    auto yl = Leg::variable< 1 >( p );
    auto fl = exp( xl * yl ) + xl * xl - 3.0 * yl;

    for ( std::size_t k = 0; k < NewE::nCoefficients; ++k )
        EXPECT_NEAR( fn[k], fl[k], 1e-13 ) << "coeff " << k;
}

TEST( SeriesUnify, IoStreamsInBasis )
{
    auto x = tax::TaylorSeries< 3 >::variable();
    auto f = 1.0 + 2.0 * x;  // monomial labels
    EXPECT_EQ( tax::to_string( f ), "1 + 2*x" );

    tax::ChebyshevSeries< 2 > g{ std::array< double, 3 >{ 1.0, 0.0, 3.0 } };
    EXPECT_EQ( tax::to_string( g ), "1 + 3*T_2" );

    tax::TaylorSeries< 2 > zero{};
    EXPECT_EQ( tax::to_string( zero ), "0" );
}
