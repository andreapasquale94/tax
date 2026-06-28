#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <tax/tax.hpp>

// After the destructive merge, `TaylorExpansion` IS `Expansion` over TaylorBasis
// — one class, not two that happen to agree.
static_assert(
    std::is_same_v< tax::TaylorExpansion< double, tax::IsotropicScheme< 5, 2 > >,
                    tax::Expansion< double, tax::TaylorBasis, tax::IsotropicScheme< 5, 2 > > > );
static_assert(
    std::is_same_v< tax::TE< 4 >,
                    tax::Expansion< double, tax::TaylorBasis, tax::IsotropicScheme< 4, 1 > > > );

// The basis-generic carrier over TaylorBasis must still reproduce the same
// monomial mathematics through the shared kernels.

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
    // The generic series IO labels terms in the basis (here Chebyshev).
    tax::ChebyshevExpansion< 2 > g{ std::array< double, 3 >{ 1.0, 0.0, 3.0 } };
    EXPECT_EQ( tax::to_string( g ), "1 + 3*T_2(x₀)" );

    tax::ChebyshevExpansion< 2 > zero{};
    EXPECT_EQ( tax::to_string( zero ), "0" );
}

// Multivariate orthogonal printing: each variable carries its own basis label,
// joined by '*'. Build the coefficient array directly and pin one cross term.
TEST( SeriesUnify, MultivariateOrthogonalPrints )
{
    using E = tax::ChebyshevExpansion< 3, 2 >;  // order 3, 2 vars
    std::array< double, E::nCoefficients > a{};
    a[0] = 2.0;  // constant
    tax::MultiIndex< 2 > m21{};
    m21[0] = 2;  // x₀ degree 2
    m21[1] = 1;  // x₁ degree 1
    a[tax::flatIndex< 2 >( m21 )] = 5.0;
    E g{ a };
    EXPECT_EQ( tax::to_string( g ), "2 + 5*T_2(x₀)*T_1(x₁)" );
}

// Named-over-orthogonal printing: the variable symbol is the axis name.
TEST( SeriesUnify, NamedOrthogonalUsesAxisName )
{
    auto u = tax::variable< "u", 4, tax::ChebyshevBasis >( 0.0 );
    const auto s = tax::to_string( tax::exp( u ) );
    EXPECT_NE( s.find( "(u)" ), std::string::npos );
}
