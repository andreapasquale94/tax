#include <gtest/gtest.h>

#include <tax/expansion/scheme.hpp>
#include <tax/expansion/detail/recurrence_stencil.hpp>
#include <tax/tax.hpp>

using tax::IsotropicScheme;
using tax::detail::kernels::RecurrenceEntry;

// IsotropicScheme<N,M> must reproduce the legacy numMonomials/forEachRecurrenceRow tables exactly.
TEST( IndexScheme, IsotropicMatchesLegacyShape )
{
    using S = IsotropicScheme< 5, 2 >;
    static_assert( S::nCoeff == tax::numMonomials( 5, 2 ) );
    static_assert( S::isUnivariate == false );
    static_assert( S::order == 5 );
    static_assert( IsotropicScheme< 4, 1 >::isUnivariate == true );
}

TEST( IndexScheme, IsotropicRecurrenceRowsMatchLegacy )
{
    constexpr int N = 5, M = 2;
    using S = IsotropicScheme< N, M >;

    // Collect rows from the legacy walker.
    std::vector< std::tuple< std::size_t, int, std::vector< std::array< std::uint32_t, 3 > > > >
        legacy;
    tax::detail::kernels::forEachRecurrenceRow< N, M >(
        [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
            std::vector< std::array< std::uint32_t, 3 > > es;
            for ( const auto& e : row ) es.push_back( { e.b_idx, e.g_idx, e.db } );
            legacy.push_back( { ai, d, es } );
        } );

    // Collect rows from the scheme.
    std::vector< std::tuple< std::size_t, int, std::vector< std::array< std::uint32_t, 3 > > > >
        viaScheme;
    S::forEachRecurrenceRow( [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
        std::vector< std::array< std::uint32_t, 3 > > es;
        for ( const auto& e : row ) es.push_back( { e.b_idx, e.g_idx, e.db } );
        viaScheme.push_back( { ai, d, es } );
    } );

    EXPECT_EQ( legacy, viaScheme );
}

TEST( IndexScheme, SchemeGenericKernelMatchesPublicSurface )
{
    constexpr int N = 6, M = 2;
    using S = tax::IsotropicScheme< N, M >;
    using TE = tax::TE< N, M >;

    typename TE::Input p{ 0.3, -0.2 };
    auto x = TE::template variable< 0 >( p );
    auto fx = exp( x );  // public surface

    std::array< double, S::nCoeff > a = x.coefficients();
    std::array< double, S::nCoeff > out{};
    tax::detail::kernels::seriesExp< double, S >( out, a );  // scheme-generic kernel

    for ( std::size_t k = 0; k < S::nCoeff; ++k ) EXPECT_DOUBLE_EQ( out[k], fx[k] );
}
