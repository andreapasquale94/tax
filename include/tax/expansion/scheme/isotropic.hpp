#pragma once

#include <array>
#include <cstddef>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/scheme/concept.hpp>
#include <tax/expansion/detail/cauchy.hpp>
#include <tax/expansion/detail/recurrence_stencil.hpp>
#include <type_traits>

namespace tax
{

/// The classic single-order graded-lex scheme: total degree <= N over M vars.
template < int N, int M >
struct IsotropicScheme
{
    static constexpr std::size_t nCoeff = numMonomials( N, M );
    static constexpr int order = N;
    static constexpr int vars = M;
    static constexpr bool isUnivariate = ( M == 1 );

    /// Sentinel returned by flatOf for multi-indices outside the kept set (|alpha| > N).
    static constexpr std::size_t kNotInBox = std::size_t( -1 );

    [[nodiscard]] static constexpr std::size_t flatOf( const MultiIndex< M >& a ) noexcept
    {
        if ( totalDegree( a ) > N ) return kNotInBox;
        return flatIndex< M >( a );
    }
    [[nodiscard]] static constexpr MultiIndex< M > multiOf( std::size_t k ) noexcept
    {
        return unflatIndex< M >( k );
    }

    /// Graded recurrence-row walker (M >= 2).
    template < class RowFn >
    static constexpr void forEachRecurrenceRow( RowFn&& fn ) noexcept
        requires( M >= 2 )
    {
        detail::kernels::forEachRecurrenceRow< N, M >( static_cast< RowFn&& >( fn ) );
    }

    /// Scheme-owned Cauchy product (unroll/stencil/loop dispatch).
    template < typename T >
    static constexpr void cauchyProduct( std::array< T, nCoeff >& out,
                                         const std::array< T, nCoeff >& a,
                                         const std::array< T, nCoeff >& b ) noexcept
    {
        detail::kernels::cauchyProduct< T, N, M >( out, a, b );
    }

    /// Scheme-owned self-product (M == 1: symmetric loop; M >= 2: cauchyProduct(f,f)).
    template < typename T >
    static constexpr void cauchySelfProduct( std::array< T, nCoeff >& out,
                                             const std::array< T, nCoeff >& f ) noexcept
    {
        if constexpr ( M == 1 )
        {
            out = {};
            for ( int d = 0; d <= N; ++d )
            {
                for ( int k = 0; k + k < d; ++k )
                    out[std::size_t( d )] += T{ 2 } * f[std::size_t( k )] * f[std::size_t( d - k )];
                if ( d % 2 == 0 )
                    out[std::size_t( d )] += f[std::size_t( d / 2 )] * f[std::size_t( d / 2 )];
            }
        } else
        {
            // Route through this scheme's own product so a Scheme that
            // overrides cauchyProduct is honoured.
            cauchyProduct< T >( out, f, f );
        }
    }
};

/// Trait: true iff `S` is an `IsotropicScheme<N, M>` instantiation.
template < typename S >
struct is_isotropic_scheme : std::false_type
{
};
template < int N, int M >
struct is_isotropic_scheme< IsotropicScheme< N, M > > : std::true_type
{
};
template < typename S >
inline constexpr bool is_isotropic_scheme_v = is_isotropic_scheme< S >::value;

}  // namespace tax
