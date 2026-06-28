#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <tax/core/multi_index.hpp>
#include <tax/core/scheme/concept.hpp>
#include <tax/kernels/cauchy.hpp>
#include <tax/series/basis.hpp>

namespace tax
{

// ===========================================================================
// TaylorBasis — the monomial family  P_k(x) = x^k  (tensored over variables)
// ===========================================================================
//
// Wired onto the existing kernel/scheme layer: the product is the scheme's own
// Cauchy product (univariate unroll / multivariate stencil), so the carrier
// inherits the library's hot path. The univariate transcendental surface
// (operators.hpp) likewise delegates to the existing `series*` recurrences,
// which are themselves scheme-generic — so multivariate Taylor functions come
// for free.
// ===========================================================================

struct TaylorBasis
{
    static constexpr bool is_tax_basis = true;

    [[nodiscard]] static constexpr std::string_view name() noexcept { return "taylor"; }

    [[nodiscard]] static std::string term( int k )
    {
        if ( k == 0 ) return "1";
        if ( k == 1 ) return "x";
        return "x^" + std::to_string( k );
    }

    /// Truncated Cauchy (convolution) product, delegated to the scheme.
    template < typename T, typename Scheme >
    static constexpr void product( std::array< T, Scheme::nCoeff >& out,
                                   const std::array< T, Scheme::nCoeff >& a,
                                   const std::array< T, Scheme::nCoeff >& b ) noexcept
    {
        Scheme::template cauchyProduct< T >( out, a, b );
    }

    /// Evaluate  f(x) = Σ_k c_k x^α(k)  at the point vector x.
    template < typename T, typename Scheme >
    [[nodiscard]] static constexpr T eval(
        const std::array< T, Scheme::nCoeff >& c,
        const std::array< T, std::size_t( Scheme::vars ) >& x ) noexcept
    {
        constexpr int N = Scheme::order;
        constexpr int M = Scheme::vars;
        if constexpr ( Scheme::isUnivariate )
        {
            T r = c[std::size_t( N )];
            for ( int k = N - 1; k >= 0; --k ) r = r * x[0] + c[std::size_t( k )];
            return r;
        } else
        {
            // Power table pw[i][j] = x_i^j, then one multiply per monomial.
            std::array< std::array< T, std::size_t( N ) + 1 >, std::size_t( M ) > pw{};
            for ( int i = 0; i < M; ++i )
            {
                pw[std::size_t( i )][0] = T{ 1 };
                for ( int j = 1; j <= N; ++j )
                    pw[std::size_t( i )][std::size_t( j )] =
                        pw[std::size_t( i )][std::size_t( j - 1 )] * x[std::size_t( i )];
            }
            T r{};
            for ( std::size_t k = 0; k < Scheme::nCoeff; ++k )
            {
                if ( c[k] == T{ 0 } ) continue;
                const MultiIndex< M > alpha = Scheme::multiOf( k );
                T term = c[k];
                for ( int i = 0; i < M; ++i )
                    term *= pw[std::size_t( i )][std::size_t( alpha[std::size_t( i )] )];
                r += term;
            }
            return r;
        }
    }

    /// Coefficient-space derivative ∂/∂x_axis:  ∂(x^α)/∂x_axis = α_axis x^{α-e_axis}.
    template < typename T, typename Scheme >
    static constexpr void derivative( std::array< T, Scheme::nCoeff >& out,
                                      const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept
    {
        out = {};
        for ( std::size_t k = 0; k < Scheme::nCoeff; ++k )
        {
            if ( c[k] == T{ 0 } ) continue;
            MultiIndex< Scheme::vars > alpha = Scheme::multiOf( k );
            const int e = alpha[std::size_t( axis )];
            if ( e == 0 ) continue;
            alpha[std::size_t( axis )] = e - 1;
            out[Scheme::flatOf( alpha )] += c[k] * T( e );
        }
    }

    /// Coefficient-space integral ∫ dx_axis (constant 0). The term that would
    /// exceed the kept set is dropped by truncation.
    template < typename T, typename Scheme >
    static constexpr void integral( std::array< T, Scheme::nCoeff >& out,
                                    const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept
    {
        out = {};
        for ( std::size_t k = 0; k < Scheme::nCoeff; ++k )
        {
            if ( c[k] == T{ 0 } ) continue;
            MultiIndex< Scheme::vars > alpha = Scheme::multiOf( k );
            const int e = alpha[std::size_t( axis )];
            alpha[std::size_t( axis )] = e + 1;
            const std::size_t kk = Scheme::flatOf( alpha );
            if ( kk == Scheme::kNotInBox ) continue;
            out[kk] = c[k] / T( e + 1 );
        }
    }
};

static_assert( BasisPolicy< TaylorBasis > );

}  // namespace tax
