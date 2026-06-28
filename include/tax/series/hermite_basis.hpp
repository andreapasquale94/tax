#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <tax/series/basis.hpp>
#include <tax/series/ortho.hpp>

namespace tax
{

// ===========================================================================
// HermiteBasis — probabilists' Hermite polynomials He_n (tensored over vars)
// ===========================================================================
//
// Recurrence:  He_0 = 1, He_1 = x,  He_{n+1} = x He_n − n He_{n-1},
// i.e.  x He_n = He_{n+1} + n He_{n-1}.  Orthogonal w.r.t. the weight
// e^{−x²/2} on ℝ (the standard-normal density up to a constant).
// Closed-form calculus:
//   f' :  He_n' = n He_{n-1}  ⇒  b_m = (m+1) a_{m+1}
//   ∫f :  ∫He_n = He_{n+1}/(n+1)  ⇒  B_m = a_{m-1}/m   (constant 0)
// ===========================================================================

struct HermiteBasis
{
    static constexpr bool is_tax_basis = true;

    [[nodiscard]] static constexpr std::string_view name() noexcept { return "hermite"; }

    [[nodiscard]] static std::string term( int k )
    {
        if ( k == 0 ) return "1";
        return "He_" + std::to_string( k );
    }

    /// x He_n = α_n He_{n+1} + β_n He_n + γ_n He_{n-1}.
    template < typename T >
    static constexpr void xmul( int n, T& alpha, T& beta, T& gamma ) noexcept
    {
        alpha = T{ 1 };
        beta = T{ 0 };
        gamma = T( n );
    }

    template < typename T, typename Scheme >
    static constexpr void product( std::array< T, Scheme::nCoeff >& out,
                                   const std::array< T, Scheme::nCoeff >& a,
                                   const std::array< T, Scheme::nCoeff >& b ) noexcept
    {
        detail::orthoProduct< HermiteBasis, T, Scheme >( out, a, b );
    }

    template < typename T, typename Scheme >
    [[nodiscard]] static constexpr T eval(
        const std::array< T, Scheme::nCoeff >& c,
        const std::array< T, std::size_t( Scheme::vars ) >& x ) noexcept
    {
        return detail::orthoEval< HermiteBasis, T, Scheme >( c, x );
    }

    template < typename T, typename Scheme >
    static constexpr void derivative( std::array< T, Scheme::nCoeff >& out,
                                      const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept
    {
        out = {};
        detail::forEachFiber< T, Scheme >(
            c, axis,
            []( const std::array< T, std::size_t( Scheme::order ) + 1 >& a, int L,
                std::array< T, std::size_t( Scheme::order ) + 1 >& b ) {
                b = {};
                for ( int m = 0; m + 1 <= L - 1; ++m )
                    b[std::size_t( m )] = T( m + 1 ) * a[std::size_t( m + 1 )];
            },
            out );
    }

    template < typename T, typename Scheme >
    static constexpr void integral( std::array< T, Scheme::nCoeff >& out,
                                    const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept
    {
        out = {};
        detail::forEachFiber< T, Scheme >(
            c, axis,
            []( const std::array< T, std::size_t( Scheme::order ) + 1 >& a, int L,
                std::array< T, std::size_t( Scheme::order ) + 1 >& b ) {
                b = {};
                for ( int m = 1; m <= L - 1; ++m )
                    b[std::size_t( m )] = a[std::size_t( m - 1 )] / T( m );
            },
            out );
    }
};

static_assert( Basis< HermiteBasis > );

}  // namespace tax
