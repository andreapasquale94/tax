#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <tax/kernels/cauchy.hpp>
#include <tax/series/basis.hpp>

namespace tax
{

// ===========================================================================
// TaylorBasis — the monomial family  P_k(x) = x^k
// ===========================================================================
//
// This is the classical Taylor / power basis. It is wired straight onto the
// existing `tax` kernel layer: the product reuses the unrolled/loop Cauchy
// convolution kernel, so the new `Series` carrier inherits the library's hot
// path for free and proves the basis abstraction wraps the existing engine
// rather than duplicating it.
//
// Note: the carrier `Series` treats a Taylor expansion as a polynomial in the
// absolute variable `x` (centre 0), so `eval(x)` returns f(x). The classic
// `tax::TaylorExpansion` displacement-from-x0 view is the special case of
// expanding about a chosen centre.
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

    /// Truncated Cauchy (convolution) product, delegated to the shared kernel.
    template < typename T, int N >
    static constexpr void product( std::array< T, std::size_t( N ) + 1 >& out,
                                   const std::array< T, std::size_t( N ) + 1 >& a,
                                   const std::array< T, std::size_t( N ) + 1 >& b ) noexcept
    {
        detail::kernels::cauchyProduct< T, N, 1 >( out, a, b );
    }

    /// Horner evaluation of  f(x) = sum_k c_k x^k.
    template < typename T, int N >
    [[nodiscard]] static constexpr T eval( const std::array< T, std::size_t( N ) + 1 >& c,
                                           T x ) noexcept
    {
        T r = c[std::size_t( N )];
        for ( int k = N - 1; k >= 0; --k ) r = r * x + c[std::size_t( k )];
        return r;
    }

    /// Coefficient-space derivative:  (x^k)' = k x^{k-1}.
    template < typename T, int N >
    static constexpr void derivative( std::array< T, std::size_t( N ) + 1 >& out,
                                      const std::array< T, std::size_t( N ) + 1 >& c ) noexcept
    {
        out = {};
        for ( int k = 1; k <= N; ++k ) out[std::size_t( k - 1 )] = T( k ) * c[std::size_t( k )];
    }

    /// Coefficient-space indefinite integral (constant of integration 0):
    /// integral(x^k) = x^{k+1} / (k+1). The degree-N term would land at N+1 and
    /// is dropped by truncation.
    template < typename T, int N >
    static constexpr void integral( std::array< T, std::size_t( N ) + 1 >& out,
                                    const std::array< T, std::size_t( N ) + 1 >& c ) noexcept
    {
        out = {};
        for ( int k = 1; k <= N; ++k ) out[std::size_t( k )] = c[std::size_t( k - 1 )] / T( k );
    }
};

static_assert( Basis< TaylorBasis > );

}  // namespace tax
