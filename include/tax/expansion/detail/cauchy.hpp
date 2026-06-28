#pragma once

#include <tax/expansion/detail/stencil_config.hpp>
#include <tax/expansion/enumeration.hpp>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/storage/dense.hpp>

#if TAX_USE_UNROLL
#include <tax/expansion/detail/cauchy_unroll.hpp>
#endif
#if TAX_USE_STENCIL
#include <tax/expansion/detail/cauchy_stencil.hpp>
#endif

namespace tax::detail::kernels
{

/// Loop-based Cauchy (convolution) product over graded-lex monomials.
template < typename T, int N, int M >
constexpr void cauchyProductLoop( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a,
                                  const Coeffs< T, N, M >& b ) noexcept
{
    out = {};
    tax::forEachMonomial< M, N >( [&]( const MultiIndex< M >& alpha ) {
        const std::size_t i = flatIndex< M >( alpha );
        tax::forEachSubIndex< M >( alpha,
                                   [&]( const MultiIndex< M >& k, const MultiIndex< M >& s ) {
                                       out[i] += a[flatIndex< M >( k )] * b[flatIndex< M >( s )];
                                   } );
    } );
}

/// Public dispatch entry for the Cauchy product.
template < typename T, int N, int M >
constexpr void cauchyProduct( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a,
                              const Coeffs< T, N, M >& b ) noexcept
{
#if TAX_USE_UNROLL
    if constexpr ( M == 1 )
    {
        cauchyProductUnroll< T, N, M >( out, a, b );
        return;
    }
#endif
#if TAX_USE_STENCIL
    // Gate on the table budget: oversized (N, M) — e.g. many variables at high
    // order — fall back to the loop kernel instead of a hard compile error.
    if constexpr ( M >= 2 && cauchyStencilFits< N, M > )
    {
        // The stencil table is a runtime-initialised static, so it cannot be
        // used in constant evaluation; constexpr callers get the loop kernel.
        if !consteval
        {
            cauchyProductStencil< T, N, M >( out, a, b );
            return;
        }
    }
#endif
    cauchyProductLoop< T, N, M >( out, a, b );
}

}  // namespace tax::detail::kernels
