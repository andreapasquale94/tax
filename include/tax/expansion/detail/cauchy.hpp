#pragma once

#include <tax/expansion/enumeration.hpp>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/storage/dense.hpp>

// Kernel dispatch configuration. Defaults ON here (not in the build system)
// so every consumer gets the fast paths regardless of how the headers are
// consumed. A project may pre-define either macro to 0 to fall back to the
// loop kernel, but the value MUST be identical in every translation unit
// linked together — differing values change inline definitions (ODR).
#ifndef TAX_USE_UNROLL
#    define TAX_USE_UNROLL 1
#endif
#ifndef TAX_USE_STENCIL
#    define TAX_USE_STENCIL 1
#endif
// Upper bound (bytes) on any precomputed stencil table (Cauchy or recurrence).
// Beyond this the dispatch falls back to the constexpr loop kernel instead of
// materialising a huge static table — which would otherwise be a hard compile
// error for many-variable, high-order expansions. Configured in-header like the
// other knobs; a project may pre-define it, but the value must be identical in
// every translation unit (ODR).
#ifndef TAX_STENCIL_MAX_BYTES
#    define TAX_STENCIL_MAX_BYTES ( static_cast< std::size_t >( 64 ) << 20 )
#endif

#if TAX_USE_UNROLL
#    include <tax/expansion/detail/cauchy_unroll.hpp>
#endif
#if TAX_USE_STENCIL
#    include <tax/expansion/detail/cauchy_stencil.hpp>
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
