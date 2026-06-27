#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>

// Stencil table budget (bytes). Kept here as well as in <tax/kernels/cauchy.hpp>
// — both #ifndef-guarded with the identical value — so this header is usable
// regardless of include order (the two headers include each other). A project
// overriding it must define the same value project-wide (ODR).
#ifndef TAX_STENCIL_MAX_BYTES
#    define TAX_STENCIL_MAX_BYTES ( static_cast< std::size_t >( 64 ) << 20 )
#endif

namespace tax::detail::kernels
{

/// A single entry in the Cauchy stencil table (also used by the mixed-scheme stencil).
struct StencilEntry
{
    std::uint32_t out_idx;
    std::uint32_t a_idx;
    std::uint32_t b_idx;
};

/// One (a, b) operand-index pair contributing to a single output monomial.
struct StencilPair
{
    std::uint32_t a_idx;
    std::uint32_t b_idx;
};

/// True iff the (N, M) isotropic Cauchy stencil table fits within the configured budget.
template < int N, int M >
inline constexpr bool cauchyStencilFits =
    numMonomials( N, 2 * M ) * sizeof( StencilPair ) <= ( TAX_STENCIL_MAX_BYTES );

/// Compressed (CSR-like) table of all Cauchy product contributions for a given
/// (N, M) expansion, M >= 2. The monomial enumeration walks outputs in
/// flat-index order, so every output's contributions form a contiguous run:
/// `pairs[offsets[o] .. offsets[o + 1])` are exactly the (a, b) pairs summed
/// into `out[o]`. Storing the row offsets instead of a per-entry `out_idx`
/// shrinks the table by a third and lets the kernel accumulate each output in
/// a register, writing it once — no scattered read-modify-write on `out`.
template < int N, int M >
struct CauchyStencil
{
    static_assert( M >= 2 );

    static constexpr std::size_t nOut = numMonomials( N, M );
    /// Exact number of contributions: |{(beta, gamma) : |beta| + |gamma| <= N}|.
    static constexpr std::size_t kEntries = numMonomials( N, 2 * M );

    std::array< StencilPair, kEntries > pairs{};
    std::array< std::uint32_t, nOut + 1 > offsets{};

    constexpr CauchyStencil() noexcept
    {
        std::size_t n = 0;
        std::size_t o = 0;
        tax::forEachMonomial< M, N >( [this, &n, &o]( const MultiIndex< M >& alpha ) {
            // Outputs are visited in flat-index order, so `o` advances 0,1,2,...
            offsets[o++] = static_cast< std::uint32_t >( n );
            tax::forEachSubIndex< M >(
                alpha, [this, &n]( const MultiIndex< M >& k, const MultiIndex< M >& s ) {
                    pairs[n++] = StencilPair{ static_cast< std::uint32_t >( flatIndex< M >( k ) ),
                                              static_cast< std::uint32_t >( flatIndex< M >( s ) ) };
                } );
        } );
        offsets[o] = static_cast< std::uint32_t >( n );
        // n == kEntries and o == nOut by the graded-lex bijection.
    }
};

/// Precomputed-stencil Cauchy product for M >= 2. Not usable in constant evaluation (runtime-static
/// table).
template < typename T, int N, int M >
void cauchyProductStencil( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a,
                           const Coeffs< T, N, M >& b ) noexcept
    requires( M >= 2 )
{
    static const CauchyStencil< N, M > stencil{};
    const StencilPair* const pairs = stencil.pairs.data();
    for ( std::size_t o = 0; o < CauchyStencil< N, M >::nOut; ++o )
    {
        T acc{};
        const std::size_t end = stencil.offsets[o + 1];
        for ( std::size_t j = stencil.offsets[o]; j < end; ++j )
            acc += a[pairs[j].a_idx] * b[pairs[j].b_idx];
        out[o] = acc;
    }
}

}  // namespace tax::detail::kernels
