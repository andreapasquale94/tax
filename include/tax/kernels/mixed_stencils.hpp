#pragma once

// Box-filtered Cauchy + recurrence stencils for MixedScheme: precomputed
// (out, a, b) / (b, g, db) tables for the box product. Tables are compile-time
// std::array (no heap), built in graded (ascending total degree) output order.
//
// This header does NOT include tax/core/scheme/mixed.hpp; that header includes
// this one up front (before defining MixedScheme) and relies on the forward
// declaration below to break the cycle. The stencil structs reference
// MixedScheme<Groups...> only inside template bodies instantiated later (when
// the static const stencil is first accessed), by which point the class is
// complete.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/kernels/cauchy_stencil.hpp>
#include <tax/kernels/recurrence_stencil.hpp>

namespace tax
{
/// Forward declaration; MixedScheme is fully defined in tax/core/scheme/mixed.hpp.
template < typename... Groups >
struct MixedScheme;
}  // namespace tax

namespace tax::detail::kernels
{

/// Cauchy entry count for MixedScheme<Groups...>: Π_g numMonomials(order_g, 2*dim_g).
template < typename... Groups >
inline constexpr std::size_t mixedCauchyEntries =
    ( numMonomials( Groups::order, 2 * Groups::dim ) * ... );

/// Recurrence entry count = Cauchy count − nCoeff (drop |β|==0 row per output).
template < typename... Groups >
inline constexpr std::size_t mixedRecurrenceEntries =
    mixedCauchyEntries< Groups... > - tax::MixedScheme< Groups... >::nCoeff;

/// True iff the box Cauchy stencil for MixedScheme<Groups...> fits the configured budget.
template < typename... Groups >
inline constexpr bool mixedCauchyStencilFits =
    mixedCauchyEntries< Groups... > * sizeof( StencilPair ) <= ( TAX_STENCIL_MAX_BYTES );

/// True iff the box recurrence stencil for MixedScheme<Groups...> fits the configured budget.
template < typename... Groups >
inline constexpr bool mixedRecurrenceStencilFits =
    mixedRecurrenceEntries< Groups... > * sizeof( RecurrenceEntry ) <= ( TAX_STENCIL_MAX_BYTES );

/// Compressed (CSR-like) table for the box Cauchy product of MixedScheme<Groups...>,
/// in graded output order (ai = 0 … nCoeff-1). Each output's (a, b) operand pairs
/// occupy the contiguous run `pairs[offsets[ai] .. offsets[ai + 1])`, so the kernel
/// accumulates each output in a register and writes it once — no scattered
/// read-modify-write, and a third less memory than a per-entry out_idx.
template < typename... Groups >
struct MixedBoxCauchyStencil
{
    static constexpr std::size_t kNCoeff = tax::MixedScheme< Groups... >::nCoeff;
    static constexpr std::size_t kEntries = mixedCauchyEntries< Groups... >;

    std::array< StencilPair, kEntries > pairs{};
    std::array< std::uint32_t, kNCoeff + 1 > offsets{};

    using Scheme = tax::MixedScheme< Groups... >;
    static constexpr int V = Scheme::vars;

    constexpr MixedBoxCauchyStencil() noexcept
    {
        std::size_t n = 0;
        // Outputs in graded order; within each output α, β iterates by ascending
        // flat index (a brute-force (i,j) double-loop walk). The constexpr fallback
        // in MixedScheme::cauchyProduct uses a forEachSubIndex walk instead, so the
        // two paths sum in different orders and agree only to round-off — the same
        // loop-vs-stencil contract documented for the isotropic kernels.
        for ( std::size_t ai = 0; ai < kNCoeff; ++ai )
        {
            offsets[ai] = static_cast< std::uint32_t >( n );
            const MultiIndex< V > alpha = Scheme::multiOf( ai );
            for ( std::size_t bi = 0; bi < kNCoeff; ++bi )
            {
                const MultiIndex< V > beta = Scheme::multiOf( bi );
                // β must be componentwise ≤ α (so γ = α − β has non-negative components).
                bool ok = true;
                for ( int v = 0; v < V; ++v )
                    if ( beta[std::size_t( v )] > alpha[std::size_t( v )] )
                    {
                        ok = false;
                        break;
                    }
                if ( !ok ) continue;
                MultiIndex< V > gamma{};
                for ( int v = 0; v < V; ++v )
                    gamma[std::size_t( v )] = alpha[std::size_t( v )] - beta[std::size_t( v )];
                pairs[n++] = StencilPair{ static_cast< std::uint32_t >( bi ),
                                          static_cast< std::uint32_t >( Scheme::flatOf( gamma ) ) };
            }
        }
        offsets[kNCoeff] = static_cast< std::uint32_t >( n );
        // n == kEntries: each (β,γ) pair with β+γ ∈ Box contributes one entry.
    }
};

/// Decomposition table for the degree-by-degree recurrence over MixedScheme<Groups...>, in graded
/// output order; drops the |β|==0 row per output (db >= 1 always).
template < typename... Groups >
struct MixedBoxRecurrenceStencil
{
    static constexpr std::size_t kNCoeff = tax::MixedScheme< Groups... >::nCoeff;
    static constexpr std::size_t kEntries = mixedRecurrenceEntries< Groups... >;
    // Instantiated only when mixedRecurrenceStencilFits<Groups...> holds (the
    // forEachRecurrenceRow dispatch gates on it), so no size static_assert here:
    // oversized boxes take the constexpr loop fallback instead of a hard error.

    std::array< RecurrenceEntry, kEntries > entries{};
    /// Row bounds: recurrence entries for output ai live in [row[ai], row[ai+1]).
    std::array< std::uint32_t, kNCoeff + 1 > row{};
    /// Total degree |α| for each output flat index.
    std::array< std::int32_t, kNCoeff > degree{};

    using Scheme = tax::MixedScheme< Groups... >;
    static constexpr int V = Scheme::vars;

    constexpr MixedBoxRecurrenceStencil() noexcept
    {
        std::size_t n = 0;
        for ( std::size_t ai = 0; ai < kNCoeff; ++ai )
        {
            row[ai] = static_cast< std::uint32_t >( n );
            const MultiIndex< V > alpha = Scheme::multiOf( ai );
            degree[ai] = totalDegree( alpha );

            // Enumerate β ≤ α, skip |β|==0.
            forEachSubIndex< V >(
                alpha, [&]( const MultiIndex< V >& beta, const MultiIndex< V >& gamma ) {
                    const int db = totalDegree( beta );
                    if ( db == 0 ) return;
                    entries[n++] =
                        RecurrenceEntry{ static_cast< std::uint32_t >( Scheme::flatOf( beta ) ),
                                         static_cast< std::uint32_t >( Scheme::flatOf( gamma ) ),
                                         static_cast< std::uint32_t >( db ) };
                } );
        }
        row[kNCoeff] = static_cast< std::uint32_t >( n );
    }
};

/// Shared per-MixedScheme stencil accessor (runtime static — one instance per type).
template < typename... Groups >
[[nodiscard]] inline const MixedBoxCauchyStencil< Groups... >& mixedBoxCauchyStencil() noexcept
{
    static const MixedBoxCauchyStencil< Groups... > s{};
    return s;
}

/// Shared per-MixedScheme stencil accessor (runtime static — one instance per type).
template < typename... Groups >
[[nodiscard]] inline const MixedBoxRecurrenceStencil< Groups... >&
mixedBoxRecurrenceStencil() noexcept
{
    static const MixedBoxRecurrenceStencil< Groups... > s{};
    return s;
}

}  // namespace tax::detail::kernels
