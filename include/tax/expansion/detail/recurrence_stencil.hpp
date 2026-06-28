#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tax/expansion/enumeration.hpp>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/detail/cauchy.hpp>  // TAX_USE_STENCIL configuration

namespace tax::detail::kernels
{

/// A single decomposition entry of the recurrence stencil.
struct RecurrenceEntry
{
    std::uint32_t b_idx;
    std::uint32_t g_idx;
    std::uint32_t db;
};

/// True iff the (N, M) recurrence stencil table fits the configured budget.
template < int N, int M >
inline constexpr bool recurrenceStencilFits =
    ( numMonomials( N, 2 * M ) - numMonomials( N, M ) ) * sizeof( RecurrenceEntry ) <=
    ( TAX_STENCIL_MAX_BYTES );

/// Tight upper bound on the entry count of any single recurrence row.
/// A row for output α holds the β with 0 <= β <= α (componentwise), so it has
/// Π(α_i + 1) - 1 entries. Over all |α| <= N that product is maximised by the
/// most balanced exponents (fixed term count, bounded sum), so the bound below
/// is exact and independent of nCoeff — keeping the loop-fallback buffer small.
template < int N, int M >
[[nodiscard]] constexpr std::size_t maxRecurrenceRow() noexcept
{
    const int total = N + M;  // Σ(α_i + 1) when |α| == N
    const int q = total / M;
    const int r = total % M;
    std::size_t p = 1;
    for ( int i = 0; i < r; ++i ) p *= std::size_t( q + 1 );
    for ( int i = 0; i < M - r; ++i ) p *= std::size_t( q );
    return p;  // = max Π(α_i + 1) >= (row entries) + 1
}

/// Decomposition table driving the degree-by-degree recurrence kernels for M >= 2.
template < int N, int M >
struct RecurrenceStencil
{
    static_assert( M >= 2 );

    static constexpr std::size_t NC = numMonomials( N, M );
    static constexpr std::size_t kEntries = numMonomials( N, 2 * M ) - NC;

    std::array< RecurrenceEntry, kEntries > entries{};
    /// Row bounds: entries for output index ai live in [row[ai], row[ai+1]).
    std::array< std::uint32_t, NC + 1 > row{};
    /// Total degree |alpha| per output index.
    std::array< std::int32_t, NC > degree{};

    constexpr RecurrenceStencil() noexcept
    {
        std::size_t n = 0;
        std::size_t ai = 0;
        tax::forEachMonomial< M, N >( [this, &n, &ai]( const MultiIndex< M >& alpha ) {
            row[ai] = static_cast< std::uint32_t >( n );
            degree[ai] = totalDegree( alpha );
            tax::forEachSubIndex< M >(
                alpha, [this, &n]( const MultiIndex< M >& beta, const MultiIndex< M >& gamma ) {
                    int db = 0;
                    for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                    if ( db == 0 ) return;
                    entries[n++] =
                        RecurrenceEntry{ static_cast< std::uint32_t >( flatIndex< M >( beta ) ),
                                         static_cast< std::uint32_t >( flatIndex< M >( gamma ) ),
                                         static_cast< std::uint32_t >( db ) };
                } );
            ++ai;
        } );
        row[NC] = static_cast< std::uint32_t >( n );
        // n == kEntries by the bijection documented above.
    }
};

/// Shared per-(N, M) table instance (kept out of the RowFn-templated
/// walker below so each kernel instantiation reuses the same static).
template < int N, int M >
[[nodiscard]] inline const RecurrenceStencil< N, M >& recurrenceStencil() noexcept
{
    static const RecurrenceStencil< N, M > s{};
    return s;
}

/// Walk all recurrence rows (M >= 2) in graded-lex order, so each output sees its lower-degree
/// dependencies already computed. Loop and stencil paths enumerate the same rows.
template < int N, int M, class RowFn >
constexpr void forEachRecurrenceRow( RowFn&& fn ) noexcept
{
    static_assert( M >= 2 );
    constexpr std::size_t NC = numMonomials( N, M );

#if TAX_USE_STENCIL
    // Oversized tables (many variables at high order) fall back to the loop
    // enumeration below instead of a hard compile error.
    if constexpr ( recurrenceStencilFits< N, M > )
        if !consteval
        {
            const RecurrenceStencil< N, M >& st = recurrenceStencil< N, M >();
            for ( std::size_t ai = 1; ai < NC; ++ai )
            {
                fn( ai, int( st.degree[ai] ),
                    std::span< const RecurrenceEntry >( st.entries.data() + st.row[ai],
                                                        st.entries.data() + st.row[ai + 1] ) );
            }
            return;
        }
#endif
    // Per-row buffer: one row never exceeds maxRecurrenceRow(), which is bounded
    // independently of nCoeff, so this stays small even for many-variable cases.
    std::array< RecurrenceEntry, maxRecurrenceRow< N, M >() > buf{};
    std::size_t ai = 0;
    tax::forEachMonomial< M, N >( [&]( const MultiIndex< M >& alpha ) {
        const std::size_t i = ai++;
        if ( i == 0 ) return;  // alpha == 0 has no |beta| >= 1 decompositions
        std::size_t n = 0;
        tax::forEachSubIndex< M >(
            alpha, [&]( const MultiIndex< M >& beta, const MultiIndex< M >& gamma ) {
                int db = 0;
                for ( int q = 0; q < M; ++q ) db += beta[std::size_t( q )];
                if ( db == 0 ) return;
                buf[n++] = RecurrenceEntry{ static_cast< std::uint32_t >( flatIndex< M >( beta ) ),
                                            static_cast< std::uint32_t >( flatIndex< M >( gamma ) ),
                                            static_cast< std::uint32_t >( db ) };
            } );
        fn( i, totalDegree( alpha ), std::span< const RecurrenceEntry >( buf.data(), n ) );
    } );
}

}  // namespace tax::detail::kernels
