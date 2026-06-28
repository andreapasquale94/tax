#pragma once

// MixedScheme: an anisotropic ("box") IndexScheme.
//
// Each variable group g has its own truncation order order_g over its own
// dim_g variables. A monomial is kept iff the total degree of every per-group
// block is within that group's order — i.e. the kept set is the *product* of
// per-group simplices (a box), not a single joint simplex. There is no joint
// total-degree cap (order == Σ order_g).
//
// Layout (graded mixed-radix): iterate total degree d = 0 … Σ order_g; within
// d iterate the per-group degree tuples (d_0,…,d_{G-1}) with Σ d_g = d and
// d_g ≤ order_g in lexicographic order (group 0 most significant); within a
// tuple iterate the per-group degree-d_g monomials in graded-lex as a
// mixed-radix product (group 0 outermost). Flat indices 0,1,2,… follow that
// visitation order. flatOf / multiOf are the inverse maps — pure constexpr
// index math. The cauchyProduct / forEachRecurrenceRow members use the
// runtime-static box stencils (TAX_USE_STENCIL) with a constexpr on-the-fly
// fallback for constant evaluation.

#include <array>
#include <cstddef>
#include <span>
#include <tax/expansion/detail/mixed_stencils.hpp>
#include <tax/expansion/detail/stencil_config.hpp>
#include <tax/expansion/enumeration.hpp>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/scheme/concept.hpp>

namespace tax
{

/// One anisotropic variable group: `Dim` variables truncated at total order `Order`.
template < int Dim, int Order >
struct Group
{
    static_assert( Dim >= 1, "Group Dim must be >= 1" );
    static_assert( Order >= 0, "Group Order must be >= 0" );
    static constexpr int dim = Dim;
    static constexpr int order = Order;
};

namespace detail
{

/// Per-group descriptor used by the runtime (constexpr) layout loops.
struct MixedGroupDesc
{
    int dim;        ///< number of variables in the group
    int order;      ///< per-group truncation order
    int varOffset;  ///< first variable index of this block in α
};

/// Number of degree-exactly-`d` monomials over `m` variables (within-degree block size).
constexpr std::size_t withinDegreeCount( int d, int m ) noexcept
{
    return binom( d + m - 1, m - 1 );
}

}  // namespace detail

/// Anisotropic box index scheme over a list of `Group`s.
template < typename... Groups >
struct MixedScheme
{
    static constexpr int groupCount = static_cast< int >( sizeof...( Groups ) );
    static_assert( groupCount >= 1, "MixedScheme requires at least one Group" );

    static constexpr int vars = ( Groups::dim + ... );
    static constexpr int order = ( Groups::order + ... );
    static constexpr bool isUnivariate = false;

    static constexpr std::size_t kNotInBox = std::size_t( -1 );

   private:
    /// Per-group descriptors {dim, order, varOffset}, built from the pack.
    static constexpr std::array< detail::MixedGroupDesc, std::size_t( groupCount ) > kGroups = [] {
        std::array< detail::MixedGroupDesc, std::size_t( groupCount ) > g{};
        int idx = 0;
        int offset = 0;
        (
            [&] {
                g[std::size_t( idx )] =
                    detail::MixedGroupDesc{ Groups::dim, Groups::order, offset };
                offset += Groups::dim;
                ++idx;
            }(),
            ... );
        return g;
    }();

    /// Global flat index of the first degree-`d` monomial over `m` variables.
    static constexpr std::size_t degreeBlockBase( int d, int m ) noexcept
    {
        return d == 0 ? std::size_t( 0 ) : numMonomials( d - 1, m );
    }

   public:
    /// Box size = product of per-group simplex sizes.
    static constexpr std::size_t nCoeff = ( numMonomials( Groups::order, Groups::dim ) * ... );

    /// Flat index of `a`, or `kNotInBox` if any group block exceeds its order.
    [[nodiscard]] static constexpr std::size_t flatOf( const MultiIndex< vars >& a ) noexcept
    {
        // Decompose α into per-group degree d_g and within-degree rank rank_g.
        std::array< int, std::size_t( groupCount ) > deg{};
        std::array< std::size_t, std::size_t( groupCount ) > rank{};
        int total = 0;
        for ( int g = 0; g < groupCount; ++g )
        {
            const detail::MixedGroupDesc& gd = kGroups[std::size_t( g )];
            const std::size_t base = subBlockFlat( a, gd );
            if ( base == kNotInBox ) return kNotInBox;  // out of box for this group
            const int dg = subBlockDegree( a, gd );
            deg[std::size_t( g )] = dg;
            // within-degree rank = global flat (over dim_g vars) minus the
            // degree-d_g block base.
            rank[std::size_t( g )] = base - degreeBlockBase( dg, gd.dim );
            total += dg;
        }

        std::size_t flat = 0;

        // (1) all in-box monomials of total degree strictly less than `total`.
        for ( int d = 0; d < total; ++d ) flat += degreeTotalCount( d );

        // (2) all valid tuples at degree `total` that precede `deg` lexicographically.
        flat += precedingTupleCount( deg, total );

        // (3) mixed-radix offset of (rank_0,…) within this tuple (group 0 outermost).
        std::size_t offset = 0;
        for ( int g = 0; g < groupCount; ++g )
        {
            const detail::MixedGroupDesc& gd = kGroups[std::size_t( g )];
            const std::size_t radix = detail::withinDegreeCount( deg[std::size_t( g )], gd.dim );
            offset = offset * radix + rank[std::size_t( g )];
        }
        flat += offset;
        return flat;
    }

    /// Multi-index occupying flat slot `k` (`k` assumed in [0, nCoeff)).
    [[nodiscard]] static constexpr MultiIndex< vars > multiOf( std::size_t k ) noexcept
    {
        // (1) find the total degree band that contains k.
        int total = 0;
        std::size_t base = 0;
        for ( ;; ++total )
        {
            const std::size_t cnt = degreeTotalCount( total );
            if ( k < base + cnt ) break;
            base += cnt;
        }
        std::size_t within = k - base;  // offset inside the degree-`total` band

        // (2) walk valid degree tuples (lex order) until `within` lands in one.
        std::array< int, std::size_t( groupCount ) > deg{};
        forEachTuple( total,
                      [&]( const std::array< int, std::size_t( groupCount ) >& t,
                           std::size_t block ) -> bool {
                          if ( within < block )
                          {
                              deg = t;
                              return true;  // stop
                          }
                          within -= block;
                          return false;
                      } );

        // (3) decode the mixed-radix offset (group 0 outermost) into per-group ranks.
        std::array< std::size_t, std::size_t( groupCount ) > rank{};
        for ( int g = groupCount - 1; g >= 0; --g )
        {
            const detail::MixedGroupDesc& gd = kGroups[std::size_t( g )];
            const std::size_t radix = detail::withinDegreeCount( deg[std::size_t( g )], gd.dim );
            rank[std::size_t( g )] = within % radix;
            within /= radix;
        }

        // (4) reassemble α from per-group (degree, rank) via unflatIndex.
        MultiIndex< vars > a{};
        for ( int g = 0; g < groupCount; ++g )
        {
            const detail::MixedGroupDesc& gd = kGroups[std::size_t( g )];
            const std::size_t gflat =
                degreeBlockBase( deg[std::size_t( g )], gd.dim ) + rank[std::size_t( g )];
            writeSubBlock( a, gd, gflat );
        }
        return a;
    }

    // IndexScheme member surface — required by the scheme-generic kernels.

    /// Box-filtered Cauchy product (precomputed stencil at runtime; on-the-fly sub-index
    /// enumeration in constant evaluation).
    template < typename T >
    static constexpr void cauchyProduct( std::array< T, nCoeff >& out,
                                         const std::array< T, nCoeff >& a,
                                         const std::array< T, nCoeff >& b ) noexcept
    {
#if TAX_USE_STENCIL
        // Oversized box stencils (many variables at high order) fall back to the
        // constexpr enumeration below instead of a hard compile error.
        if constexpr ( detail::kernels::mixedCauchyStencilFits< Groups... > )
            if !consteval
            {
                const auto& st = detail::kernels::mixedBoxCauchyStencil< Groups... >();
                const detail::kernels::StencilPair* const pairs = st.pairs.data();
                for ( std::size_t ai = 0; ai < nCoeff; ++ai )
                {
                    T acc{};
                    const std::size_t end = st.offsets[ai + 1];
                    for ( std::size_t j = st.offsets[ai]; j < end; ++j )
                        acc += a[pairs[j].a_idx] * b[pairs[j].b_idx];
                    out[ai] = acc;
                }
                return;
            }
#endif
        // Constexpr fallback: enumerate sub-indices on the fly.
        out = {};
        for ( std::size_t ai = 0; ai < nCoeff; ++ai )
        {
            const MultiIndex< vars > alpha = multiOf( ai );
            forEachSubIndex< vars >(
                alpha, [&]( const MultiIndex< vars >& beta, const MultiIndex< vars >& gamma ) {
                    out[ai] += a[flatOf( beta )] * b[flatOf( gamma )];
                } );
        }
    }

    /// Box-filtered self-product: delegates to cauchyProduct(f, f).
    template < typename T >
    static constexpr void cauchySelfProduct( std::array< T, nCoeff >& out,
                                             const std::array< T, nCoeff >& f ) noexcept
    {
        cauchyProduct< T >( out, f, f );
    }

    /// Graded recurrence-row walker fn(ai, degree, span<RecurrenceEntry>): stencil at
    /// runtime, on-the-fly in constant evaluation.
    template < class RowFn >
    static constexpr void forEachRecurrenceRow( RowFn&& fn ) noexcept
    {
#if TAX_USE_STENCIL
        // Oversized box stencils (many variables at high order) fall back to the
        // constexpr enumeration below instead of a hard compile error.
        if constexpr ( detail::kernels::mixedRecurrenceStencilFits< Groups... > )
            if !consteval
            {
                const auto& st = detail::kernels::mixedBoxRecurrenceStencil< Groups... >();
                for ( std::size_t ai = 1; ai < nCoeff; ++ai )
                {
                    fn( ai, int( st.degree[ai] ),
                        std::span< const detail::kernels::RecurrenceEntry >(
                            st.entries.data() + st.row[ai], st.entries.data() + st.row[ai + 1] ) );
                }
                return;
            }
#endif
        // Constexpr fallback: build each row on the fly (same entries, no precomputed table).
        std::array< detail::kernels::RecurrenceEntry, nCoeff > buf{};
        for ( std::size_t ai = 1; ai < nCoeff; ++ai )
        {
            const MultiIndex< vars > alpha = multiOf( ai );
            const int deg = totalDegree( alpha );
            std::size_t n = 0;
            forEachSubIndex< vars >(
                alpha, [&]( const MultiIndex< vars >& beta, const MultiIndex< vars >& gamma ) {
                    const int db = totalDegree( beta );
                    if ( db == 0 ) return;
                    buf[n++] = detail::kernels::RecurrenceEntry{
                        static_cast< std::uint32_t >( flatOf( beta ) ),
                        static_cast< std::uint32_t >( flatOf( gamma ) ),
                        static_cast< std::uint32_t >( db ) };
                } );
            fn( ai, deg, std::span< const detail::kernels::RecurrenceEntry >( buf.data(), n ) );
        }
    }

   private:
    /// Total degree of the group `gd` block within α.
    static constexpr int subBlockDegree( const MultiIndex< vars >& a,
                                         const detail::MixedGroupDesc& gd ) noexcept
    {
        int d = 0;
        for ( int v = 0; v < gd.dim; ++v ) d += a[std::size_t( gd.varOffset + v )];
        return d;
    }

    /// Global flat index (over dim_g vars) of the group block, or kNotInBox if d > order_g.
    static constexpr std::size_t subBlockFlat( const MultiIndex< vars >& a,
                                               const detail::MixedGroupDesc& gd ) noexcept
    {
        if ( subBlockDegree( a, gd ) > gd.order ) return kNotInBox;
        return flatBlock( a, gd );
    }

    /// flatIndex over the group's own dim_g variables (dispatch on compile-time dims).
    static constexpr std::size_t flatBlock( const MultiIndex< vars >& a,
                                            const detail::MixedGroupDesc& gd ) noexcept
    {
        return flatBlockImpl< Groups::dim... >( a, gd, 0 );
    }
    static constexpr void writeSubBlock( MultiIndex< vars >& a, const detail::MixedGroupDesc& gd,
                                         std::size_t gflat ) noexcept
    {
        writeSubBlockImpl< Groups::dim... >( a, gd, gflat, 0 );
    }

    // Compile-time dispatch over each group's dim so flatIndex<Dim>/unflatIndex<Dim> get
    // their constant template argument while the descriptor selects the runtime block.
    template < int D0, int... Rest >
    static constexpr std::size_t flatBlockImpl( const MultiIndex< vars >& a,
                                                const detail::MixedGroupDesc& gd, int g ) noexcept
    {
        if ( gd.varOffset == varOffsetOf( g ) && D0 == gd.dim )
        {
            MultiIndex< D0 > sub{};
            for ( int v = 0; v < D0; ++v )
                sub[std::size_t( v )] = a[std::size_t( gd.varOffset + v )];
            return flatIndex< D0 >( sub );
        }
        if constexpr ( sizeof...( Rest ) > 0 )
            return flatBlockImpl< Rest... >( a, gd, g + 1 );
        else
            return kNotInBox;
    }
    template < int D0, int... Rest >
    static constexpr void writeSubBlockImpl( MultiIndex< vars >& a,
                                             const detail::MixedGroupDesc& gd, std::size_t gflat,
                                             int g ) noexcept
    {
        if ( gd.varOffset == varOffsetOf( g ) && D0 == gd.dim )
        {
            const MultiIndex< D0 > sub = unflatIndex< D0 >( gflat );
            for ( int v = 0; v < D0; ++v )
                a[std::size_t( gd.varOffset + v )] = sub[std::size_t( v )];
            return;
        }
        if constexpr ( sizeof...( Rest ) > 0 ) writeSubBlockImpl< Rest... >( a, gd, gflat, g + 1 );
    }

    static constexpr int varOffsetOf( int g ) noexcept
    {
        return kGroups[std::size_t( g )].varOffset;
    }

    /// Block size of one degree tuple = Π_g (within-degree count for d_g).
    static constexpr std::size_t tupleBlock(
        const std::array< int, std::size_t( groupCount ) >& deg ) noexcept
    {
        std::size_t b = 1;
        for ( int g = 0; g < groupCount; ++g )
            b *= detail::withinDegreeCount( deg[std::size_t( g )], kGroups[std::size_t( g )].dim );
        return b;
    }

    /// Visit valid degree tuples summing to `target` in lex order (group 0 most significant);
    /// `fn(tuple, block)` returns true to stop early; returns whether it stopped early.
    template < class Fn >
    static constexpr bool forEachTuple( int target, Fn&& fn ) noexcept
    {
        std::array< int, std::size_t( groupCount ) > deg{};
        return tupleRec( 0, target, deg, fn );
    }
    template < class Fn >
    static constexpr bool tupleRec( int g, int remaining,
                                    std::array< int, std::size_t( groupCount ) >& deg,
                                    Fn& fn ) noexcept
    {
        if ( g == groupCount )
        {
            if ( remaining != 0 ) return false;
            return fn( deg, tupleBlock( deg ) );
        }
        const int cap = kGroups[std::size_t( g )].order;
        for ( int dg = 0; dg <= cap && dg <= remaining; ++dg )
        {
            deg[std::size_t( g )] = dg;
            if ( tupleRec( g + 1, remaining - dg, deg, fn ) ) return true;
        }
        deg[std::size_t( g )] = 0;
        return false;
    }

    /// Number of kept monomials of total degree exactly `d` (sum of tuple blocks).
    static constexpr std::size_t degreeTotalCount( int d ) noexcept
    {
        std::size_t total = 0;
        forEachTuple(
            d,
            [&]( const std::array< int, std::size_t( groupCount ) >&, std::size_t block ) -> bool {
                total += block;
                return false;
            } );
        return total;
    }

    /// Sum of tuple block sizes for valid degree-`target` tuples lexicographically before `deg`.
    static constexpr std::size_t precedingTupleCount(
        const std::array< int, std::size_t( groupCount ) >& deg, int target ) noexcept
    {
        std::size_t before = 0;
        forEachTuple( target,
                      [&]( const std::array< int, std::size_t( groupCount ) >& t,
                           std::size_t block ) -> bool {
                          if ( t == deg ) return true;  // reached it: stop
                          before += block;
                          return false;
                      } );
        return before;
    }
};

}  // namespace tax
