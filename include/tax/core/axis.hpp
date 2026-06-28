#pragma once

// Named axis types, axis-set lookups, merge, and the source->target remap shared by the named
// and mixed layers.

#include <array>
#include <cstddef>
#include <type_traits>
#include <tax/core/meta.hpp>

namespace tax::named
{

// ---------------------------------------------------------------------------
// Axis — a named block of `Dim` consecutive variables
// ---------------------------------------------------------------------------

/// A named axis: the compile-time string `Name` labels a block of `Dim` consecutive variables of
/// the underlying expansion.
template < FixedString Name, int Dim >
struct Axis
{
    static constexpr auto name = Name;
    static constexpr int dim = Dim;
    static_assert( Dim >= 1, "Axis dimension must be at least 1" );
};

/// Sign of the name comparison of two axes (-1 / 0 / 1).
/// `fixedCompare` already returns exactly -1/0/1, so no further clamping is needed.
template < typename A, typename B >
inline constexpr int axisSign = fixedCompare< A::name, B::name >;

// ---------------------------------------------------------------------------
// Compile-time axis-list merge and lookup machinery
// ---------------------------------------------------------------------------

namespace detail
{

// --- Same-name combine policies for Merge -----------------------------------

/// Default combine policy: require identical dimension, keep the first axis.
struct SameNameRequireEqual
{
    template < typename A0, typename B0 >
    struct apply
    {
        static_assert( A0::dim == B0::dim,
                       "named axis used with inconsistent dimension across operands" );
        using type = A0;
    };
};

// --- Merge two name-sorted axis lists into one sorted, unique list ----------

template < int Cmp, typename A, typename B, typename Combine >
struct MergeChoose;

template < typename A, typename B, typename Combine = SameNameRequireEqual >
struct Merge;

template < typename... Bs, typename Combine >
struct Merge< TypeList<>, TypeList< Bs... >, Combine >
{
    using type = TypeList< Bs... >;
};

template < typename A0, typename... As, typename Combine >
struct Merge< TypeList< A0, As... >, TypeList<>, Combine >
{
    using type = TypeList< A0, As... >;
};

template < typename A0, typename... As, typename B0, typename... Bs, typename Combine >
struct Merge< TypeList< A0, As... >, TypeList< B0, Bs... >, Combine >
    : MergeChoose< axisSign< A0, B0 >, TypeList< A0, As... >, TypeList< B0, Bs... >, Combine >
{
};

// A0 < B0 : take A0
template < typename A0, typename... As, typename B0, typename... Bs, typename Combine >
struct MergeChoose< -1, TypeList< A0, As... >, TypeList< B0, Bs... >, Combine >
{
    using type = typename Prepend<
        A0, typename Merge< TypeList< As... >, TypeList< B0, Bs... >, Combine >::type >::type;
};

// A0 > B0 : take B0
template < typename A0, typename... As, typename B0, typename... Bs, typename Combine >
struct MergeChoose< 1, TypeList< A0, As... >, TypeList< B0, Bs... >, Combine >
{
    using type = typename Prepend<
        B0, typename Merge< TypeList< A0, As... >, TypeList< Bs... >, Combine >::type >::type;
};

// A0 == B0 (same name) : combine via policy, advance both
template < typename A0, typename... As, typename B0, typename... Bs, typename Combine >
struct MergeChoose< 0, TypeList< A0, As... >, TypeList< B0, Bs... >, Combine >
{
    using Merged = typename Combine::template apply< A0, B0 >::type;
    using type   = typename Prepend<
        Merged, typename Merge< TypeList< As... >, TypeList< Bs... >, Combine >::type >::type;
};

/// Left-fold `Merge` over a pack of (singleton) axis lists, forwarding a Combine policy.
template < typename Combine, typename Acc, typename... Rest >
struct MergeFoldWith
{
    using type = Acc;
};
template < typename Combine, typename Acc, typename First, typename... Rest >
struct MergeFoldWith< Combine, Acc, First, Rest... >
{
    using type = typename MergeFoldWith< Combine,
                                         typename Merge< Acc, First, Combine >::type,
                                         Rest... >::type;
};

/// Convenience alias: left-fold with the default (named) combine policy.
template < typename Acc, typename... Rest >
struct MergeFold : MergeFoldWith< SameNameRequireEqual, Acc, Rest... >
{
};

// --- Lookups ----------------------------------------------------------------

/// Variable offset of an axis (matched by name) within a list, or -1.
template < typename List, typename Ax >
struct OffsetOf;
template < typename Ax >
struct OffsetOf< TypeList<>, Ax >
{
    static constexpr int value = -1;
};
template < typename H, typename... Ts, typename Ax >
struct OffsetOf< TypeList< H, Ts... >, Ax >
{
   private:
    static constexpr int tail = OffsetOf< TypeList< Ts... >, Ax >::value;

   public:
    static constexpr int value = ( axisSign< H, Ax > == 0 ) ? 0 : ( tail < 0 ? -1 : H::dim + tail );
};

/// Dimension of the axis named `Name` within a list, or -1 if absent.
template < typename List, FixedString Name >
struct DimOfName;
template < FixedString Name >
struct DimOfName< TypeList<>, Name >
{
    static constexpr int value = -1;
};
template < typename H, typename... Ts, FixedString Name >
struct DimOfName< TypeList< H, Ts... >, Name >
{
    static constexpr int value = ( fixedCompare< H::name, Name > == 0 )
                                     ? H::dim
                                     : DimOfName< TypeList< Ts... >, Name >::value;
};

/// Total number of variables (sum of axis dimensions) in a list.
template < typename List >
struct TotalDim;
template < typename... Axes >
struct TotalDim< TypeList< Axes... > >
{
    static constexpr int value = ( Axes::dim + ... + 0 );
};

/// True if the axes are sorted by name with no duplicates.
template < typename List >
struct IsCanonical : std::true_type
{
};
template < typename A0 >
struct IsCanonical< TypeList< A0 > > : std::true_type
{
};
template < typename A0, typename A1, typename... Rest >
struct IsCanonical< TypeList< A0, A1, Rest... > >
    : std::bool_constant< ( axisSign< A0, A1 > < 0 ) &&
                          IsCanonical< TypeList< A1, Rest... > >::value >
{
};

/// True if every axis of `Sub` is present in `Super` with the same dim.
template < typename Sub, typename Super >
struct IsSubsetOf;
template < typename Super, typename... Bs >
struct IsSubsetOf< TypeList< Bs... >, Super >
    : std::bool_constant< ( ( DimOfName< Super, Bs::name >::value == Bs::dim ) && ... ) >
{
};

// --- Source -> target variable index map -----------------------------------

template < typename Tgt, bool allowDrop, typename... SrcAxes >
[[nodiscard]] constexpr auto buildAxisMapImpl( TypeList< SrcAxes... > ) noexcept
{
    constexpr int Msrc = ( SrcAxes::dim + ... + 0 );
    std::array< int, std::size_t( Msrc ) > map{};
    int so = 0;
    auto place = [&]< typename Ax >() constexpr {
        constexpr int to = OffsetOf< Tgt, Ax >::value;
        static_assert( allowDrop || to >= 0,
                       "embed(): target axis set is not a superset of the source" );
        for ( int l = 0; l < Ax::dim; ++l )
            map[std::size_t( so + l )] = ( to < 0 ) ? -1 : ( to + l );
        so += Ax::dim;
    };
    ( place.template operator()< SrcAxes >(), ... );
    return map;
}

/// Build the per-variable index map from a source axis layout to a target axis layout.
template < typename Src, typename Tgt, bool allowDrop >
[[nodiscard]] constexpr auto buildAxisMap() noexcept
{
    return buildAxisMapImpl< Tgt, allowDrop >( Src{} );
}

}  // namespace detail

}  // namespace tax::named
