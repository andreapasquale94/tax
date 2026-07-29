#pragma once

// Named, sliceable, composable Taylor expansions (dense): NamedTaylorExpansion
// wraps a dense TaylorExpansion and attaches a canonical (sorted, unique) list
// of named axes, supporting embed / compose-across-axis-sets / slice.

#include <array>
#include <cstddef>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <utility>

namespace tax::named
{

/// A null-terminated compile-time string suitable as a non-type template parameter (e.g. `Axis<
/// "x", 3 >`).
template < std::size_t K >
struct FixedString
{
    char data[K]{};

    /*implicit*/ constexpr FixedString( const char ( &s )[K] ) noexcept
    {
        for ( std::size_t i = 0; i < K; ++i ) data[i] = s[i];
    }

    [[nodiscard]] static constexpr std::size_t size() noexcept { return K - 1; }

    [[nodiscard]] constexpr char operator[]( std::size_t i ) const noexcept { return data[i]; }
};

/// Three-way lexicographic comparison of two FixedStrings (-1 / 0 / 1).
template < std::size_t A, std::size_t B >
[[nodiscard]] constexpr int compareFixed( const FixedString< A >& a,
                                          const FixedString< B >& b ) noexcept
{
    const std::size_t la = a.size();
    const std::size_t lb = b.size();
    const std::size_t n = la < lb ? la : lb;
    for ( std::size_t i = 0; i < n; ++i )
    {
        // Compare as unsigned char: plain char signedness is implementation-defined,
        // so signed comparison would order non-ASCII axis names inconsistently and
        // could make the canonical merged-type ordering platform-dependent.
        const unsigned char ca = static_cast< unsigned char >( a[i] );
        const unsigned char cb = static_cast< unsigned char >( b[i] );
        if ( ca != cb ) return ca < cb ? -1 : 1;
    }
    if ( la == lb ) return 0;
    return la < lb ? -1 : 1;
}

/// Three-way comparison of two FixedString NTTP values (-1 / 0 / 1).
template < FixedString A, FixedString B >
inline constexpr int fixedCompare = compareFixed( A, B );

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

// Compile-time axis-list machinery

namespace detail
{

/// A list of axis types.
template < typename... Ts >
struct TypeList
{
    static constexpr std::size_t size = sizeof...( Ts );
};

template < typename Head, typename List >
struct Prepend;
template < typename Head, typename... Ts >
struct Prepend< Head, TypeList< Ts... > >
{
    using type = TypeList< Head, Ts... >;
};

// --- Merge two name-sorted axis lists into one sorted, unique list ----------

/// Same-name combination policy: how two axes with equal names merge into the
/// union list. The primary template covers plain `Axis` (must be identical);
/// `OrderedAxis` (core/mixed_named.hpp) specialises it to take max(order).
template < typename A, typename B >
struct CombineAxes
{
    static_assert( A::dim == B::dim,
                   "named axis used with inconsistent dimension across operands" );
    using type = A;
};

template < int Cmp, typename A, typename B >
struct MergeChoose;

template < typename A, typename B >
struct Merge;

template < typename... Bs >
struct Merge< TypeList<>, TypeList< Bs... > >
{
    using type = TypeList< Bs... >;
};

template < typename A0, typename... As >
struct Merge< TypeList< A0, As... >, TypeList<> >
{
    using type = TypeList< A0, As... >;
};

template < typename A0, typename... As, typename B0, typename... Bs >
struct Merge< TypeList< A0, As... >, TypeList< B0, Bs... > >
    : MergeChoose< axisSign< A0, B0 >, TypeList< A0, As... >, TypeList< B0, Bs... > >
{
};

// A0 < B0 : take A0
template < typename A0, typename... As, typename B0, typename... Bs >
struct MergeChoose< -1, TypeList< A0, As... >, TypeList< B0, Bs... > >
{
    using type =
        typename Prepend< A0,
                          typename Merge< TypeList< As... >, TypeList< B0, Bs... > >::type >::type;
};

// A0 > B0 : take B0
template < typename A0, typename... As, typename B0, typename... Bs >
struct MergeChoose< 1, TypeList< A0, As... >, TypeList< B0, Bs... > >
{
    using type =
        typename Prepend< B0,
                          typename Merge< TypeList< A0, As... >, TypeList< Bs... > >::type >::type;
};

// A0 == B0 (same name) : combine per the axis kind's policy, advance both
template < typename A0, typename... As, typename B0, typename... Bs >
struct MergeChoose< 0, TypeList< A0, As... >, TypeList< B0, Bs... > >
{
    using type =
        typename Prepend< typename CombineAxes< A0, B0 >::type,
                          typename Merge< TypeList< As... >, TypeList< Bs... > >::type >::type;
};

/// Left-fold `Merge` over a pack of (singleton) axis lists.
template < typename Acc, typename... Rest >
struct MergeFold
{
    using type = Acc;
};
template < typename Acc, typename First, typename... Rest >
struct MergeFold< Acc, First, Rest... >
{
    using type = typename MergeFold< typename Merge< Acc, First >::type, Rest... >::type;
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

/// Reindex the coefficients of a named-family expansion `src` into the target
/// expansion type `R` along the source->target variable map. One body serves
/// embed (AllowDrop = false: the map is total and injective), slice
/// (AllowDrop = true: monomials with nonzero degree in a dropped variable are
/// discarded) and per-axis order truncation (identity map, smaller target
/// box). Writes go through the target scheme's flatOf, so monomials outside
/// the target's kept set are dropped rather than written out of bounds.
template < typename R, bool AllowDrop, typename SrcExp >
[[nodiscard]] constexpr R reindexAxes( const SrcExp& src ) noexcept
{
    using T = typename SrcExp::scalar_type;
    using SrcInner = typename SrcExp::Inner;
    constexpr auto map =
        buildAxisMap< typename SrcExp::axis_list, typename R::axis_list, AllowDrop >();

    typename R::Inner::Data out{};
    for ( std::size_t k = 0; k < SrcInner::nCoefficients; ++k )
    {
        const T c = src.inner()[k];
        if ( c == T{} ) continue;
        const auto a_src = SrcInner::scheme::multiOf( k );
        MultiIndex< R::vars_v > a_dst{};
        bool keep = true;
        for ( int j = 0; j < SrcExp::vars_v; ++j )
        {
            const int to = map[std::size_t( j )];
            if ( to < 0 )
            {
                if ( a_src[std::size_t( j )] != 0 )
                {
                    keep = false;
                    break;
                }
            } else
            {
                a_dst[std::size_t( to )] = a_src[std::size_t( j )];
            }
        }
        if ( !keep ) continue;
        const std::size_t dst = R::Inner::scheme::flatOf( a_dst );
        if ( dst != R::Inner::scheme::kNotInBox ) out[dst] += c;
    }
    return R{ typename R::Inner{ out } };
}

}  // namespace detail

// Forward declaration so `detail::Rebind` can name the named type.
template < typename T, int N, typename... Axes >
    requires Scalar< T >
class NamedTaylorExpansion;

namespace detail
{

/// Rebind a `TypeList` of axes into an `NamedTaylorExpansion< T, N, Axes... >`.
template < typename T, int N, typename List >
struct Rebind;
template < typename T, int N, typename... Axes >
struct Rebind< T, N, TypeList< Axes... > >
{
    using type = NamedTaylorExpansion< T, N, Axes... >;
};

/// The named type over the merged (union) axis set of two operands.
template < typename T, int N, typename ListA, typename ListB >
using MergedNamedTaylorExpansion =
    typename Rebind< T, N, typename Merge< ListA, ListB >::type >::type;

}  // namespace detail

/// A named, sliceable, composable dense Taylor expansion over a canonical axis set.
template < typename T, int N, typename... Axes >
    requires Scalar< T >
class NamedTaylorExpansion
{
   public:
    using axis_list = detail::TypeList< Axes... >;
    using scalar_type = T;

    static_assert( detail::IsCanonical< axis_list >::value,
                   "NamedTaylorExpansion axes must be sorted by name and unique; build via "
                   "variables()/composition rather than spelling them by hand" );

    /// Number of underlying variables (sum of axis dimensions).
    static constexpr int vars_v = detail::TotalDim< axis_list >::value;
    static constexpr int order_v = N;

    /// Underlying anonymous dense expansion type.
    using Inner = TaylorExpansion< T, IsotropicScheme< N, vars_v > >;
    using Input = typename Inner::Input;

    // Mirror the underlying storage traits so NamedTaylorExpansion satisfies the
    // tax::TaylorPolynomial concept and can flow through concept-constrained helpers.
    using container_t = typename Inner::container_t;
    static constexpr std::size_t nCoefficients = Inner::nCoefficients;

    constexpr NamedTaylorExpansion() noexcept = default;

    /// Constant expansion (value in every axis direction is flat).
    /*implicit*/ constexpr NamedTaylorExpansion( T v ) noexcept : inner_{ v } {}

    /// Wrap an existing anonymous expansion carrying these axes.
    explicit constexpr NamedTaylorExpansion( const Inner& inner ) noexcept : inner_{ inner } {}

    /// Implicit promotion from an expansion over a subset of these axes.
    template < typename... B >
        requires( !std::is_same_v< detail::TypeList< B... >, axis_list > &&
                  detail::IsSubsetOf< detail::TypeList< B... >, axis_list >::value )
    /*implicit*/ constexpr NamedTaylorExpansion(
        const NamedTaylorExpansion< T, N, B... >& other ) noexcept
        : inner_{ other.template embed< NamedTaylorExpansion >().inner() }
    {
    }

    /// The I-th coordinate variable of the joint variable space at `p`.
    template < int I >
    [[nodiscard]] static constexpr NamedTaylorExpansion variable( const Input& p ) noexcept
        requires( I >= 0 && I < vars_v )
    {
        return NamedTaylorExpansion{ Inner::template variable< I >( p ) };
    }

    /// Constant (zeroth) coefficient.
    [[nodiscard]] constexpr T value() const noexcept { return inner_.value(); }

    /// The underlying anonymous expansion.
    [[nodiscard]] constexpr const Inner& inner() const noexcept { return inner_; }
    [[nodiscard]] constexpr Inner& inner() noexcept { return inner_; }

    /// Embed into the target named type `R`, whose axes must be a superset of this expansion's
    /// axes.
    template < typename R >
    [[nodiscard]] constexpr R embed() const noexcept
    {
        return detail::reindexAxes< R, /*AllowDrop=*/false >( *this );
    }

    /// Project onto the subset of axes named by `Names...`.
    template < FixedString... Names >
    [[nodiscard]] constexpr auto slice() const noexcept
    {
        static_assert( sizeof...( Names ) >= 1, "slice() needs at least one axis name" );
        // Check name existence *before* forming Axis< Name, DimOfName::value >: an
        // absent name yields Dim == -1, which would otherwise trip Axis's own
        // "dimension must be at least 1" assert with a confusing message.
        static_assert( ( ( detail::DimOfName< axis_list, Names >::value >= 1 ) && ... ),
                       "slice(): every requested axis name must exist in this expansion" );
        using Tgt = typename detail::MergeFold<
            detail::TypeList<>,
            detail::TypeList< Axis< Names, detail::DimOfName< axis_list, Names >::value > >... >::
            type;
        using R = typename detail::Rebind< T, N, Tgt >::type;
        return detail::reindexAxes< R, /*AllowDrop=*/true >( *this );
    }

    // Per-axis differentiation and integration (axis set preserved)

    /// Global variable index of local coordinate `Local` of axis `Name`.
    template < FixedString Name, int Local >
    static constexpr int axisVar() noexcept
    {
        constexpr int dim = detail::DimOfName< axis_list, Name >::value;
        static_assert( dim >= 1, "axis name not present in this expansion" );
        static_assert( Local >= 0 && Local < dim, "local axis index out of range" );
        return detail::OffsetOf< axis_list, Axis< Name, dim > >::value + Local;
    }

    /// Partial derivative with respect to one coordinate of a named axis.
    template < FixedString Name, int Local = 0 >
    [[nodiscard]] constexpr NamedTaylorExpansion deriv() const noexcept
    {
        return NamedTaylorExpansion{ inner_.template deriv< axisVar< Name, Local >() >() };
    }

    /// Indefinite integral with respect to one coordinate of a named axis.
    template < FixedString Name, int Local = 0 >
    [[nodiscard]] constexpr NamedTaylorExpansion integ() const noexcept
    {
        return NamedTaylorExpansion{ inner_.template integ< axisVar< Name, Local >() >() };
    }

    // Truncation (axis set preserved)

    /// Order-reducing truncation: drop monomials of degree > N2, yielding a lower-order expansion.
    template < int N2 >
    [[nodiscard]] constexpr NamedTaylorExpansion< T, N2, Axes... > truncate() const noexcept
        requires( N2 >= 0 && N2 <= N )
    {
        return NamedTaylorExpansion< T, N2, Axes... >{ inner_.template truncate< N2 >() };
    }

    /// Same-order truncation: zero every coefficient of total degree > d (d>=N copies, d<0 zeroes).
    [[nodiscard]] constexpr NamedTaylorExpansion truncate( int d ) const noexcept
    {
        return NamedTaylorExpansion{ inner_.truncate( d ) };
    }

   private:
    Inner inner_{};
};

// Coordinate-variable factory for a single named axis

/// Build the `D` coordinate variables of a single named axis `Name`.
template < FixedString Name, int N, typename T, std::size_t D >
[[nodiscard]] constexpr auto variables( const std::array< T, D >& x0 ) noexcept
{
    using Ax = Axis< Name, int( D ) >;
    using E = NamedTaylorExpansion< T, N, Ax >;
    std::array< E, D > out{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out[I] = E::template variable< int( I ) >( x0 ) ), ... );
    }( std::make_index_sequence< D >{} );
    return out;
}

/// Build the single coordinate variable of a 1-D named axis `Name`.
template < FixedString Name, int N, typename T >
    requires Scalar< T >
[[nodiscard]] constexpr auto variable( T x0 ) noexcept
{
    using E = NamedTaylorExpansion< T, N, Axis< Name, 1 > >;
    typename E::Input p{ x0 };
    return E::template variable< 0 >( p );
}

/// `NE< N, Axes... >` — double-valued named expansion of order N.
template < int N, typename... Axes >
using NE = NamedTaylorExpansion< double, N, Axes... >;

}  // namespace tax::named

// Public re-exports: the named type API is reachable directly under `tax`. The
// free-function operator / math surface (and its `tax::` re-exports) lives in
// operators/named_arithmetic.hpp, named_math_unary.hpp, named_math_binary.hpp.

namespace tax
{
using named::Axis;
using named::FixedString;
using named::NamedTaylorExpansion;
using named::NE;
using named::variable;
using named::variables;
}  // namespace tax
