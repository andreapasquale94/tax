#pragma once

// Named, per-axis-order Taylor expansions: MixedExpansion wraps a dense
// TaylorExpansion<T, MixedScheme<Group<Dim,Order>...>> and attaches a canonical
// (sorted, unique) list of named ordered axes (OrderedAxis<Name, Dim, Order>).
// Complements the joint-simplex NamedTaylorExpansion with per-axis truncation.

#include <array>
#include <cstddef>
#include <tax/expansion/axis.hpp>
#include <tax/expansion/concepts.hpp>
#include <tax/expansion/expansion.hpp>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/scheme/mixed.hpp>
#include <utility>

namespace tax::mixed
{

// FixedString (the NTTP labelling each axis) lives in tax::named; borrow it here.
using tax::named::FixedString;

// Forward declaration so detail::RebindMixed can name the mixed type.
template < typename T, typename... Axes >
    requires Scalar< T >
class MixedExpansion;

/// Named axis: `Name` labels a block of `Dim` consecutive variables truncated at `Order`.
template < FixedString Name, int Dim, int Order >
struct OrderedAxis
{
    static constexpr auto name = Name;
    static constexpr int dim = Dim;
    static constexpr int order = Order;
    static_assert( Dim >= 1, "OrderedAxis dimension must be at least 1" );
    static_assert( Order >= 0, "OrderedAxis order must be non-negative" );
};

namespace detail
{

// The shared axis metaprogramming lives in tax::named::detail (expansion/axis.hpp
// + meta.hpp) and stays there; borrow the pieces the mixed layer needs so the
// qualified `detail::X` spellings below keep resolving inside tax::mixed::detail.
using tax::named::fixedCompare;
using tax::named::detail::buildAxisMap;
using tax::named::detail::DimOfName;
using tax::named::detail::IsCanonical;
using tax::named::detail::Merge;
using tax::named::detail::MergeFoldWith;
using tax::named::detail::OffsetOf;
using tax::named::detail::Prepend;
using tax::named::detail::TotalDim;
using tax::named::detail::TypeList;

/// Map a pack of OrderedAxis types to MixedScheme<Group<Dim,Order>...> (axis i -> group i).
template < typename... Axes >
struct AxesToMixedScheme
{
    using type = MixedScheme< Group< Axes::dim, Axes::order >... >;
};

template < typename... Axes >
using AxesToMixedScheme_t = typename AxesToMixedScheme< Axes... >::type;

/// Mixed-layer combine policy: require identical dimension, keep the axis with the max order.
struct SameNameMaxOrder
{
    template < typename A0, typename B0 >
    struct apply
    {
        static_assert( A0::dim == B0::dim,
                       "named axis used with inconsistent dimension across operands" );
        using type =
            OrderedAxis< A0::name, A0::dim, ( A0::order > B0::order ? A0::order : B0::order ) >;
    };
};

/// Rebind a `TypeList` of ordered axes into a `MixedExpansion< T, Axes... >`.
template < typename T, typename List >
struct RebindMixed;
template < typename T, typename... Axes >
struct RebindMixed< T, TypeList< Axes... > >
{
    using type = MixedExpansion< T, Axes... >;
};

/// The mixed type over the merged (union, max-order) axis set of two operands.
template < typename T, typename ListA, typename ListB >
using MergedMixedTaylorExpansion =
    typename RebindMixed< T, typename Merge< ListA, ListB, SameNameMaxOrder >::type >::type;

/// Order of the axis named `Name` within a list, or -1 if absent.
template < typename List, FixedString Name >
struct OrderOfName;
template < FixedString Name >
struct OrderOfName< TypeList<>, Name >
{
    static constexpr int value = -1;
};
template < typename H, typename... Ts, FixedString Name >
struct OrderOfName< TypeList< H, Ts... >, Name >
{
    static constexpr int value = ( fixedCompare< H::name, Name > == 0 )
                                     ? H::order
                                     : OrderOfName< TypeList< Ts... >, Name >::value;
};

/// Replace the per-axis order of the axis named `Name` in list `L` with `N2`,
/// leaving all other axes unchanged. `Name` must be present.
template < typename L, FixedString Name, int N2 >
struct ReplaceAxisOrder;
template < FixedString Name, int N2 >
struct ReplaceAxisOrder< TypeList<>, Name, N2 >
{
    // Name not found — the static_assert in axisVar catches this earlier.
    using type = TypeList<>;
};
template < typename H, typename... Ts, FixedString Name, int N2 >
struct ReplaceAxisOrder< TypeList< H, Ts... >, Name, N2 >
{
    using type = typename std::conditional_t<
        fixedCompare< H::name, Name > == 0,
        Prepend< OrderedAxis< Name, H::dim, N2 >, TypeList< Ts... > >,
        Prepend< H, typename ReplaceAxisOrder< TypeList< Ts... >, Name, N2 >::type > >::type;
};

}  // namespace detail

/// Named per-axis-order Taylor expansion over a canonical (sorted, unique) axis set.
template < typename T, typename... Axes >
    requires Scalar< T >
class MixedExpansion
{
   public:
    using axis_list = detail::TypeList< Axes... >;
    using scalar_type = T;

    static_assert( detail::IsCanonical< axis_list >::value,
                   "MixedExpansion axes must be sorted by name and unique; build via "
                   "tax::mixed::variable()/variables() rather than spelling them by hand" );

    /// Number of underlying variables (sum of axis dimensions).
    static constexpr int vars_v = detail::TotalDim< axis_list >::value;

    /// Underlying anonymous dense expansion type (MixedScheme backing).
    using Inner = TaylorExpansion< T, detail::AxesToMixedScheme_t< Axes... >, storage::Dense >;
    using Input = typename Inner::Input;

    // Mirror the underlying storage traits.
    using container_t = typename Inner::container_t;
    static constexpr std::size_t nCoefficients = Inner::nCoefficients;

    constexpr MixedExpansion() noexcept = default;

    /// Constant expansion (value, all higher-order coefficients zero).
    /*implicit*/ constexpr MixedExpansion( T v ) noexcept : inner_{ v } {}

    /// Wrap an existing anonymous expansion carrying these axes.
    explicit constexpr MixedExpansion( const Inner& inner ) noexcept : inner_{ inner } {}

    /// Constant (zeroth) coefficient.
    [[nodiscard]] constexpr T value() const noexcept { return inner_.value(); }

    /// The underlying anonymous expansion (const and mutable).
    [[nodiscard]] constexpr const Inner& inner() const noexcept { return inner_; }
    [[nodiscard]] constexpr Inner& inner() noexcept { return inner_; }

    /// Read coefficient at flat index `k`.
    [[nodiscard]] constexpr T operator[]( std::size_t k ) const noexcept { return inner_[k]; }

    /// Write coefficient at flat index `k`.
    [[nodiscard]] constexpr T& operator[]( std::size_t k ) noexcept { return inner_[k]; }

    /// Runtime multi-index coefficient lookup.
    [[nodiscard]] constexpr T coeff( const MultiIndex< vars_v >& alpha ) const noexcept
    {
        return inner_.coeff( alpha );
    }

    /// Compile-time multi-index coefficient lookup.
    template < int... Alpha >
    [[nodiscard]] constexpr T coeff() const noexcept
    {
        return inner_.template coeff< Alpha... >();
    }

    /// Runtime partial derivative value `d^|alpha| f / dx^alpha` at x0.
    [[nodiscard]] constexpr T derivative( const MultiIndex< vars_v >& alpha ) const noexcept
    {
        return inner_.derivative( alpha );
    }

    /// Embed into the target mixed type `R`, whose axes are a superset of this
    /// expansion's and whose per-axis orders are >= these (this is a sub-box of
    /// R). Reindexes box -> box via the MixedScheme, remapping per-axis blocks.
    template < typename R >
    [[nodiscard]] constexpr R embed() const noexcept
    {
        constexpr auto map =
            detail::buildAxisMap< axis_list, typename R::axis_list, /*allowDrop=*/false >();
        typename R::Inner::Data out{};
        for ( std::size_t k = 0; k < Inner::nCoefficients; ++k )
        {
            const T c = inner_[k];
            if ( c == T{} ) continue;
            const auto a_src = Inner::scheme::multiOf( k );
            MultiIndex< R::vars_v > a_dst{};
            for ( int j = 0; j < vars_v; ++j )
                a_dst[std::size_t( map[std::size_t( j )] )] = a_src[std::size_t( j )];
            const std::size_t dst = R::Inner::scheme::flatOf( a_dst );
            // R's axes superset this one's; with R's per-axis orders >= these
            // (the intended use) every monomial stays in R's box. Guard the write
            // so a target with a lower per-axis order drops the out-of-box term
            // instead of writing out of bounds (size_t(-1)).
            if ( dst != R::Inner::scheme::kNotInBox ) out[dst] = c;
        }
        return R{ typename R::Inner{ out } };
    }

    // Per-axis differentiation and integration (axis set preserved).

    /// Global variable index of local coordinate `Local` of axis `Name`.
    template < FixedString Name, int Local = 0 >
    static constexpr int axisVar() noexcept
    {
        constexpr int dim = detail::DimOfName< axis_list, Name >::value;
        static_assert( dim >= 1, "axis name not present in this expansion" );
        static_assert( Local >= 0 && Local < dim, "local axis index out of range" );
        // OffsetOf matches by name only, so the order slot is irrelevant here.
        using Ax = OrderedAxis< Name, dim, 0 >;
        return detail::OffsetOf< axis_list, Ax >::value + Local;
    }

    /// Partial derivative with respect to one coordinate of a named axis.
    template < FixedString Name, int Local = 0 >
    [[nodiscard]] constexpr MixedExpansion deriv() const noexcept
    {
        return MixedExpansion{ inner_.template deriv< axisVar< Name, Local >() >() };
    }

    /// Indefinite integral with respect to one coordinate of a named axis.
    template < FixedString Name, int Local = 0 >
    [[nodiscard]] constexpr MixedExpansion integ() const noexcept
    {
        return MixedExpansion{ inner_.template integ< axisVar< Name, Local >() >() };
    }

    /// Project onto the subset of axes named by `Names...`; source monomials
    /// with nonzero degree in any dropped axis are discarded.
    template < FixedString... Names >
    [[nodiscard]] constexpr auto slice() const noexcept
    {
        static_assert( sizeof...( Names ) >= 1, "slice() needs at least one axis name" );
        static_assert( ( ( detail::DimOfName< axis_list, Names >::value >= 1 ) && ... ),
                       "slice(): every requested axis name must exist in this expansion" );
        // Build target axis list (sorted, unique): each named axis with its current dim+order.
        using Tgt = typename detail::MergeFoldWith<
            detail::SameNameMaxOrder, detail::TypeList<>,
            detail::TypeList< OrderedAxis< Names, detail::DimOfName< axis_list, Names >::value,
                                           detail::OrderOfName< axis_list, Names >::value > >... >::
            type;
        using R = typename detail::RebindMixed< T, Tgt >::type;

        // Variable remap: source var j -> target var map[j] (or -1 if dropped).
        constexpr auto map = detail::buildAxisMap< axis_list, Tgt, /*allowDrop=*/true >();
        typename R::Inner::Data out{};
        for ( std::size_t k = 0; k < Inner::nCoefficients; ++k )
        {
            const T c = inner_[k];
            if ( c == T{} ) continue;
            const auto a_src = Inner::scheme::multiOf( k );
            MultiIndex< R::vars_v > a_dst{};
            bool keep = true;
            for ( int j = 0; j < vars_v; ++j )
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
            if ( keep )
            {
                const std::size_t dst = R::Inner::scheme::flatOf( a_dst );
                out[dst] += c;
            }
        }
        return R{ typename R::Inner{ out } };
    }

    /// Lower axis `Name`'s per-axis order to `N2`. Monomials whose `Name`-axis
    /// block degree exceeds `N2` are silently dropped (they fall outside the
    /// smaller box). The variable layout is unchanged; only the scheme changes.
    template < FixedString Name, int N2 >
    [[nodiscard]] constexpr auto truncate() const noexcept
    {
        constexpr int cur_order = detail::OrderOfName< axis_list, Name >::value;
        static_assert( cur_order >= 0, "axis name not present in this expansion" );
        static_assert( N2 >= 0 && N2 <= cur_order,
                       "truncateAxis<Name,N2>: N2 must be in [0, current order of Name]" );
        using TgtList = typename detail::ReplaceAxisOrder< axis_list, Name, N2 >::type;
        using R = typename detail::RebindMixed< T, TgtList >::type;

        typename R::Inner::Data out{};
        for ( std::size_t k = 0; k < Inner::nCoefficients; ++k )
        {
            const T c = inner_[k];
            if ( c == T{} ) continue;
            const auto a_src = Inner::scheme::multiOf( k );
            // Re-flat through the target (smaller) scheme — same variable layout.
            const std::size_t dst = R::Inner::scheme::flatOf( a_src );
            if ( dst != R::Inner::scheme::kNotInBox ) out[dst] = c;
        }
        return R{ typename R::Inner{ out } };
    }

   private:
    Inner inner_{};
};

// ---------------------------------------------------------------------------
// Convenience alias (double-valued)
// ---------------------------------------------------------------------------

/// `MTE< Axes... >` — double-valued mixed-order named expansion.
template < typename... Axes >
using MTE = MixedExpansion< double, Axes... >;

}  // namespace tax::mixed

// Public re-exports: OrderedAxis, MixedExpansion and MTE under `tax`.
namespace tax
{
using mixed::MixedExpansion;
using mixed::MTE;
using mixed::OrderedAxis;
}  // namespace tax

// Factories: `tax::mixed::variable` / `tax::mixed::variables`.
namespace tax::mixed
{

/// Single coordinate variable of a 1-D ordered axis `Name` at `x0`, truncated to `Order`.
template < tax::named::FixedString Name, int Order >
[[nodiscard]] constexpr auto variable( double x0 ) noexcept
{
    using Ax = OrderedAxis< Name, 1, Order >;
    using E = MixedExpansion< double, Ax >;
    typename E::Input p{ x0 };
    return E{ E::Inner::template variable< 0 >( p ) };
}

/// The `D` coordinate variables of a `D`-dimensional ordered axis `Name` at
/// `x0`, each truncated to `Order`; returned as a plain `std::array`.
template < tax::named::FixedString Name, int Order, std::size_t D >
[[nodiscard]] constexpr auto variables( const std::array< double, D >& x0 ) noexcept
{
    using Ax = OrderedAxis< Name, int( D ), Order >;
    using E = MixedExpansion< double, Ax >;
    std::array< E, D > out{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out[I] = E{ E::Inner::template variable< int( I ) >( x0 ) } ), ... );
    }( std::make_index_sequence< D >{} );
    return out;
}

// Unnamed counterparts: the I-th / all coordinate variables of a bare
// `MixedScheme<Groups...>` dense expansion, indexed by integer rather than name.

/// The `I`-th coordinate variable of an unnamed mixed-order dense expansion at point `p`.
template < int I, typename... Groups, tax::Scalar T = double >
[[nodiscard]] constexpr auto variable(
    const std::array< T, std::size_t( tax::MixedScheme< Groups... >::vars ) >& p ) noexcept
{
    return tax::TaylorExpansion< T, tax::MixedScheme< Groups... > >::template variable< I >( p );
}

/// All coordinate variables of an unnamed mixed-order dense expansion at point `p`.
template < typename... Groups, tax::Scalar T = double >
[[nodiscard]] constexpr auto variables(
    const std::array< T, std::size_t( tax::MixedScheme< Groups... >::vars ) >& p ) noexcept
{
    using E = tax::TaylorExpansion< T, tax::MixedScheme< Groups... > >;
    constexpr std::size_t V = std::size_t( tax::MixedScheme< Groups... >::vars );
    std::array< E, V > out{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out[I] = E::template variable< int( I ) >( p ) ), ... );
    }( std::make_index_sequence< V >{} );
    return out;
}

}  // namespace tax::mixed
