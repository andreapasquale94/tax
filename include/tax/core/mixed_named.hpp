#pragma once

// Named, per-axis-order Taylor expansions: MixedTaylorExpansion wraps a dense
// TaylorExpansion<T, MixedScheme<Group<Dim,Order>...>> and attaches a canonical
// (sorted, unique) list of named ordered axes (OrderedAxis<Name, Dim, Order>).
// Complements the joint-simplex NamedTaylorExpansion with per-axis truncation.

#include <array>
#include <cstddef>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/named.hpp>
#include <tax/core/scheme/mixed.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/operators/arithmetic.hpp>
#include <tax/operators/math_binary.hpp>
#include <tax/operators/math_unary.hpp>
#include <utility>

namespace tax::named
{

// Forward declaration so detail::RebindMixed can name the mixed type.
template < typename T, typename... Axes >
    requires Scalar< T >
class MixedTaylorExpansion;

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

/// Map a pack of OrderedAxis types to MixedScheme<Group<Dim,Order>...> (axis i -> group i).
template < typename... Axes >
struct AxesToMixedScheme
{
    using type = MixedScheme< Group< Axes::dim, Axes::order >... >;
};

template < typename... Axes >
using AxesToMixedScheme_t = typename AxesToMixedScheme< Axes... >::type;

// Same-name merge policy for ordered axes: a shared axis must accommodate
// both operands, so the union takes the MAX of the two per-axis orders. The
// generic sorted-merge machinery (Merge / MergeFold) lives in core/named.hpp
// and picks this policy up through the CombineAxes customisation point.
template < FixedString NameA, int DimA, int OrderA, FixedString NameB, int DimB, int OrderB >
struct CombineAxes< OrderedAxis< NameA, DimA, OrderA >, OrderedAxis< NameB, DimB, OrderB > >
{
    static_assert( DimA == DimB, "named axis used with inconsistent dimension across operands" );
    using type = OrderedAxis< NameA, DimA, ( OrderA > OrderB ? OrderA : OrderB ) >;
};

/// Rebind a `TypeList` of ordered axes into a `MixedTaylorExpansion< T, Axes... >`.
template < typename T, typename List >
struct RebindMixed;
template < typename T, typename... Axes >
struct RebindMixed< T, TypeList< Axes... > >
{
    using type = MixedTaylorExpansion< T, Axes... >;
};

/// The mixed type over the merged (union, max-order) axis set of two operands.
template < typename T, typename ListA, typename ListB >
using MergedMixedTaylorExpansion =
    typename RebindMixed< T, typename Merge< ListA, ListB >::type >::type;

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
class MixedTaylorExpansion
{
   public:
    using axis_list = detail::TypeList< Axes... >;
    using scalar_type = T;

    static_assert( detail::IsCanonical< axis_list >::value,
                   "MixedTaylorExpansion axes must be sorted by name and unique; build via "
                   "tax::mixed::variable()/variables() rather than spelling them by hand" );

    /// Number of underlying variables (sum of axis dimensions).
    static constexpr int vars_v = detail::TotalDim< axis_list >::value;

    /// Underlying anonymous dense expansion type (MixedScheme backing).
    using Inner = TaylorExpansion< T, detail::AxesToMixedScheme_t< Axes... > >;
    using Input = typename Inner::Input;

    // Mirror the underlying storage traits.
    using container_t = typename Inner::container_t;
    static constexpr std::size_t nCoefficients = Inner::nCoefficients;

    constexpr MixedTaylorExpansion() noexcept = default;

    /// Constant expansion (value, all higher-order coefficients zero).
    /*implicit*/ constexpr MixedTaylorExpansion( T v ) noexcept : inner_{ v } {}

    /// Wrap an existing anonymous expansion carrying these axes.
    explicit constexpr MixedTaylorExpansion( const Inner& inner ) noexcept : inner_{ inner } {}

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
    /// R). Reindexes box -> box via the MixedScheme, remapping per-axis blocks;
    /// a target with a lower per-axis order drops out-of-box terms.
    template < typename R >
    [[nodiscard]] constexpr R embed() const noexcept
    {
        return detail::reindexAxes< R, /*AllowDrop=*/false >( *this );
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
    [[nodiscard]] constexpr MixedTaylorExpansion deriv() const noexcept
    {
        return MixedTaylorExpansion{ inner_.template deriv< axisVar< Name, Local >() >() };
    }

    /// Indefinite integral with respect to one coordinate of a named axis.
    template < FixedString Name, int Local = 0 >
    [[nodiscard]] constexpr MixedTaylorExpansion integ() const noexcept
    {
        return MixedTaylorExpansion{ inner_.template integ< axisVar< Name, Local >() >() };
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
        using Tgt = typename detail::MergeFold<
            detail::TypeList<>,
            detail::TypeList< OrderedAxis< Names, detail::DimOfName< axis_list, Names >::value,
                                           detail::OrderOfName< axis_list, Names >::value > >... >::
            type;
        using R = typename detail::RebindMixed< T, Tgt >::type;
        return detail::reindexAxes< R, /*AllowDrop=*/true >( *this );
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
        // Identity variable map; re-flatting through the smaller target box
        // drops monomials whose Name-axis degree exceeds N2.
        return detail::reindexAxes< R, /*AllowDrop=*/false >( *this );
    }

   private:
    Inner inner_{};
};

// Composition operators (union axis set, max order per shared axis). Each binary
// op embeds both operands into the merged mixed type, then delegates to the
// underlying TaylorExpansion operator on the (now box-compatible) inners.
#define TAX_MIXED_BINARY_OP( OP )                                                                 \
    template < typename T, typename... A, typename... B >                                         \
    [[nodiscard]] constexpr auto operator OP( const MixedTaylorExpansion< T, A... >& a,           \
                                              const MixedTaylorExpansion< T, B... >& b ) noexcept \
    {                                                                                             \
        using R = detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >,                \
                                                      detail::TypeList< B... > >;                 \
        return R{ a.template embed< R >().inner() OP b.template embed< R >().inner() };           \
    }

TAX_MIXED_BINARY_OP( +)
TAX_MIXED_BINARY_OP( -)
TAX_MIXED_BINARY_OP( * )
TAX_MIXED_BINARY_OP( / )

#undef TAX_MIXED_BINARY_OP

// Compound assignment (same axis set: no axis-union widening).
#define TAX_MIXED_COMPOUND_OP( OP )                                                             \
    template < typename T, typename... A >                                                      \
    constexpr MixedTaylorExpansion< T, A... >& operator OP(                                     \
        MixedTaylorExpansion< T, A... >& a, const MixedTaylorExpansion< T, A... >& b ) noexcept \
    {                                                                                           \
        a.inner() OP b.inner();                                                                 \
        return a;                                                                               \
    }

TAX_MIXED_COMPOUND_OP( += )
TAX_MIXED_COMPOUND_OP( -= )
TAX_MIXED_COMPOUND_OP( *= )

#undef TAX_MIXED_COMPOUND_OP

// Scalar combinations (axis set unchanged).

#define TAX_MIXED_SCALAR_OP( OP )                                                        \
    template < typename T, typename... A >                                               \
    [[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator OP(                 \
        const MixedTaylorExpansion< T, A... >& a, std::type_identity_t< T > s ) noexcept \
    {                                                                                    \
        return MixedTaylorExpansion< T, A... >{ a.inner() OP s };                        \
    }

TAX_MIXED_SCALAR_OP( +)
TAX_MIXED_SCALAR_OP( -)
TAX_MIXED_SCALAR_OP( * )
TAX_MIXED_SCALAR_OP( / )

#undef TAX_MIXED_SCALAR_OP

template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator+(
    std::type_identity_t< T > s, const MixedTaylorExpansion< T, A... >& a ) noexcept
{
    return a + s;
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator*(
    std::type_identity_t< T > s, const MixedTaylorExpansion< T, A... >& a ) noexcept
{
    return a * s;
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator-(
    std::type_identity_t< T > s, const MixedTaylorExpansion< T, A... >& a ) noexcept
{
    return MixedTaylorExpansion< T, A... >{ s - a.inner() };
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator-(
    const MixedTaylorExpansion< T, A... >& a ) noexcept
{
    return MixedTaylorExpansion< T, A... >{ -a.inner() };
}

// Unary math functions (forwarded to the inner expansion, axis set preserved).

#define TAX_MIXED_UNARY_FN( FN )                                        \
    template < typename T, typename... A >                              \
    [[nodiscard]] MixedTaylorExpansion< T, A... > FN(                   \
        const MixedTaylorExpansion< T, A... >& a ) noexcept             \
    {                                                                   \
        return MixedTaylorExpansion< T, A... >{ tax::FN( a.inner() ) }; \
    }

TAX_MIXED_UNARY_FN( square )
TAX_MIXED_UNARY_FN( cube )
TAX_MIXED_UNARY_FN( sqrt )
TAX_MIXED_UNARY_FN( cbrt )
TAX_MIXED_UNARY_FN( reciprocal )
TAX_MIXED_UNARY_FN( exp )
TAX_MIXED_UNARY_FN( log )
TAX_MIXED_UNARY_FN( sin )
TAX_MIXED_UNARY_FN( cos )
TAX_MIXED_UNARY_FN( tan )
TAX_MIXED_UNARY_FN( asin )
TAX_MIXED_UNARY_FN( acos )
TAX_MIXED_UNARY_FN( atan )
TAX_MIXED_UNARY_FN( sinh )
TAX_MIXED_UNARY_FN( cosh )
TAX_MIXED_UNARY_FN( tanh )
TAX_MIXED_UNARY_FN( asinh )
TAX_MIXED_UNARY_FN( acosh )
TAX_MIXED_UNARY_FN( atanh )
TAX_MIXED_UNARY_FN( erf )

#undef TAX_MIXED_UNARY_FN

/// `MTE< Axes... >` — double-valued mixed-order named expansion.
template < typename... Axes >
using MTE = MixedTaylorExpansion< double, Axes... >;

}  // namespace tax::named

// Public re-exports: OrderedAxis, MixedTaylorExpansion and MTE under `tax`.
namespace tax
{
using named::MixedTaylorExpansion;
using named::MTE;
using named::OrderedAxis;
}  // namespace tax

// Factories: `tax::mixed::variable` / `tax::mixed::variables`.
namespace tax::mixed
{

/// Single coordinate variable of a 1-D ordered axis `Name` at `x0`, truncated to `Order`.
template < tax::named::FixedString Name, int Order >
[[nodiscard]] constexpr auto variable( double x0 ) noexcept
{
    using Ax = tax::named::OrderedAxis< Name, 1, Order >;
    using E = tax::named::MixedTaylorExpansion< double, Ax >;
    typename E::Input p{ x0 };
    return E{ E::Inner::template variable< 0 >( p ) };
}

/// The `D` coordinate variables of a `D`-dimensional ordered axis `Name` at
/// `x0`, each truncated to `Order`; returned as a plain `std::array`.
template < tax::named::FixedString Name, int Order, std::size_t D >
[[nodiscard]] constexpr auto variables( const std::array< double, D >& x0 ) noexcept
{
    using Ax = tax::named::OrderedAxis< Name, int( D ), Order >;
    using E = tax::named::MixedTaylorExpansion< double, Ax >;
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
