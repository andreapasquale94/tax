#pragma once

// Named, sliceable, composable Taylor expansions (dense): NamedTaylorExpansion
// wraps a dense TaylorExpansion and attaches a canonical (sorted, unique) list
// of named axes, supporting embed / compose-across-axis-sets / slice.

#include <array>
#include <cstddef>
#include <tax/expansion/axis.hpp>
#include <tax/expansion/concepts.hpp>
#include <tax/expansion/expansion.hpp>
#include <tax/expansion/multi_index.hpp>
#include <utility>

namespace tax::named
{

// Forward declaration so `detail::Rebind` can name the named type.
template < typename T, typename Basis, int N, typename... Axes >
    requires Scalar< T > && BasisPolicy< Basis >
class NamedExpansion;

namespace detail
{

/// Rebind a `TypeList` of axes into a `NamedExpansion< T, Basis, N, Axes... >`.
template < typename T, typename Basis, int N, typename List >
struct Rebind;
template < typename T, typename Basis, int N, typename... Axes >
struct Rebind< T, Basis, N, TypeList< Axes... > >
{
    using type = NamedExpansion< T, Basis, N, Axes... >;
};

/// The named type over the merged (union) axis set of two operands.
template < typename T, typename Basis, int N, typename ListA, typename ListB >
using MergedNamedExpansion =
    typename Rebind< T, Basis, N, typename Merge< ListA, ListB >::type >::type;

}  // namespace detail

// ---------------------------------------------------------------------------
// NamedTaylorExpansion — a named Taylor expansion
// ---------------------------------------------------------------------------

template < typename T, typename Basis, int N, typename... Axes >
    requires Scalar< T > && BasisPolicy< Basis >
class NamedExpansion
{
   public:
    using axis_list = detail::TypeList< Axes... >;
    using scalar_type = T;
    using basis = Basis;

    static_assert( detail::IsCanonical< axis_list >::value,
                   "NamedExpansion axes must be sorted by name and unique; build via "
                   "variables()/composition rather than spelling them by hand" );

    /// Number of underlying variables (sum of axis dimensions).
    static constexpr int vars_v = detail::TotalDim< axis_list >::value;
    static constexpr int order_v = N;

    /// Underlying anonymous dense expansion type.
    using Inner = Expansion< T, Basis, IsotropicScheme< N, vars_v >, storage::Dense >;
    using Input = typename Inner::Input;

    // Mirror the underlying storage traits so NamedTaylorExpansion satisfies the
    // tax::TaylorPolynomial concept and can flow through concept-constrained helpers.
    using container_t = typename Inner::container_t;
    static constexpr std::size_t nCoefficients = Inner::nCoefficients;

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    constexpr NamedExpansion() noexcept = default;

    /// Constant expansion (value in every axis direction is flat).
    /*implicit*/ constexpr NamedExpansion( T v ) noexcept : inner_{ v } {}

    /// Wrap an existing anonymous expansion carrying these axes.
    explicit constexpr NamedExpansion( const Inner& inner ) noexcept : inner_{ inner } {}

    /// Implicit promotion from an expansion over a subset of these axes.
    template < typename... B >
        requires( !std::is_same_v< detail::TypeList< B... >, axis_list > &&
                  detail::IsSubsetOf< detail::TypeList< B... >, axis_list >::value )
    /*implicit*/ constexpr NamedExpansion(
        const NamedExpansion< T, Basis, N, B... >& other ) noexcept
        : inner_{ other.template embed< NamedExpansion >().inner() }
    {
    }

    // ------------------------------------------------------------------
    // Coordinate variables
    // ------------------------------------------------------------------

    /// The I-th coordinate variable of the joint variable space at `p`.
    template < int I >
    [[nodiscard]] static constexpr NamedExpansion variable( const Input& p ) noexcept
        requires( I >= 0 && I < vars_v )
    {
        return NamedExpansion{ Inner::template variable< I >( p ) };
    }

    // ------------------------------------------------------------------
    // Access
    // ------------------------------------------------------------------

    /// Constant (zeroth) coefficient.
    [[nodiscard]] constexpr T value() const noexcept { return inner_.value(); }

    /// The underlying anonymous expansion.
    [[nodiscard]] constexpr const Inner& inner() const noexcept { return inner_; }
    [[nodiscard]] constexpr Inner& inner() noexcept { return inner_; }

    // ------------------------------------------------------------------
    // Embedding and slicing
    // ------------------------------------------------------------------

    /// Embed into the target named type `R`, whose axes must be a superset of this expansion's
    /// axes.
    template < typename R >
    [[nodiscard]] constexpr R embed() const noexcept
    {
        constexpr auto map =
            detail::buildAxisMap< axis_list, typename R::axis_list, /*allowDrop=*/false >();
        typename R::Inner out{};
        for ( std::size_t k = 0; k < Inner::nCoefficients; ++k )
        {
            const T c = inner_[k];
            if ( c == T{} ) continue;
            const auto a_src = unflatIndex< vars_v >( k );
            MultiIndex< R::vars_v > a_dst{};
            for ( int j = 0; j < vars_v; ++j )
                a_dst[std::size_t( map[std::size_t( j )] )] = a_src[std::size_t( j )];
            out[flatIndex< R::vars_v >( a_dst )] = c;
        }
        return R{ out };
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
        using R = typename detail::Rebind< T, Basis, N, Tgt >::type;

        constexpr auto map = detail::buildAxisMap< axis_list, Tgt, /*allowDrop=*/true >();
        typename R::Inner out{};
        for ( std::size_t k = 0; k < Inner::nCoefficients; ++k )
        {
            const T c = inner_[k];
            if ( c == T{} ) continue;
            const auto a_src = unflatIndex< vars_v >( k );
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
            if ( keep ) out[flatIndex< R::vars_v >( a_dst )] += c;
        }
        return R{ out };
    }

    // ------------------------------------------------------------------
    // Per-axis differentiation and integration (axis set preserved)
    // ------------------------------------------------------------------

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
    [[nodiscard]] constexpr NamedExpansion deriv() const noexcept
    {
        return NamedExpansion{ inner_.template deriv< axisVar< Name, Local >() >() };
    }

    /// Indefinite integral with respect to one coordinate of a named axis.
    template < FixedString Name, int Local = 0 >
    [[nodiscard]] constexpr NamedExpansion integ() const noexcept
    {
        return NamedExpansion{ inner_.template integ< axisVar< Name, Local >() >() };
    }

    // ------------------------------------------------------------------
    // Truncation (axis set preserved)
    // ------------------------------------------------------------------

    /// Order-reducing truncation: drop monomials of degree > N2, yielding a lower-order expansion.
    template < int N2 >
    [[nodiscard]] constexpr NamedExpansion< T, Basis, N2, Axes... > truncate() const noexcept
        requires( N2 >= 0 && N2 <= N )
    {
        return NamedExpansion< T, Basis, N2, Axes... >{ inner_.template truncate< N2 >() };
    }

    /// Same-order truncation: zero every coefficient of total degree > d (d>=N copies, d<0 zeroes).
    [[nodiscard]] constexpr NamedExpansion truncate( int d ) const noexcept
    {
        return NamedExpansion{ inner_.truncate( d ) };
    }

   private:
    Inner inner_{};
};

// ---------------------------------------------------------------------------
// Coordinate-variable factory for a single named axis
// ---------------------------------------------------------------------------

/// Build the `D` coordinate variables of a single named axis `Name` (basis-generic;
/// `Basis` defaults to TaylorBasis).
template < FixedString Name, int N, typename Basis = TaylorBasis, typename T, std::size_t D >
[[nodiscard]] constexpr auto variables( const std::array< T, D >& x0 ) noexcept
{
    using Ax = Axis< Name, int( D ) >;
    using E = NamedExpansion< T, Basis, N, Ax >;
    std::array< E, D > out{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out[I] = E::template variable< int( I ) >( x0 ) ), ... );
    }( std::make_index_sequence< D >{} );
    return out;
}

/// Build the single coordinate variable of a 1-D named axis `Name`.
template < FixedString Name, int N, typename Basis = TaylorBasis, typename T >
    requires Scalar< T >
[[nodiscard]] constexpr auto variable( T x0 ) noexcept
{
    using E = NamedExpansion< T, Basis, N, Axis< Name, 1 > >;
    typename E::Input p{ x0 };
    return E::template variable< 0 >( p );
}

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------

/// The Taylor instance: `NamedTaylorExpansion< T, N, Axes... >`.
template < typename T, int N, typename... Axes >
using NamedTaylorExpansion = NamedExpansion< T, TaylorBasis, N, Axes... >;

/// `NE< N, Axes... >` — double-valued named Taylor expansion of order N.
template < int N, typename... Axes >
using NE = NamedExpansion< double, TaylorBasis, N, Axes... >;

}  // namespace tax::named

// ---------------------------------------------------------------------------
// Public re-exports: the named type API is reachable directly under `tax`. The
// free-function operator / math surface (and its `tax::` re-exports) lives in
// operators/named_arithmetic.hpp, named_math_unary.hpp, named_math_binary.hpp.
// ---------------------------------------------------------------------------

namespace tax
{
using named::Axis;
using named::FixedString;
using named::NamedExpansion;
using named::NamedTaylorExpansion;
using named::NE;
using named::variable;
using named::variables;
}  // namespace tax
