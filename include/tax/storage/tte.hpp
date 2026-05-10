// SPDX-License-Identifier: BSD-3-Clause
//
// Unified truncated Taylor expansion storage: TaylorExpansionT<T, Order, Vars>.
//
// Modelled on `Eigen::Matrix<T, Rows, Cols>`: the size template parameters are
// signed `int`s so that `Eigen::Dynamic` (= -1) doubles as the runtime-size
// sentinel.  Two configurations are publicly supported:
//
//   - Both `Order` and `Vars` are non-negative integers — fully static, with
//     coefficients in an `Eigen::Matrix<T, monomialCount(N, M), 1>` allocated
//     in-place (no heap, no runtime shape state, EBO collapses the
//     ShapeBase to zero size).
//
//   - Both `Order` and `Vars` are `Eigen::Dynamic` — fully runtime, with
//     coefficients in `Eigen::VectorX<T>` and the order/nvars tracked as
//     members of a private `DynamicShape` base.
//
// Mixed dynamism (one static, one dynamic) is rejected at compile time.

#pragma once

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "tax/concepts.hpp"
#include "tax/expr/base.hpp"
#include "tax/fwd.hpp"
#include "tax/util/multi_index.hpp"

namespace tax
{

namespace detail
{

// Empty-base shape helper for the static configuration.  Holds nothing;
// `order()` and `nvars()` return the constexpr template values.
template < int Order, int Vars >
struct ConstexprShape
{
    [[nodiscard]] static constexpr std::size_t order() noexcept
    {
        return static_cast< std::size_t >( Order );
    }
    [[nodiscard]] static constexpr std::size_t nvars() noexcept
    {
        return static_cast< std::size_t >( Vars );
    }
};

// Runtime shape helper for the dynamic configuration.  Carries the
// (order, nvars) pair as members; ctor takes them at construction time.
struct DynamicShape
{
    std::size_t order_;
    std::size_t nvars_;

    DynamicShape() noexcept : order_( 0 ), nvars_( 0 )
    {
    }
    DynamicShape( std::size_t o, std::size_t n ) noexcept : order_( o ), nvars_( n )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept
    {
        return order_;
    }
    [[nodiscard]] std::size_t nvars() const noexcept
    {
        return nvars_;
    }
};

template < int Order, int Vars >
using ShapeBase = std::conditional_t<
    ( Order != Eigen::Dynamic ) && ( Vars != Eigen::Dynamic ),
    ConstexprShape< Order, Vars >, DynamicShape >;

}  // namespace detail

// ----------------------------------------------------------------------
// TaylorExpansionT<T, Order, Vars>
//
// Public inheritance from ShapeBase brings `order()` / `nvars()` into the
// public interface uniformly across the static and dynamic cases.  When
// `ConstexprShape` (the static helper) is empty, EBO collapses it to zero
// size, so a static TE costs only its Eigen::Matrix.
template < class T, int Order, int Vars >
class TaylorExpansionT : public detail::ShapeBase< Order, Vars >
{
    static_assert( ( Order == Eigen::Dynamic ) == ( Vars == Eigen::Dynamic ),
                   "TaylorExpansionT requires Order and Vars to be both "
                   "non-negative (static) or both Eigen::Dynamic (runtime)." );
    static_assert( Order == Eigen::Dynamic || Order >= 0,
                   "Order must be >= 0 or Eigen::Dynamic." );
    static_assert( Vars == Eigen::Dynamic || Vars >= 1,
                   "Vars must be >= 1 or Eigen::Dynamic." );

  public:
    using Scalar = T;
    static constexpr int OrderAtCompileTime = Order;
    static constexpr int VarsAtCompileTime = Vars;
    static constexpr bool IsStatic =
        ( Order != Eigen::Dynamic ) && ( Vars != Eigen::Dynamic );
    static constexpr bool IsDynamic = !IsStatic;

  private:
    // Compile-time monomial count for the static path; never instantiated for
    // the dynamic path (Coeffs uses Eigen::Dynamic instead).
    static constexpr Eigen::Index coeffs_size =
        IsStatic ? static_cast< Eigen::Index >( util::monomialCount(
                      static_cast< std::size_t >( Order ),
                      static_cast< std::size_t >( Vars ) ) )
                : Eigen::Dynamic;

  public:
    using Coeffs = Eigen::Matrix< T, coeffs_size, 1 >;

    using ShapeBaseT = detail::ShapeBase< Order, Vars >;
    using ShapeBaseT::nvars;
    using ShapeBaseT::order;

    // ---- ctors ---------------------------------------------------------

    TaylorExpansionT() noexcept
        requires IsStatic
        : coeffs_( Coeffs::Zero() )
    {
    }

    explicit TaylorExpansionT( const Coeffs& c ) noexcept
        requires IsStatic
        : coeffs_( c )
    {
    }

    TaylorExpansionT( std::size_t order_v, std::size_t nvars_v )
        requires( !IsStatic )
        : ShapeBaseT( order_v, nvars_v ),
          coeffs_( Coeffs::Zero(
              static_cast< Eigen::Index >( util::monomialCount( order_v, nvars_v ) ) ) )
    {
        assert( nvars_v >= 1 && "TaylorExpansionT requires nvars >= 1" );
    }

    TaylorExpansionT( std::size_t order_v, std::size_t nvars_v, Coeffs c )
        requires( !IsStatic )
        : ShapeBaseT( order_v, nvars_v ), coeffs_( std::move( c ) )
    {
        assert( nvars_v >= 1 && "TaylorExpansionT requires nvars >= 1" );
        assert( static_cast< std::size_t >( coeffs_.size() )
                == util::monomialCount( order_v, nvars_v ) );
    }

    // ---- factories: static path ----------------------------------------

    [[nodiscard]] static TaylorExpansionT zero() noexcept
        requires IsStatic
    {
        return TaylorExpansionT{};
    }

    [[nodiscard]] static TaylorExpansionT one() noexcept
        requires IsStatic
    {
        return constant( T{ 1 } );
    }

    [[nodiscard]] static TaylorExpansionT constant( T c ) noexcept
        requires IsStatic
    {
        TaylorExpansionT out;
        out.coeffs_( 0 ) = c;
        return out;
    }

    // Univariate convenience: x = x0 + dx.
    [[nodiscard]] static TaylorExpansionT variable( T x0 ) noexcept
        requires( IsStatic && Vars == 1 )
    {
        TaylorExpansionT out;
        out.coeffs_( 0 ) = x0;
        if constexpr ( Order >= 1 )
        {
            out.coeffs_( 1 ) = T{ 1 };
        }
        return out;
    }

    // Multivariate variable factory: returns a tuple of Vars independent
    // variables x_i = x0_i + dx_i seeded against the i-th component.
    template < class Vec >
    [[nodiscard]] static auto variables( const Vec& x0 ) noexcept
        requires IsStatic
    {
        return makeVarsImpl( x0, std::make_index_sequence< static_cast< std::size_t >(
                                     Vars ) >{} );
    }

    // Convenience: variables(a, b, ...) accepting Vars scalars.
    template < class... Args >
        requires( IsStatic && sizeof...( Args ) == static_cast< std::size_t >( Vars ) )
    [[nodiscard]] static auto variables( Args... a ) noexcept
    {
        std::array< T, static_cast< std::size_t >( Vars ) > x0{ static_cast< T >( a )... };
        return variables( x0 );
    }

    // ---- factories: dynamic path ---------------------------------------

    [[nodiscard]] static TaylorExpansionT zero( std::size_t order_v,
                                                std::size_t nvars_v )
        requires( !IsStatic )
    {
        return TaylorExpansionT( order_v, nvars_v );
    }

    [[nodiscard]] static TaylorExpansionT one( std::size_t order_v,
                                               std::size_t nvars_v )
        requires( !IsStatic )
    {
        return constant( T{ 1 }, order_v, nvars_v );
    }

    [[nodiscard]] static TaylorExpansionT constant( T c, std::size_t order_v,
                                                    std::size_t nvars_v )
        requires( !IsStatic )
    {
        TaylorExpansionT out( order_v, nvars_v );
        out.coeffs_( 0 ) = c;
        return out;
    }

    [[nodiscard]] static TaylorExpansionT variable( T x0, std::size_t order_v,
                                                    std::size_t nvars_v,
                                                    std::size_t var_idx )
        requires( !IsStatic )
    {
        TaylorExpansionT out( order_v, nvars_v );
        out.coeffs_( 0 ) = x0;
        if ( order_v >= 1 )
        {
            std::vector< std::size_t > a( nvars_v, 0 );
            a[ var_idx ] = 1;
            const std::size_t fi =
                util::flatIndex( std::span< const std::size_t >( a.data(), nvars_v ) );
            out.coeffs_( static_cast< Eigen::Index >( fi ) ) = T{ 1 };
        }
        return out;
    }

    // Returns a vector of independent variables sharing the same truncation
    // order, each seeded against its own dx_i.
    [[nodiscard]] static std::vector< TaylorExpansionT >
    variables( const std::vector< T >& x0, std::size_t order_v )
        requires( !IsStatic )
    {
        const std::size_t M = x0.size();
        std::vector< TaylorExpansionT > out;
        out.reserve( M );
        for ( std::size_t i = 0; i < M; ++i )
        {
            out.push_back( variable( x0[ i ], order_v, M, i ) );
        }
        return out;
    }

    // ---- TaylorExpansion concept methods -------------------------------

    [[nodiscard]] T value() const noexcept
    {
        return coeffs_( 0 );
    }

    [[nodiscard]] T coeff( std::span< const std::size_t > alpha ) const
    {
        return coeffs_( static_cast< Eigen::Index >( util::flatIndex( alpha ) ) );
    }

    // std::array convenience overload for the static path.
    [[nodiscard]] T coeff(
        const std::array< std::size_t, static_cast< std::size_t >( Vars ) >& alpha ) const
        requires IsStatic
    {
        return coeff(
            std::span< const std::size_t >( alpha.data(), static_cast< std::size_t >( Vars ) ) );
    }

    // Compile-time multi-index: `result.coeff<1, 0>()`.
    template < std::size_t... Alpha >
        requires( IsStatic && sizeof...( Alpha ) == static_cast< std::size_t >( Vars ) )
    [[nodiscard]] T coeff() const noexcept
    {
        static constexpr std::array< std::size_t, static_cast< std::size_t >( Vars ) > a{
            Alpha...
        };
        constexpr std::size_t fi = util::flatIndex(
            std::span< const std::size_t >( a.data(), static_cast< std::size_t >( Vars ) ) );
        return coeffs_( static_cast< Eigen::Index >( fi ) );
    }

    // Partial derivative at the expansion centre = alpha! * coeff(alpha).
    [[nodiscard]] T derivative( std::span< const std::size_t > alpha ) const
    {
        const std::size_t f = util::factorial( alpha );
        return coeffs_( static_cast< Eigen::Index >( util::flatIndex( alpha ) ) )
               * static_cast< T >( f );
    }

    [[nodiscard]] T derivative(
        const std::array< std::size_t, static_cast< std::size_t >( Vars ) >& alpha ) const
        requires IsStatic
    {
        return derivative(
            std::span< const std::size_t >( alpha.data(), static_cast< std::size_t >( Vars ) ) );
    }

    template < std::size_t... Alpha >
        requires( IsStatic && sizeof...( Alpha ) == static_cast< std::size_t >( Vars ) )
    [[nodiscard]] T derivative() const noexcept
    {
        static constexpr std::array< std::size_t, static_cast< std::size_t >( Vars ) > a{
            Alpha...
        };
        constexpr std::size_t fi = util::flatIndex(
            std::span< const std::size_t >( a.data(), static_cast< std::size_t >( Vars ) ) );
        constexpr std::size_t f = util::factorial(
            std::span< const std::size_t >( a.data(), static_cast< std::size_t >( Vars ) ) );
        return coeffs_( static_cast< Eigen::Index >( fi ) ) * static_cast< T >( f );
    }

    // Evaluate the truncated polynomial at displacement dx.
    template < class Vec >
    [[nodiscard]] T eval( const Vec& dx ) const
    {
        T acc{ 0 };
        const std::size_t N = order();
        const std::size_t M = nvars();
        for ( std::size_t d = 0; d <= N; ++d )
        {
            T degree_acc{ 0 };
            util::forEachMultiIndexOfDegree(
                d, M, [ & ]( std::span< const std::size_t > a ) {
                    T monom{ 1 };
                    for ( std::size_t k = 0; k < M; ++k )
                    {
                        for ( std::size_t p = 0; p < a[ k ]; ++p )
                        {
                            monom *= dx[ k ];
                        }
                    }
                    degree_acc +=
                        coeffs_( static_cast< Eigen::Index >( util::flatIndex( a ) ) ) * monom;
                } );
            acc += degree_acc;
        }
        return acc;
    }

    // std::array overload of eval for the static path.
    [[nodiscard]] T eval(
        const std::array< T, static_cast< std::size_t >( Vars ) >& dx ) const
        requires IsStatic
    {
        return eval< std::array< T, static_cast< std::size_t >( Vars ) > >( dx );
    }

    // ---- coefficient norms ---------------------------------------------

    [[nodiscard]] T coeffsNormInf() const noexcept
    {
        return coeffs_.cwiseAbs().maxCoeff();
    }

    template < int P >
    [[nodiscard]] T coeffsNorm() const noexcept
    {
        if constexpr ( P == 1 )
        {
            return coeffs_.cwiseAbs().sum();
        }
        else if constexpr ( P == 2 )
        {
            return std::sqrt( coeffs_.squaredNorm() );
        }
        else
        {
            T acc{ 0 };
            for ( Eigen::Index i = 0; i < coeffs_.size(); ++i )
            {
                acc += std::pow( std::abs( coeffs_( i ) ), static_cast< T >( P ) );
            }
            return std::pow( acc, T{ 1 } / static_cast< T >( P ) );
        }
    }

    // ---- raw access ----------------------------------------------------

    [[nodiscard]] T* data() noexcept
    {
        return coeffs_.data();
    }
    [[nodiscard]] const T* data() const noexcept
    {
        return coeffs_.data();
    }

    [[nodiscard]] const Coeffs& coeffs() const noexcept
    {
        return coeffs_;
    }
    [[nodiscard]] Coeffs& coeffs() noexcept
    {
        return coeffs_;
    }

    [[nodiscard]] T rawCoeff( std::size_t i ) const noexcept
    {
        return coeffs_( static_cast< Eigen::Index >( i ) );
    }
    void setRawCoeff( std::size_t i, T v ) noexcept
    {
        coeffs_( static_cast< Eigen::Index >( i ) ) = v;
    }

    // Degree slice as an Eigen vector view.
    [[nodiscard]] auto slice( std::size_t d ) noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }
    [[nodiscard]] auto slice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

    // Streaming: storage is fully populated, so advanceTo is a no-op.
    void advanceTo( std::size_t /*d*/ ) const noexcept
    {
    }

  private:
    Coeffs coeffs_{};

    template < class Vec, std::size_t... Is >
    static auto makeVarsImpl( const Vec& x0, std::index_sequence< Is... > ) noexcept
        requires IsStatic
    {
        return std::tuple{ makeVar( static_cast< T >( x0[ Is ] ), Is )... };
    }

    static TaylorExpansionT makeVar( T x0, std::size_t var_idx ) noexcept
        requires IsStatic
    {
        TaylorExpansionT out;
        out.coeffs_( 0 ) = x0;
        if constexpr ( Order >= 1 )
        {
            // Variable i seeds the degree-1 slot whose multi-index has 1 in
            // position `var_idx`.  In our graded reverse-lex order that is
            // flat index 1 + var_idx in the degree-1 block.
            std::array< std::size_t, static_cast< std::size_t >( Vars ) > a{};
            a[ var_idx ] = 1;
            const std::size_t fi = util::flatIndex( std::span< const std::size_t >(
                a.data(), static_cast< std::size_t >( Vars ) ) );
            out.coeffs_( static_cast< Eigen::Index >( fi ) ) = T{ 1 };
        }
        return out;
    }
};

template < class T, int Order, int Vars >
struct expr_traits< TaylorExpansionT< T, Order, Vars > >
{
    static constexpr bool is_static =
        ( Order != Eigen::Dynamic ) && ( Vars != Eigen::Dynamic );
    static constexpr bool is_dynamic = !is_static;
};

}  // namespace tax

// Out-of-line definition of `expr::Expr<Derived>::eval()`.  Lives here
// because the body materialises the result into a concrete
// `TaylorExpansionT`, which has to be complete at the point of
// definition.
namespace tax::expr
{

template < class Derived >
inline auto Expr< Derived >::eval() const
{
    using D = Derived;
    using S = typename D::Scalar;
    using Out = TaylorExpansionT< S, D::OrderAtCompileTime, D::VarsAtCompileTime >;

    const D& self = static_cast< const D& >( *this );

    auto make_out = [ & ]() {
        if constexpr ( D::IsStatic )
        {
            return Out{};
        }
        else
        {
            return Out( self.order(), self.nvars() );
        }
    };
    Out out = make_out();

    const std::size_t N = self.order();
    const std::size_t M = self.nvars();
    for ( std::size_t d = 0; d <= N; ++d )
    {
        self.advanceTo( d );
        auto out_d = out.slice( d );
        auto in_d = self.slice( d );
        const Eigen::Index n = static_cast< Eigen::Index >( util::degreeSize( d, M ) );
        for ( Eigen::Index i = 0; i < n; ++i )
        {
            out_d.coeffRef( i ) = in_d.coeff( i );
        }
    }
    return out;
}

}  // namespace tax::expr
