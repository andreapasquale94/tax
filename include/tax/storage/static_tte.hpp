// SPDX-License-Identifier: BSD-3-Clause
//
// Static-size truncated Taylor expansion: TruncatedTaylorExpansionT<T,N,M>.
//
// Coefficients live in an Eigen::Matrix<T, monomialCount(N,M), 1> on the
// stack.  All sizes are constexpr.  This is the C++ hot path; user code
// pays no allocation overhead and the compiler can inline through every
// kernel call.

#pragma once

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>

#include "tax/concepts.hpp"
#include "tax/fwd.hpp"
#include "tax/ops/assign.hpp"
#include "tax/util/multi_index.hpp"

namespace tax
{

template < class T, std::size_t Order, std::size_t Vars >
class TruncatedTaylorExpansionT
{
    static_assert( Vars >= 1, "Vars (M) must be >= 1" );

  public:
    using Scalar = T;
    static constexpr bool kStatic = true;
    static constexpr std::size_t kOrder = Order;
    static constexpr std::size_t kVars = Vars;
    static constexpr std::size_t kSize = util::monomialCount( Order, Vars );

    using Coeffs = Eigen::Matrix< T, static_cast< Eigen::Index >( kSize ), 1 >;

    // ---- ctors ---------------------------------------------------------

    TruncatedTaylorExpansionT() noexcept : coeffs_( Coeffs::Zero() )
    {
    }

    explicit TruncatedTaylorExpansionT( const Coeffs& c ) noexcept : coeffs_( c )
    {
    }

    // ---- factories -----------------------------------------------------

    [[nodiscard]] static TruncatedTaylorExpansionT zero() noexcept
    {
        return TruncatedTaylorExpansionT{};
    }

    [[nodiscard]] static TruncatedTaylorExpansionT one() noexcept
    {
        return constant( T{ 1 } );
    }

    [[nodiscard]] static TruncatedTaylorExpansionT constant( T c ) noexcept
    {
        TruncatedTaylorExpansionT out;
        out.coeffs_( 0 ) = c;
        return out;
    }

    // Univariate convenience: x = x0 + dx.
    [[nodiscard]] static TruncatedTaylorExpansionT variable( T x0 ) noexcept
        requires( Vars == 1 )
    {
        TruncatedTaylorExpansionT out;
        out.coeffs_( 0 ) = x0;
        if constexpr ( Order >= 1 )
        {
            out.coeffs_( 1 ) = T{ 1 };
        }
        return out;
    }

    // Multivariate variable factory: returns a tuple of M independent
    // variables x_i = x0_i + dx_i with seeds at the first-degree slot.
    template < class Vec >
    [[nodiscard]] static auto variables( const Vec& x0 ) noexcept
    {
        return makeVarsImpl( x0, std::make_index_sequence< Vars >{} );
    }

    // Convenience overload: variables(a, b, ...) accepting M scalars.
    template < class... Args >
        requires( sizeof...( Args ) == Vars )
    [[nodiscard]] static auto variables( Args... a ) noexcept
    {
        std::array< T, Vars > x0{ static_cast< T >( a )... };
        return variables( x0 );
    }

    // ---- TaylorExpansion concept methods -------------------------------

    [[nodiscard]] static constexpr std::size_t order() noexcept
    {
        return Order;
    }
    [[nodiscard]] static constexpr std::size_t nvars() noexcept
    {
        return Vars;
    }

    [[nodiscard]] T value() const noexcept
    {
        return coeffs_( 0 );
    }

    [[nodiscard]] T coeff( std::span< const std::size_t > alpha ) const
    {
        return coeffs_( static_cast< Eigen::Index >( util::flatIndex( alpha ) ) );
    }

    // std::array convenience overload — lets callers write
    // `result.coeff({1, 0})` without going through std::span.
    [[nodiscard]] T coeff( const std::array< std::size_t, Vars >& alpha ) const
    {
        return coeff( std::span< const std::size_t >( alpha.data(), Vars ) );
    }

    // Compile-time multi-index overload: `result.coeff<1, 0>()`.  All
    // index arithmetic resolves at compile time; the runtime lookup is a
    // single constant-offset Eigen access.
    template < std::size_t... Alpha >
        requires( sizeof...( Alpha ) == Vars )
    [[nodiscard]] T coeff() const noexcept
    {
        static constexpr std::array< std::size_t, Vars > a{ Alpha... };
        constexpr std::size_t fi =
            util::flatIndex( std::span< const std::size_t >( a.data(), Vars ) );
        return coeffs_( static_cast< Eigen::Index >( fi ) );
    }

    // Partial derivative at the expansion centre = alpha! * coeff(alpha).
    [[nodiscard]] T derivative( std::span< const std::size_t > alpha ) const
    {
        const std::size_t f = util::factorial( alpha );
        return coeffs_( static_cast< Eigen::Index >( util::flatIndex( alpha ) ) )
               * static_cast< T >( f );
    }

    [[nodiscard]] T derivative( const std::array< std::size_t, Vars >& alpha ) const
    {
        return derivative( std::span< const std::size_t >( alpha.data(), Vars ) );
    }

    // Compile-time multi-index overload: `result.derivative<1, 1>()`.
    template < std::size_t... Alpha >
        requires( sizeof...( Alpha ) == Vars )
    [[nodiscard]] T derivative() const noexcept
    {
        static constexpr std::array< std::size_t, Vars > a{ Alpha... };
        constexpr std::size_t fi =
            util::flatIndex( std::span< const std::size_t >( a.data(), Vars ) );
        constexpr std::size_t f =
            util::factorial( std::span< const std::size_t >( a.data(), Vars ) );
        return coeffs_( static_cast< Eigen::Index >( fi ) ) * static_cast< T >( f );
    }

    // Evaluate the truncated polynomial at displacement dx.
    template < class Vec >
    [[nodiscard]] T eval( const Vec& dx ) const
    {
        T acc{ 0 };
        for ( std::size_t d = 0; d <= Order; ++d )
        {
            T degree_acc{ 0 };
            util::forEachMultiIndexOfDegree(
                d, Vars, [ & ]( std::span< const std::size_t > a ) {
                    T monom{ 1 };
                    for ( std::size_t k = 0; k < Vars; ++k )
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

    // std::array overload of eval so callers can write
    // `result.eval({0.1, 0.05})` against the brief's example.
    [[nodiscard]] T eval( const std::array< T, Vars >& dx ) const
    {
        return eval< std::array< T, Vars > >( dx );
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

    // Raw flat coefficient access by graded reverse-lex index.
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
        return coeffs_.segment( static_cast< Eigen::Index >( util::degreeOffset( d, Vars ) ),
                                static_cast< Eigen::Index >( util::degreeSize( d, Vars ) ) );
    }
    [[nodiscard]] auto slice( std::size_t d ) const noexcept
    {
        return coeffs_.segment( static_cast< Eigen::Index >( util::degreeOffset( d, Vars ) ),
                                static_cast< Eigen::Index >( util::degreeSize( d, Vars ) ) );
    }

    // Streaming: storage is fully populated, so advanceTo is a no-op.
    void advanceTo( std::size_t /*d*/ ) const noexcept
    {
    }

    // ---- <<= driver: realize a streaming expression into this storage --

    template < class Expr >
        requires StreamingExpression< std::remove_cvref_t< Expr > >
    TruncatedTaylorExpansionT& operator<<=( Expr&& expr )
    {
        detail::streamingAssign( *this, expr );
        return *this;
    }

  private:
    Coeffs coeffs_{};

    template < class Vec, std::size_t... Is >
    static auto makeVarsImpl( const Vec& x0, std::index_sequence< Is... > ) noexcept
    {
        return std::tuple{ makeVar( static_cast< T >( x0[ Is ] ), Is )... };
    }

    static TruncatedTaylorExpansionT makeVar( T x0, std::size_t var_idx ) noexcept
    {
        TruncatedTaylorExpansionT out;
        out.coeffs_( 0 ) = x0;
        if constexpr ( Order >= 1 )
        {
            // Variable i seeds the degree-1 slot whose multi-index has 1 in
            // position `var_idx`.  In our graded reverse-lex order the
            // degree-1 block is laid out so flat index = 1 + var_idx for
            // variable var_idx in the M-element multi-index.
            std::array< std::size_t, Vars > a{};
            a[ var_idx ] = 1;
            const std::size_t fi = util::flatIndex(
                std::span< const std::size_t >( a.data(), Vars ) );
            out.coeffs_( static_cast< Eigen::Index >( fi ) ) = T{ 1 };
        }
        return out;
    }
};

template < class T, std::size_t Order, std::size_t Vars >
struct expr_traits< TruncatedTaylorExpansionT< T, Order, Vars > >
{
    static constexpr bool is_static = true;
    static constexpr bool is_dynamic = false;
};

}  // namespace tax
