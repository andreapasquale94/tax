// SPDX-License-Identifier: BSD-3-Clause
//
// Dynamic-size truncated Taylor expansion: DynamicTaylorExpansion<T>.
//
// The order_ and nvars_ members are runtime-fixed at construction; the
// coefficient buffer is an Eigen::VectorX<T>.  This is the storage type
// exposed to Python via nanobind (no JIT, no variant dispatch over a
// (N, M) grid).  The mathematical kernels are identical to the static
// path; only the size resolution moves from constexpr to runtime.

#pragma once

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "tax/concepts.hpp"
#include "tax/fwd.hpp"
#include "tax/ops/assign.hpp"
#include "tax/util/multi_index.hpp"

namespace tax
{

template < class T >
class DynamicTaylorExpansion
{
  public:
    using Scalar = T;
    static constexpr bool kStatic = false;
    // kOrder / kVars are placeholders for the unified ET tree machinery; on
    // the dynamic path the runtime members order_ / nvars_ are authoritative
    // and the constexpr values are never used to size buffers.
    static constexpr std::size_t kOrder = 0;
    static constexpr std::size_t kVars = 0;

    using Coeffs = Eigen::Matrix< T, Eigen::Dynamic, 1 >;

    // ---- ctors ---------------------------------------------------------

    DynamicTaylorExpansion() noexcept : order_( 0 ), nvars_( 0 )
    {
    }

    DynamicTaylorExpansion( std::size_t order, std::size_t nvars )
        : order_( order ),
          nvars_( nvars ),
          coeffs_( Coeffs::Zero( static_cast< Eigen::Index >(
              util::monomialCount( order, nvars ) ) ) )
    {
        assert( nvars >= 1 && "DynamicTaylorExpansion requires nvars >= 1" );
    }

    DynamicTaylorExpansion( std::size_t order, std::size_t nvars, Coeffs c )
        : order_( order ), nvars_( nvars ), coeffs_( std::move( c ) )
    {
        assert( nvars >= 1 && "DynamicTaylorExpansion requires nvars >= 1" );
        assert( static_cast< std::size_t >( coeffs_.size() )
                == util::monomialCount( order, nvars ) );
    }

    // ---- factories -----------------------------------------------------

    [[nodiscard]] static DynamicTaylorExpansion zero( std::size_t order,
                                                      std::size_t nvars )
    {
        return DynamicTaylorExpansion( order, nvars );
    }

    [[nodiscard]] static DynamicTaylorExpansion one( std::size_t order, std::size_t nvars )
    {
        return constant( T{ 1 }, order, nvars );
    }

    [[nodiscard]] static DynamicTaylorExpansion constant( T c, std::size_t order,
                                                          std::size_t nvars )
    {
        DynamicTaylorExpansion out( order, nvars );
        out.coeffs_( 0 ) = c;
        return out;
    }

    // x_i = x0 + dx_i.  var_idx selects which variable this expansion is
    // seeded against in the dx vector.
    [[nodiscard]] static DynamicTaylorExpansion variable( T x0, std::size_t order,
                                                          std::size_t nvars,
                                                          std::size_t var_idx )
    {
        DynamicTaylorExpansion out( order, nvars );
        out.coeffs_( 0 ) = x0;
        if ( order >= 1 )
        {
            std::vector< std::size_t > a( nvars, 0 );
            a[ var_idx ] = 1;
            const std::size_t fi = util::flatIndex(
                std::span< const std::size_t >( a.data(), nvars ) );
            out.coeffs_( static_cast< Eigen::Index >( fi ) ) = T{ 1 };
        }
        return out;
    }

    // Convenience: build a vector of M independent variables.
    [[nodiscard]] static std::vector< DynamicTaylorExpansion >
    variables( const std::vector< T >& x0, std::size_t order )
    {
        const std::size_t M = x0.size();
        std::vector< DynamicTaylorExpansion > out;
        out.reserve( M );
        for ( std::size_t i = 0; i < M; ++i )
        {
            out.push_back( variable( x0[ i ], order, M, i ) );
        }
        return out;
    }

    // ---- TaylorExpansion concept methods -------------------------------

    [[nodiscard]] std::size_t order() const noexcept
    {
        return order_;
    }
    [[nodiscard]] std::size_t nvars() const noexcept
    {
        return nvars_;
    }

    [[nodiscard]] T value() const noexcept
    {
        return coeffs_( 0 );
    }

    [[nodiscard]] T coeff( std::span< const std::size_t > alpha ) const
    {
        return coeffs_( static_cast< Eigen::Index >( util::flatIndex( alpha ) ) );
    }

    [[nodiscard]] T derivative( std::span< const std::size_t > alpha ) const
    {
        const std::size_t f = util::factorial( alpha );
        return coeffs_( static_cast< Eigen::Index >( util::flatIndex( alpha ) ) )
               * static_cast< T >( f );
    }

    template < class Vec >
    [[nodiscard]] T eval( const Vec& dx ) const
    {
        T acc{ 0 };
        for ( std::size_t d = 0; d <= order_; ++d )
        {
            util::forEachMultiIndexOfDegree(
                d, nvars_, [ & ]( std::span< const std::size_t > a ) {
                    T monom{ 1 };
                    for ( std::size_t k = 0; k < nvars_; ++k )
                    {
                        for ( std::size_t p = 0; p < a[ k ]; ++p )
                        {
                            monom *= dx[ k ];
                        }
                    }
                    acc += coeffs_( static_cast< Eigen::Index >( util::flatIndex( a ) ) )
                           * monom;
                } );
        }
        return acc;
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

    [[nodiscard]] auto degreeSlice( std::size_t d ) noexcept
    {
        return coeffs_.segment( static_cast< Eigen::Index >( util::degreeOffset( d, nvars_ ) ),
                                static_cast< Eigen::Index >( util::degreeSize( d, nvars_ ) ) );
    }
    [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
    {
        return coeffs_.segment( static_cast< Eigen::Index >( util::degreeOffset( d, nvars_ ) ),
                                static_cast< Eigen::Index >( util::degreeSize( d, nvars_ ) ) );
    }

    void advanceTo( std::size_t /*d*/ ) const noexcept
    {
    }

    template < class Expr >
        requires StreamingExpression< std::remove_cvref_t< Expr > >
    DynamicTaylorExpansion& operator<<=( Expr&& expr )
    {
        detail::streamingAssign( *this, expr );
        return *this;
    }

  private:
    std::size_t order_{};
    std::size_t nvars_{};
    Coeffs coeffs_{};
};

template < class T >
struct expr_traits< DynamicTaylorExpansion< T > >
{
    static constexpr bool is_static = false;
    static constexpr bool is_dynamic = true;
};

}  // namespace tax
