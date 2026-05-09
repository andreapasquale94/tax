// SPDX-License-Identifier: BSD-3-Clause
//
// Buffered ET nodes.
//
// Each one owns a `coeffs_` buffer (sized like a TTE in the same kind),
// drives the per-degree kernel through a monotonic `advanceTo`, and exposes
// its own degree slices as Eigen::VectorBlocks.  Internal mutable state
// keeps the public interface const-callable so a buffered node can sit
// behind a `const&` in a parent ET tree.

#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cmath>

#include "tax/expr/base.hpp"
#include "tax/kernels/cauchy.hpp"
#include "tax/kernels/elementary.hpp"
#include "tax/kernels/exp_log.hpp"
#include "tax/kernels/trig.hpp"
#include "tax/util/multi_index.hpp"

namespace tax::expr
{

namespace detail
{

template < class E >
[[nodiscard]] inline auto rawSlice( E& self, std::size_t d ) noexcept
{
    return self.coeffs_buffer().segment(
        static_cast< Eigen::Index >( util::degreeOffset( d, self.nvars() ) ),
        static_cast< Eigen::Index >( util::degreeSize( d, self.nvars() ) ) );
}

}  // namespace detail

// ----------------------------------------------------------------------
// MulExpr: Cauchy product.
template < class L, class R >
    requires SameKindExpression< L, R >
class MulExpr : public Expr< MulExpr< L, R > >
{
  public:
    using Scalar = typename L::Scalar;
    static constexpr bool kStatic = L::kStatic;
    static constexpr std::size_t kOrder = L::kOrder;
    static constexpr std::size_t kVars = L::kVars;
    using Coeffs = detail::coeffs_for_t< MulExpr >;

    MulExpr( const L& l, const R& r )
        : lhs_( l ), rhs_( r ), coeffs_( detail::makeCoeffsLike( l ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept
    {
        return lhs_.order();
    }
    [[nodiscard]] std::size_t nvars() const noexcept
    {
        return lhs_.nvars();
    }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ )
        {
            return;
        }
        lhs_.advanceTo( d );
        rhs_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            auto out_e = degreeSlice( e );
            kernels::cauchyMulComputeDegree< Scalar >( e, nvars(), lhs_, rhs_, out_e );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    etstore_t< L > lhs_;
    etstore_t< R > rhs_;
    mutable Coeffs coeffs_;
    mutable std::size_t next_{ 0 };
};

// ----------------------------------------------------------------------
// DivExpr: a / b.
template < class L, class R >
    requires SameKindExpression< L, R >
class DivExpr : public Expr< DivExpr< L, R > >
{
  public:
    using Scalar = typename L::Scalar;
    static constexpr bool kStatic = L::kStatic;
    static constexpr std::size_t kOrder = L::kOrder;
    static constexpr std::size_t kVars = L::kVars;
    using Coeffs = detail::coeffs_for_t< DivExpr >;

    DivExpr( const L& l, const R& r )
        : lhs_( l ), rhs_( r ), coeffs_( detail::makeCoeffsLike( l ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept
    {
        return lhs_.order();
    }
    [[nodiscard]] std::size_t nvars() const noexcept
    {
        return lhs_.nvars();
    }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ )
        {
            return;
        }
        lhs_.advanceTo( d );
        rhs_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            kernels::divComputeDegree< Scalar >( e, nvars(), lhs_, rhs_, *this );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    etstore_t< L > lhs_;
    etstore_t< R > rhs_;
    mutable Coeffs coeffs_;
    mutable std::size_t next_{ 0 };
};

// ----------------------------------------------------------------------
// SquareExpr: x * x without re-walking the lhs/rhs sub-trees twice.
template < class E >
    requires TaxExpression< E >
class SquareExpr : public Expr< SquareExpr< E > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool kStatic = E::kStatic;
    static constexpr std::size_t kOrder = E::kOrder;
    static constexpr std::size_t kVars = E::kVars;
    using Coeffs = detail::coeffs_for_t< SquareExpr >;

    explicit SquareExpr( const E& e )
        : inner_( e ), coeffs_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept
    {
        return inner_.order();
    }
    [[nodiscard]] std::size_t nvars() const noexcept
    {
        return inner_.nvars();
    }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ )
        {
            return;
        }
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            auto out_e = degreeSlice( e );
            kernels::squareComputeDegree< Scalar >( e, nvars(), inner_, out_e );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    etstore_t< E > inner_;
    mutable Coeffs coeffs_;
    mutable std::size_t next_{ 0 };
};

// ----------------------------------------------------------------------
// SqrtExpr.
template < class E >
    requires TaxExpression< E >
class SqrtExpr : public Expr< SqrtExpr< E > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool kStatic = E::kStatic;
    static constexpr std::size_t kOrder = E::kOrder;
    static constexpr std::size_t kVars = E::kVars;
    using Coeffs = detail::coeffs_for_t< SqrtExpr >;

    explicit SqrtExpr( const E& e )
        : inner_( e ), coeffs_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept
    {
        return inner_.order();
    }
    [[nodiscard]] std::size_t nvars() const noexcept
    {
        return inner_.nvars();
    }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ )
        {
            return;
        }
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            kernels::sqrtComputeDegree< Scalar >( e, nvars(), inner_, *this );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    etstore_t< E > inner_;
    mutable Coeffs coeffs_;
    mutable std::size_t next_{ 0 };
};

// ----------------------------------------------------------------------
// ExpExpr.
template < class E >
    requires TaxExpression< E >
class ExpExpr : public Expr< ExpExpr< E > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool kStatic = E::kStatic;
    static constexpr std::size_t kOrder = E::kOrder;
    static constexpr std::size_t kVars = E::kVars;
    using Coeffs = detail::coeffs_for_t< ExpExpr >;

    explicit ExpExpr( const E& e )
        : inner_( e ), coeffs_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept
    {
        return inner_.order();
    }
    [[nodiscard]] std::size_t nvars() const noexcept
    {
        return inner_.nvars();
    }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ )
        {
            return;
        }
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            kernels::expComputeDegree< Scalar >( e, nvars(), inner_, *this );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    etstore_t< E > inner_;
    mutable Coeffs coeffs_;
    mutable std::size_t next_{ 0 };
};

// ----------------------------------------------------------------------
// LogExpr.
template < class E >
    requires TaxExpression< E >
class LogExpr : public Expr< LogExpr< E > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool kStatic = E::kStatic;
    static constexpr std::size_t kOrder = E::kOrder;
    static constexpr std::size_t kVars = E::kVars;
    using Coeffs = detail::coeffs_for_t< LogExpr >;

    explicit LogExpr( const E& e )
        : inner_( e ), coeffs_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept
    {
        return inner_.order();
    }
    [[nodiscard]] std::size_t nvars() const noexcept
    {
        return inner_.nvars();
    }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ )
        {
            return;
        }
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            kernels::logComputeDegree< Scalar >( e, nvars(), inner_, *this );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    etstore_t< E > inner_;
    mutable Coeffs coeffs_;
    mutable std::size_t next_{ 0 };
};

// ----------------------------------------------------------------------
// Sin / Cos paired evaluator.  Each ET carries an internal companion
// buffer (the "other" side of the recurrence) so that user code can call
// either sin or cos without affecting the other branch's allocations.

template < class E, bool ReturnSin >
    requires TaxExpression< E >
class SinCosExpr : public Expr< SinCosExpr< E, ReturnSin > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool kStatic = E::kStatic;
    static constexpr std::size_t kOrder = E::kOrder;
    static constexpr std::size_t kVars = E::kVars;
    using Coeffs = detail::coeffs_for_t< SinCosExpr >;

    explicit SinCosExpr( const E& e )
        : inner_( e ),
          sin_( detail::makeCoeffsLike( e ) ),
          cos_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept
    {
        return inner_.order();
    }
    [[nodiscard]] std::size_t nvars() const noexcept
    {
        return inner_.nvars();
    }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ )
        {
            return;
        }
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            kernels::sinCosComputeDegree< Scalar >( e, nvars(), inner_, sinView(),
                                                    cosView() );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
    {
        if constexpr ( ReturnSin )
        {
            return sin_.segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
        }
        else
        {
            return cos_.segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
        }
    }

  private:
    // Wrappers exposing the sin/cos buffers in the slice-providing
    // interface that the kernel expects.
    struct SideView
    {
        Coeffs* buf;
        std::size_t nvars_v;
        [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
        {
            return buf->segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars_v ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars_v ) ) );
        }
    };

    [[nodiscard]] SideView sinView() const noexcept
    {
        return { &sin_, nvars() };
    }
    [[nodiscard]] SideView cosView() const noexcept
    {
        return { &cos_, nvars() };
    }

    etstore_t< E > inner_;
    mutable Coeffs sin_;
    mutable Coeffs cos_;
    mutable std::size_t next_{ 0 };
};

template < class E >
using SinExpr = SinCosExpr< E, true >;
template < class E >
using CosExpr = SinCosExpr< E, false >;

// ----------------------------------------------------------------------
// sinh / cosh paired evaluator (same shape, no sign flip in cos).
template < class E, bool ReturnSinh >
    requires TaxExpression< E >
class SinhCoshExpr : public Expr< SinhCoshExpr< E, ReturnSinh > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool kStatic = E::kStatic;
    static constexpr std::size_t kOrder = E::kOrder;
    static constexpr std::size_t kVars = E::kVars;
    using Coeffs = detail::coeffs_for_t< SinhCoshExpr >;

    explicit SinhCoshExpr( const E& e )
        : inner_( e ),
          sinh_( detail::makeCoeffsLike( e ) ),
          cosh_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept
    {
        return inner_.order();
    }
    [[nodiscard]] std::size_t nvars() const noexcept
    {
        return inner_.nvars();
    }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ )
        {
            return;
        }
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            kernels::sinhCoshComputeDegree< Scalar >( e, nvars(), inner_, sinhView(),
                                                      coshView() );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
    {
        if constexpr ( ReturnSinh )
        {
            return sinh_.segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
        }
        else
        {
            return cosh_.segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
        }
    }

  private:
    struct SideView
    {
        Coeffs* buf;
        std::size_t nvars_v;
        [[nodiscard]] auto degreeSlice( std::size_t d ) const noexcept
        {
            return buf->segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars_v ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars_v ) ) );
        }
    };

    [[nodiscard]] SideView sinhView() const noexcept
    {
        return { &sinh_, nvars() };
    }
    [[nodiscard]] SideView coshView() const noexcept
    {
        return { &cosh_, nvars() };
    }

    etstore_t< E > inner_;
    mutable Coeffs sinh_;
    mutable Coeffs cosh_;
    mutable std::size_t next_{ 0 };
};

template < class E >
using SinhExpr = SinhCoshExpr< E, true >;
template < class E >
using CoshExpr = SinhCoshExpr< E, false >;

}  // namespace tax::expr
