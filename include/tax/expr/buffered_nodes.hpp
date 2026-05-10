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
#include "tax/kernels/inverse_trig.hpp"
#include "tax/kernels/trig.hpp"
#include "tax/util/multi_index.hpp"

namespace tax::expr
{

// ----------------------------------------------------------------------
// MulExpr: Cauchy product.
template < class L, class R >
    requires SameKindExpression< L, R >
class MulExpr : public Expr< MulExpr< L, R > >
{
  public:
    using Scalar = typename L::Scalar;
    static constexpr bool IsStatic = L::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = L::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = L::VarsAtCompileTime;
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
            auto out_e = slice( e );
            kernels::cauchyMulComputeDegree< Scalar >( e, nvars(), lhs_, rhs_, out_e );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
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
    static constexpr bool IsStatic = L::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = L::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = L::VarsAtCompileTime;
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

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
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
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
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
            auto out_e = slice( e );
            kernels::squareComputeDegree< Scalar >( e, nvars(), inner_, out_e );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
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
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
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

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
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
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
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

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
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
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
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

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
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
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
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

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
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
        [[nodiscard]] auto slice( std::size_t d ) const noexcept
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
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
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

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
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
        [[nodiscard]] auto slice( std::size_t d ) const noexcept
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

// ----------------------------------------------------------------------
// SinCosNodeExpr / SinhCoshNodeExpr — shared cores used by tax::sincos /
// tax::sinhcosh.  Same recurrences as SinCosExpr / SinhCoshExpr but the
// node is meant to be held by a `SinCosPair` / `SinhCoshPair` owner and
// referenced by lightweight view ETs (see `SinCosPairView` /
// `SinhCoshPairView` below).  This way, both the sin and cos branches
// of a single `tax::sincos(x)` call share one set of buffers, and the
// joint recurrence runs only once.
template < class E >
    requires TaxExpression< E >
class SinCosNodeExpr
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
    using Coeffs = detail::coeffs_for_t< SinCosNodeExpr >;

    explicit SinCosNodeExpr( const E& e )
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
            kernels::sinCosComputeDegree< Scalar >( e, nvars(), inner_, sinSide(),
                                                    cosSide() );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto sinSlice( std::size_t d ) const noexcept
    {
        return sin_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }
    [[nodiscard]] auto cosSlice( std::size_t d ) const noexcept
    {
        return cos_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    struct SideView
    {
        Coeffs* buf;
        std::size_t nvars_v;
        [[nodiscard]] auto slice( std::size_t d ) const noexcept
        {
            return buf->segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars_v ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars_v ) ) );
        }
    };
    [[nodiscard]] SideView sinSide() const noexcept { return { &sin_, nvars() }; }
    [[nodiscard]] SideView cosSide() const noexcept { return { &cos_, nvars() }; }

    etstore_t< E > inner_;
    mutable Coeffs sin_;
    mutable Coeffs cos_;
    mutable std::size_t next_{ 0 };
};

template < class E >
    requires TaxExpression< E >
class SinhCoshNodeExpr
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
    using Coeffs = detail::coeffs_for_t< SinhCoshNodeExpr >;

    explicit SinhCoshNodeExpr( const E& e )
        : inner_( e ),
          sinh_( detail::makeCoeffsLike( e ) ),
          cosh_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept { return inner_.order(); }
    [[nodiscard]] std::size_t nvars() const noexcept { return inner_.nvars(); }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ ) return;
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            kernels::sinhCoshComputeDegree< Scalar >( e, nvars(), inner_, sinhSide(),
                                                      coshSide() );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto sinhSlice( std::size_t d ) const noexcept
    {
        return sinh_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }
    [[nodiscard]] auto coshSlice( std::size_t d ) const noexcept
    {
        return cosh_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    struct SideView
    {
        Coeffs* buf;
        std::size_t nvars_v;
        [[nodiscard]] auto slice( std::size_t d ) const noexcept
        {
            return buf->segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars_v ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars_v ) ) );
        }
    };
    [[nodiscard]] SideView sinhSide() const noexcept { return { &sinh_, nvars() }; }
    [[nodiscard]] SideView coshSide() const noexcept { return { &cosh_, nvars() }; }

    etstore_t< E > inner_;
    mutable Coeffs sinh_;
    mutable Coeffs cosh_;
    mutable std::size_t next_{ 0 };
};

// View ETs that reference a shared SinCos / SinhCosh node by const& and
// expose one of its two buffers via slice(d).  These satisfy the
// TaxExpression / StreamingExpression contract; users get them by
// calling .sin() / .cos() (or .sinh() / .cosh()) on a Pair owner.
//
// Lifetime caveat: the Pair must outlive the view.  The view is meant
// to be consumed in the same enclosing expression that produced it
// (e.g. `out <<= pair.sin();` is the canonical usage).
template < class Node, bool ReturnSin >
class SinCosPairView : public Expr< SinCosPairView< Node, ReturnSin > >
{
  public:
    using Scalar = typename Node::Scalar;
    static constexpr bool IsStatic = Node::IsStatic;
    static constexpr bool IsDynamic = Node::IsDynamic;
    static constexpr int OrderAtCompileTime = Node::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = Node::VarsAtCompileTime;

    explicit SinCosPairView( const Node& n ) noexcept : node_( n )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept { return node_.order(); }
    [[nodiscard]] std::size_t nvars() const noexcept { return node_.nvars(); }

    void advanceTo( std::size_t d ) const { node_.advanceTo( d ); }

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
    {
        if constexpr ( ReturnSin )
        {
            return node_.sinSlice( d );
        }
        else
        {
            return node_.cosSlice( d );
        }
    }

  private:
    const Node& node_;
};

template < class Node, bool ReturnSinh >
class SinhCoshPairView : public Expr< SinhCoshPairView< Node, ReturnSinh > >
{
  public:
    using Scalar = typename Node::Scalar;
    static constexpr bool IsStatic = Node::IsStatic;
    static constexpr bool IsDynamic = Node::IsDynamic;
    static constexpr int OrderAtCompileTime = Node::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = Node::VarsAtCompileTime;

    explicit SinhCoshPairView( const Node& n ) noexcept : node_( n )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept { return node_.order(); }
    [[nodiscard]] std::size_t nvars() const noexcept { return node_.nvars(); }

    void advanceTo( std::size_t d ) const { node_.advanceTo( d ); }

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
    {
        if constexpr ( ReturnSinh )
        {
            return node_.sinhSlice( d );
        }
        else
        {
            return node_.coshSlice( d );
        }
    }

  private:
    const Node& node_;
};

// ----------------------------------------------------------------------
// CbrtExpr — cube root with a maintained F^2 auxiliary buffer.
template < class E >
    requires TaxExpression< E >
class CbrtExpr : public Expr< CbrtExpr< E > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
    using Coeffs = detail::coeffs_for_t< CbrtExpr >;

    explicit CbrtExpr( const E& e )
        : inner_( e ),
          coeffs_( detail::makeCoeffsLike( e ) ),
          aux_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept { return inner_.order(); }
    [[nodiscard]] std::size_t nvars() const noexcept { return inner_.nvars(); }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ ) return;
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            kernels::cbrtComputeDegree< Scalar >( e, nvars(), inner_, *this, auxView() );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    struct SideView
    {
        Coeffs* buf;
        std::size_t nvars_v;
        [[nodiscard]] auto slice( std::size_t d ) const noexcept
        {
            return buf->segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars_v ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars_v ) ) );
        }
    };
    [[nodiscard]] SideView auxView() const noexcept { return { &aux_, nvars() }; }

    etstore_t< E > inner_;
    mutable Coeffs coeffs_;
    mutable Coeffs aux_;
    mutable std::size_t next_{ 0 };
};

// ----------------------------------------------------------------------
// PowRealExpr — pow(u, p) with `p` a runtime real exponent.  Uses the
// `u F' = p u' F` recurrence and only needs the output buffer (no aux).
template < class E >
    requires TaxExpression< E >
class PowRealExpr : public Expr< PowRealExpr< E > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
    using Coeffs = detail::coeffs_for_t< PowRealExpr >;

    PowRealExpr( const E& e, Scalar p )
        : inner_( e ), exponent_( p ), coeffs_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept { return inner_.order(); }
    [[nodiscard]] std::size_t nvars() const noexcept { return inner_.nvars(); }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ ) return;
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            kernels::powRealComputeDegree< Scalar >( e, nvars(), exponent_, inner_,
                                                     *this );
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    etstore_t< E > inner_;
    Scalar exponent_;
    mutable Coeffs coeffs_;
    mutable std::size_t next_{ 0 };
};

// ----------------------------------------------------------------------
// Inverse-trig / inverse-hyperbolic node template.
//
// All of {atan, atanh, asin, acos, asinh, acosh} share the recurrence
// `G(u) * E[F] = sign * E[u]` with a different `G(u)` and `sign`.
// We parametrise on:
//   - GMode = 0: G = 1 + u^2     (atan)
//   - GMode = 1: G = 1 - u^2     (atanh)
//   - GMode = 2: G = sqrt(1 - u^2)  (asin, acos)
//   - GMode = 3: G = sqrt(1 + u^2)  (asinh)
//   - GMode = 4: G = sqrt(u^2 - 1)  (acosh)
//
// `sign` is +1 except for acos which uses -1.
//
// The aux buffer stores G fully; the output buffer stores F.
namespace detail
{

// G's degree-d slice from u's lower slices and aux's lower slices.
template < int GMode, class T, class U, class GBuf >
inline void fillGSlice( std::size_t d, std::size_t nvars, const U& u, const GBuf& g_obj )
{
    using namespace tax::util;
    auto g_d = g_obj.slice( d );
    const std::size_t dsize = degreeSize( d, nvars );

    if constexpr ( GMode == 0 || GMode == 1 )
    {
        // G = 1 +/- u^2.  G_0 = 1 + sign_u_sq * u_0^2; G_d = sign_u_sq * (u^2)_d.
        constexpr T sign_u_sq = ( GMode == 0 ) ? T{ 1 } : T{ -1 };
        for ( std::size_t i = 0; i < dsize; ++i )
        {
            g_d.coeffRef( static_cast< Eigen::Index >( i ) ) = T{ 0 };
        }
        if ( d == 0 )
        {
            const T u0 = static_cast< T >( u.slice( 0 ).coeff( 0 ) );
            g_d.coeffRef( 0 ) = T{ 1 } + sign_u_sq * u0 * u0;
            return;
        }
        for ( std::size_t e = 0; e <= d; ++e )
        {
            auto u_e = u.slice( e );
            auto u_de = u.slice( d - e );
            kernels::cauchyAccumulateSlice< T >( e, d - e, nvars, sign_u_sq, u_e, u_de, g_d );
        }
    }
    else
    {
        // G = sqrt(H), where H = 1 - u^2 (mode 2), 1 + u^2 (mode 3),
        // u^2 - 1 (mode 4).  Use the sqrt recurrence on H:
        //   G_0 = sqrt(H_0)
        //   G_d = (H_d - sum_{1<=|beta|<=d-1} G_beta G_{d-beta}) / (2 G_0)
        constexpr T sign_u_sq = ( GMode == 2 ) ? T{ -1 } : T{ 1 };
        constexpr T h_const = ( GMode == 2 ) ? T{ 1 } : ( GMode == 3 ) ? T{ 1 } : T{ -1 };
        if ( d == 0 )
        {
            const T u0 = static_cast< T >( u.slice( 0 ).coeff( 0 ) );
            const T h0 = h_const + sign_u_sq * u0 * u0;
            g_d.coeffRef( 0 ) = std::sqrt( h0 );
            return;
        }
        // Initialise g_d = H_d = sign_u_sq * (u^2)_d (no constant; it lived at d=0).
        for ( std::size_t i = 0; i < dsize; ++i )
        {
            g_d.coeffRef( static_cast< Eigen::Index >( i ) ) = T{ 0 };
        }
        for ( std::size_t e = 0; e <= d; ++e )
        {
            auto u_e = u.slice( e );
            auto u_de = u.slice( d - e );
            kernels::cauchyAccumulateSlice< T >( e, d - e, nvars, sign_u_sq, u_e, u_de, g_d );
        }
        // Subtract sum_{1 <= |beta| <= d-1} G_beta G_{d-beta}.
        for ( std::size_t e = 1; e + 1 <= d; ++e )
        {
            auto g_e = g_obj.slice( e );
            auto g_de = g_obj.slice( d - e );
            kernels::cauchyAccumulateSlice< T >( e, d - e, nvars, T{ -1 }, g_e, g_de, g_d );
        }
        const T inv_2g0 = T{ 1 } / ( T{ 2 } * static_cast< T >( g_obj.slice( 0 ).coeff( 0 ) ) );
        for ( std::size_t i = 0; i < dsize; ++i )
        {
            g_d.coeffRef( static_cast< Eigen::Index >( i ) ) *= inv_2g0;
        }
    }
}

template < class T >
inline T inverseFunctionAt( int kind, T u0 )
{
    switch ( kind )
    {
        case 0:
            return std::atan( u0 );
        case 1:
            return std::atanh( u0 );
        case 2:
            return std::asin( u0 );
        case 3:
            return std::acos( u0 );
        case 4:
            return std::asinh( u0 );
        case 5:
            return std::acosh( u0 );
    }
    return T{ 0 };
}

}  // namespace detail

// FunKind: which inverse-function value sits at F_0; GMode: which auxiliary.
template < class E, int FunKind, int GMode, int Sign >
    requires TaxExpression< E >
class InverseFunctionExpr : public Expr< InverseFunctionExpr< E, FunKind, GMode, Sign > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
    using Coeffs = detail::coeffs_for_t< InverseFunctionExpr >;

    explicit InverseFunctionExpr( const E& e )
        : inner_( e ),
          coeffs_( detail::makeCoeffsLike( e ) ),
          g_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept { return inner_.order(); }
    [[nodiscard]] std::size_t nvars() const noexcept { return inner_.nvars(); }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ ) return;
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            // 1) Fill G_e.
            detail::fillGSlice< GMode, Scalar >( e, nvars(), inner_, gView() );
            // 2) Fill F_e.
            if ( e == 0 )
            {
                const Scalar u0 =
                    static_cast< Scalar >( inner_.slice( 0 ).coeff( 0 ) );
                slice( 0 ).coeffRef( 0 ) = detail::inverseFunctionAt< Scalar >( FunKind, u0 );
            }
            else
            {
                kernels::eulerAuxRecurrenceComputeDegree< Scalar >(
                    e, nvars(), static_cast< Scalar >( Sign ), inner_, gView(), *this );
            }
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    struct SideView
    {
        Coeffs* buf;
        std::size_t nvars_v;
        [[nodiscard]] auto slice( std::size_t d ) const noexcept
        {
            return buf->segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars_v ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars_v ) ) );
        }
    };
    [[nodiscard]] SideView gView() const noexcept { return { &g_, nvars() }; }

    etstore_t< E > inner_;
    mutable Coeffs coeffs_;
    mutable Coeffs g_;
    mutable std::size_t next_{ 0 };
};

template < class E >
using AtanExpr = InverseFunctionExpr< E, 0, 0, +1 >;
template < class E >
using AtanhExpr = InverseFunctionExpr< E, 1, 1, +1 >;
template < class E >
using AsinExpr = InverseFunctionExpr< E, 2, 2, +1 >;
template < class E >
using AcosExpr = InverseFunctionExpr< E, 3, 2, -1 >;
template < class E >
using AsinhExpr = InverseFunctionExpr< E, 4, 3, +1 >;
template < class E >
using AcoshExpr = InverseFunctionExpr< E, 5, 4, +1 >;

// ----------------------------------------------------------------------
// Atan2Expr — atan2(y, x) with G = x^2 + y^2 maintained in an aux buffer.
template < class Y, class X >
    requires SameKindExpression< Y, X >
class Atan2Expr : public Expr< Atan2Expr< Y, X > >
{
  public:
    using Scalar = typename Y::Scalar;
    static constexpr bool IsStatic = Y::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = Y::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = Y::VarsAtCompileTime;
    using Coeffs = detail::coeffs_for_t< Atan2Expr >;

    Atan2Expr( const Y& y, const X& x )
        : y_( y ),
          x_( x ),
          coeffs_( detail::makeCoeffsLike( y ) ),
          g_( detail::makeCoeffsLike( y ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept { return y_.order(); }
    [[nodiscard]] std::size_t nvars() const noexcept { return y_.nvars(); }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ ) return;
        y_.advanceTo( d );
        x_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            // G_e = (x^2 + y^2)_e.
            auto g_e = g_.segment(
                static_cast< Eigen::Index >( util::degreeOffset( e, nvars() ) ),
                static_cast< Eigen::Index >( util::degreeSize( e, nvars() ) ) );
            const std::size_t es = util::degreeSize( e, nvars() );
            for ( std::size_t i = 0; i < es; ++i )
            {
                g_e.coeffRef( static_cast< Eigen::Index >( i ) ) = Scalar{ 0 };
            }
            for ( std::size_t k = 0; k <= e; ++k )
            {
                auto x_k = x_.slice( k );
                auto x_ek = x_.slice( e - k );
                auto y_k = y_.slice( k );
                auto y_ek = y_.slice( e - k );
                kernels::cauchyAccumulateSlice< Scalar >( k, e - k, nvars(),
                                                          Scalar{ 1 }, x_k, x_ek, g_e );
                kernels::cauchyAccumulateSlice< Scalar >( k, e - k, nvars(),
                                                          Scalar{ 1 }, y_k, y_ek, g_e );
            }
            // F_e.
            if ( e == 0 )
            {
                slice( 0 ).coeffRef( 0 ) = std::atan2(
                    static_cast< Scalar >( y_.slice( 0 ).coeff( 0 ) ),
                    static_cast< Scalar >( x_.slice( 0 ).coeff( 0 ) ) );
            }
            else
            {
                kernels::atan2ComputeDegree< Scalar >( e, nvars(), x_, y_, gView(),
                                                      *this );
            }
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    struct SideView
    {
        Coeffs* buf;
        std::size_t nvars_v;
        [[nodiscard]] auto slice( std::size_t d ) const noexcept
        {
            return buf->segment(
                static_cast< Eigen::Index >( util::degreeOffset( d, nvars_v ) ),
                static_cast< Eigen::Index >( util::degreeSize( d, nvars_v ) ) );
        }
    };
    [[nodiscard]] SideView gView() const noexcept { return { &g_, nvars() }; }

    etstore_t< Y > y_;
    etstore_t< X > x_;
    mutable Coeffs coeffs_;
    mutable Coeffs g_;
    mutable std::size_t next_{ 0 };
};

// ----------------------------------------------------------------------
// ErfExpr — error function.  Uses E[F] = (2/sqrt(pi)) H E[u], where H
// = exp(-u^2) is maintained in an auxiliary buffer.  H itself follows
// the exp recurrence applied to v = -u^2, computed slice-by-slice on
// the fly.
template < class E >
    requires TaxExpression< E >
class ErfExpr : public Expr< ErfExpr< E > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;
    using Coeffs = detail::coeffs_for_t< ErfExpr >;

    explicit ErfExpr( const E& e )
        : inner_( e ),
          coeffs_( detail::makeCoeffsLike( e ) ),
          h_( detail::makeCoeffsLike( e ) )
    {
    }

    [[nodiscard]] std::size_t order() const noexcept { return inner_.order(); }
    [[nodiscard]] std::size_t nvars() const noexcept { return inner_.nvars(); }

    void advanceTo( std::size_t d ) const
    {
        if ( d < next_ ) return;
        inner_.advanceTo( d );
        for ( std::size_t e = next_; e <= d; ++e )
        {
            // 1) Fill H_e where H = exp(-u^2).
            //    H_0 = exp(-u_0^2); for e >= 1, use exp recurrence:
            //      H_e = (1/e) * sum_{k=1..e} k * v_k * H_{e-k}
            //    where v_k = -(u^2)_k.  We accumulate v_k * H slices via
            //    cauchyAccumulateSlice on (cauchy(u, u) negated) and H.
            auto h_e = h_.segment(
                static_cast< Eigen::Index >( util::degreeOffset( e, nvars() ) ),
                static_cast< Eigen::Index >( util::degreeSize( e, nvars() ) ) );
            const std::size_t es = util::degreeSize( e, nvars() );
            if ( e == 0 )
            {
                const Scalar u0 = static_cast< Scalar >( inner_.slice( 0 ).coeff( 0 ) );
                h_e.coeffRef( 0 ) = std::exp( -u0 * u0 );
            }
            else
            {
                for ( std::size_t i = 0; i < es; ++i )
                {
                    h_e.coeffRef( static_cast< Eigen::Index >( i ) ) = Scalar{ 0 };
                }
                const Scalar inv_e = Scalar{ 1 } / static_cast< Scalar >( e );
                // sum_{k=1..e} k * v_k * H_{e-k}, where v_k = -(u*u)_k.
                // Compute (u*u)_k temporarily and treat sign at the inner kernel call.
                // Simpler: split each k contribution into pairwise cauchy products
                // (-u_a u_b * H_{e-k}) for a + b = k, and absorb the k*inv_e factor.
                for ( std::size_t k = 1; k <= e; ++k )
                {
                    const Scalar weight = -static_cast< Scalar >( k ) * inv_e;
                    // Compute the (u^2)_k slice on the fly, accumulating
                    // weight * (u^2)_k * H_{e-k} into h_e.  We need a
                    // temporary (u^2)_k buffer, but we can avoid it by
                    // expanding the triple sum:
                    //   (u^2)_k = sum_{a+b=k} u_a u_b.
                    // Then weight * (u^2)_k * H_{e-k}_alpha
                    //     = weight * sum_{a+b=k, c+d=e-k}
                    //         u_a · u_b · H_c   (combined into alpha = a+b+c)
                    // i.e. we accumulate into h_e at multi-index a+b+c.
                    // Rather than a triple loop, materialise (u^2)_k into
                    // a temporary slice in the aux buffer's "next" slot.
                    Coeffs tmp_uu_k = Coeffs::Zero(
                        static_cast< Eigen::Index >( util::degreeSize( k, nvars() ) ) );
                    for ( std::size_t a = 0; a <= k; ++a )
                    {
                        auto u_a = inner_.slice( a );
                        auto u_b = inner_.slice( k - a );
                        kernels::cauchyAccumulateSlice< Scalar >( a, k - a, nvars(),
                                                                  Scalar{ 1 }, u_a, u_b,
                                                                  tmp_uu_k );
                    }
                    auto h_ek = h_.segment(
                        static_cast< Eigen::Index >( util::degreeOffset( e - k, nvars() ) ),
                        static_cast< Eigen::Index >( util::degreeSize( e - k, nvars() ) ) );
                    kernels::cauchyAccumulateSlice< Scalar >( k, e - k, nvars(), weight,
                                                              tmp_uu_k, h_ek, h_e );
                }
            }

            // 2) Fill F_e using d * F_e = (2/sqrt(pi)) sum_{a+b=alpha, |b|>=1}
            //    H_a * |b| * u_b.
            auto f_e = slice( e );
            if ( e == 0 )
            {
                const Scalar u0 = static_cast< Scalar >( inner_.slice( 0 ).coeff( 0 ) );
                f_e.coeffRef( 0 ) = std::erf( u0 );
            }
            else
            {
                for ( std::size_t i = 0; i < es; ++i )
                {
                    f_e.coeffRef( static_cast< Eigen::Index >( i ) ) = Scalar{ 0 };
                }
                static const Scalar two_over_sqrt_pi =
                    static_cast< Scalar >( 1.1283791670955125738961589031215452 );
                const Scalar inv_e = Scalar{ 1 } / static_cast< Scalar >( e );
                for ( std::size_t k = 1; k <= e; ++k )
                {
                    // |gamma| = k, |beta| = e - k.
                    const Scalar weight = two_over_sqrt_pi * static_cast< Scalar >( k ) * inv_e;
                    auto h_ek = h_.segment(
                        static_cast< Eigen::Index >( util::degreeOffset( e - k, nvars() ) ),
                        static_cast< Eigen::Index >( util::degreeSize( e - k, nvars() ) ) );
                    auto u_k = inner_.slice( k );
                    kernels::cauchyAccumulateSlice< Scalar >( e - k, k, nvars(), weight,
                                                              h_ek, u_k, f_e );
                }
            }
        }
        next_ = d + 1;
    }

    [[nodiscard]] auto slice( std::size_t d ) const noexcept
    {
        return coeffs_.segment(
            static_cast< Eigen::Index >( util::degreeOffset( d, nvars() ) ),
            static_cast< Eigen::Index >( util::degreeSize( d, nvars() ) ) );
    }

  private:
    etstore_t< E > inner_;
    mutable Coeffs coeffs_;
    mutable Coeffs h_;
    mutable std::size_t next_{ 0 };
};

}  // namespace tax::expr
