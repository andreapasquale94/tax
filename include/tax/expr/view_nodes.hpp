// SPDX-License-Identifier: BSD-3-Clause
//
// View-like ET nodes.  None of these allocate.
//
// Each node returns a `ParentSliceView` from `slice(d)`.  The view holds
// a const-reference to the parent ET node (whose lifetime is the
// surrounding full expression) plus the degree being viewed, and
// computes each coefficient on demand by calling
// `parent.coeffAtSlice(d, i)`.
//
// Read-only by design: only `coeff(i)` and `size()` are exposed, which
// is the entire interface buffered kernels need.

#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <type_traits>

#include "tax/expr/base.hpp"

namespace tax::expr
{

// Generic read-only slice view: holds a stable reference to the parent ET
// and a degree, and forwards element accesses to `parent.coeffAtSlice(d, i)`.
template < class Parent >
class ParentSliceView
{
  public:
    using Scalar = typename Parent::Scalar;

    ParentSliceView( const Parent& p, std::size_t d ) noexcept
        : parent_( p ),
          d_( d ),
          size_( static_cast< Eigen::Index >( util::degreeSize( d, p.nvars() ) ) )
    {
    }

    [[nodiscard]] Scalar coeff( Eigen::Index i ) const
    {
        return parent_.coeffAtSlice( d_, static_cast< std::size_t >( i ) );
    }

    [[nodiscard]] Eigen::Index size() const noexcept
    {
        return size_;
    }

  private:
    const Parent& parent_;
    std::size_t d_;
    Eigen::Index size_;
};

// ----------------------------------------------------------------------
// AddExpr
template < class L, class R >
    requires SameKindExpression< L, R >
class AddExpr : public Expr< AddExpr< L, R > >
{
  public:
    using Scalar = typename L::Scalar;
    static constexpr bool IsStatic = L::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = L::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = L::VarsAtCompileTime;

    AddExpr( const L& l, const R& r ) noexcept : lhs_( l ), rhs_( r )
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
        lhs_.advanceTo( d );
        rhs_.advanceTo( d );
    }

    [[nodiscard]] Scalar coeffAtSlice( std::size_t d, std::size_t i ) const
    {
        return static_cast< Scalar >( lhs_.slice( d ).coeff( static_cast< Eigen::Index >( i ) ) )
               + static_cast< Scalar >( rhs_.slice( d ).coeff( static_cast< Eigen::Index >( i ) ) );
    }

    [[nodiscard]] ParentSliceView< AddExpr > slice( std::size_t d ) const noexcept
    {
        return ParentSliceView< AddExpr >( *this, d );
    }

  private:
    etstore_t< L > lhs_;
    etstore_t< R > rhs_;
};

// ----------------------------------------------------------------------
// SubExpr
template < class L, class R >
    requires SameKindExpression< L, R >
class SubExpr : public Expr< SubExpr< L, R > >
{
  public:
    using Scalar = typename L::Scalar;
    static constexpr bool IsStatic = L::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = L::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = L::VarsAtCompileTime;

    SubExpr( const L& l, const R& r ) noexcept : lhs_( l ), rhs_( r )
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
        lhs_.advanceTo( d );
        rhs_.advanceTo( d );
    }

    [[nodiscard]] Scalar coeffAtSlice( std::size_t d, std::size_t i ) const
    {
        return static_cast< Scalar >( lhs_.slice( d ).coeff( static_cast< Eigen::Index >( i ) ) )
               - static_cast< Scalar >( rhs_.slice( d ).coeff( static_cast< Eigen::Index >( i ) ) );
    }

    [[nodiscard]] ParentSliceView< SubExpr > slice( std::size_t d ) const noexcept
    {
        return ParentSliceView< SubExpr >( *this, d );
    }

  private:
    etstore_t< L > lhs_;
    etstore_t< R > rhs_;
};

// ----------------------------------------------------------------------
// NegExpr
template < class E >
    requires TaxExpression< E >
class NegExpr : public Expr< NegExpr< E > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;

    explicit NegExpr( const E& e ) noexcept : inner_( e )
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
        inner_.advanceTo( d );
    }

    [[nodiscard]] Scalar coeffAtSlice( std::size_t d, std::size_t i ) const
    {
        return -static_cast< Scalar >(
            inner_.slice( d ).coeff( static_cast< Eigen::Index >( i ) ) );
    }

    [[nodiscard]] ParentSliceView< NegExpr > slice( std::size_t d ) const noexcept
    {
        return ParentSliceView< NegExpr >( *this, d );
    }

  private:
    etstore_t< E > inner_;
};

// ----------------------------------------------------------------------
// ScalarMulExpr
template < class E >
    requires TaxExpression< E >
class ScalarMulExpr : public Expr< ScalarMulExpr< E > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;

    ScalarMulExpr( const E& e, Scalar s ) noexcept : inner_( e ), scale_( s )
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
        inner_.advanceTo( d );
    }

    [[nodiscard]] Scalar coeffAtSlice( std::size_t d, std::size_t i ) const
    {
        return scale_ * static_cast< Scalar >(
                            inner_.slice( d ).coeff( static_cast< Eigen::Index >( i ) ) );
    }

    [[nodiscard]] ParentSliceView< ScalarMulExpr > slice( std::size_t d ) const noexcept
    {
        return ParentSliceView< ScalarMulExpr >( *this, d );
    }

  private:
    etstore_t< E > inner_;
    Scalar scale_;
};

// ----------------------------------------------------------------------
// ScalarAddExpr: tte + scalar.  Only the (d=0, i=0) element gets shifted.
template < class E >
    requires TaxExpression< E >
class ScalarAddExpr : public Expr< ScalarAddExpr< E > >
{
  public:
    using Scalar = typename E::Scalar;
    static constexpr bool IsStatic = E::IsStatic;
    static constexpr bool IsDynamic = !IsStatic;
    static constexpr int OrderAtCompileTime = E::OrderAtCompileTime;
    static constexpr int VarsAtCompileTime = E::VarsAtCompileTime;

    ScalarAddExpr( const E& e, Scalar c ) noexcept : inner_( e ), offset_( c )
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
        inner_.advanceTo( d );
    }

    [[nodiscard]] Scalar coeffAtSlice( std::size_t d, std::size_t i ) const
    {
        const Scalar v = static_cast< Scalar >(
            inner_.slice( d ).coeff( static_cast< Eigen::Index >( i ) ) );
        return ( d == 0 && i == 0 ) ? v + offset_ : v;
    }

    [[nodiscard]] ParentSliceView< ScalarAddExpr > slice( std::size_t d ) const noexcept
    {
        return ParentSliceView< ScalarAddExpr >( *this, d );
    }

  private:
    etstore_t< E > inner_;
    Scalar offset_;
};

}  // namespace tax::expr
