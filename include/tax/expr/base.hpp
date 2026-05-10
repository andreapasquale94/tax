// SPDX-License-Identifier: BSD-3-Clause
//
// Streaming expression infrastructure shared by every ET node.
//
// Every node — view-like or buffered — exposes a uniform interface:
//   using Scalar           = ...
//   static constexpr bool IsStatic        // true iff backed by static-size storage
//   static constexpr int OrderAtCompileTime          // template arg, possibly Eigen::Dynamic
//   static constexpr int VarsAtCompileTime           // template arg, possibly Eigen::Dynamic
//   std::size_t order() const noexcept
//   std::size_t nvars() const noexcept
//   void advanceTo(std::size_t d) const     // const for both view-like and
//                                           // buffered nodes (latter use
//                                           // mutable internal state)
//   auto slice(std::size_t d) const   // const view (lazy or dense)
//
// Buffered nodes additionally expose a non-const overload of slice
// returning a writable Eigen::VectorBlock — that is what the kernels write
// into when the buffered node fills its own d-slice.
//
// The CRTP base does no real work; it only marks the type so that operators
// can recognise tax expressions without intersecting the entire C++ overload
// set.

#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <type_traits>

#include "tax/concepts.hpp"
#include "tax/util/multi_index.hpp"

namespace tax::expr
{

// Tag for "this type is a tax expression".  Operators key off this.
struct ExprTag
{
};

// Storage policy for an ET operand: ET nodes (those inheriting from ExprTag)
// are stored by value so that nested temporaries — e.g. `NegExpr<R>` inside
// `ScalarAddExpr<NegExpr<R>>` — survive past their constructor's scope.
// Concrete TTE storage types live as user-owned objects with stable address
// and are stored by const&.
template < class E >
using etstore_t = std::conditional_t< std::is_base_of_v< ExprTag, E >, E, const E& >;

template < class Derived >
class Expr : public ExprTag
{
  public:
    [[nodiscard]] Derived& derived() noexcept
    {
        return *static_cast< Derived* >( this );
    }
    [[nodiscard]] const Derived& derived() const noexcept
    {
        return *static_cast< const Derived* >( this );
    }
};

// ----------------------------------------------------------------------
// Coefficient buffer type for a streaming expression that itself owns
// storage (buffered nodes).  Resolves to a static-extent Eigen::Matrix
// when IsStatic == true and to Eigen::VectorX when IsStatic == false.

namespace detail
{

template < class E, bool S = E::IsStatic >
struct coeffs_for;

template < class E >
struct coeffs_for< E, true >
{
    using type = Eigen::Matrix< typename E::Scalar,
                                static_cast< Eigen::Index >( util::monomialCount(
                                    static_cast< std::size_t >( E::OrderAtCompileTime ),
                                    static_cast< std::size_t >( E::VarsAtCompileTime ) ) ),
                                1 >;
};

template < class E >
struct coeffs_for< E, false >
{
    using type = Eigen::Matrix< typename E::Scalar, Eigen::Dynamic, 1 >;
};

template < class E >
using coeffs_for_t = typename coeffs_for< E >::type;

// Allocate a coefficient buffer matching `op` in size and zero-initialise.
template < class E >
[[nodiscard]] coeffs_for_t< E > makeCoeffsLike( const E& op )
{
    using Coeffs = coeffs_for_t< E >;
    if constexpr ( E::IsStatic )
    {
        return Coeffs::Zero();
    }
    else
    {
        return Coeffs::Zero(
            static_cast< Eigen::Index >( util::monomialCount( op.order(), op.nvars() ) ) );
    }
}

}  // namespace detail

// Concept identifying any tax expression operand (storage type or ET node).
template < class E >
concept TaxExpression = std::is_base_of_v< ExprTag, std::remove_cvref_t< E > >
                       || requires { typename std::remove_cvref_t< E >::Scalar; }
                              && requires {
                                     {
                                         std::remove_cvref_t< E >::IsStatic
                                     } -> std::convertible_to< bool >;
                                 };

// Concept enforcing two operands match in static-ness (and dimensions when
// static).  Operators use this to reject mixed static/dynamic expressions
// at compile time.
template < class L, class R >
concept SameKindExpression =
    TaxExpression< L > && TaxExpression< R >
    && std::remove_cvref_t< L >::IsStatic == std::remove_cvref_t< R >::IsStatic
    && std::is_same_v< typename std::remove_cvref_t< L >::Scalar,
                       typename std::remove_cvref_t< R >::Scalar >;

}  // namespace tax::expr

namespace tax
{

// Re-export at top-level so users can write `requires tax::TaxExpression<E>`.
using expr::SameKindExpression;
using expr::TaxExpression;

}  // namespace tax
