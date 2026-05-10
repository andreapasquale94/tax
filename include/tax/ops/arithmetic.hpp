// SPDX-License-Identifier: BSD-3-Clause
//
// Arithmetic operators: +, -, *, /, unary -.
//
// Each operator picks the appropriate ET node and propagates IsStatic /
// dimensional checks through the SameKindExpression concept.  Scalar
// operands fold into the cheap view-like ScalarAddExpr / ScalarMulExpr.

#pragma once

#include <type_traits>

#include "tax/expr/buffered_nodes.hpp"
#include "tax/expr/view_nodes.hpp"

namespace tax
{

// ---- + ---------------------------------------------------------------
template < class L, class R >
    requires SameKindExpression< L, R >
[[nodiscard]] auto operator+( const L& l, const R& r ) noexcept
{
    return expr::AddExpr< L, R >( l, r );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto operator+( const E& e, typename std::remove_cvref_t< E >::Scalar c ) noexcept
{
    return expr::ScalarAddExpr< E >( e, c );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto operator+( typename std::remove_cvref_t< E >::Scalar c, const E& e ) noexcept
{
    return expr::ScalarAddExpr< E >( e, c );
}

// ---- - ---------------------------------------------------------------
template < class L, class R >
    requires SameKindExpression< L, R >
[[nodiscard]] auto operator-( const L& l, const R& r ) noexcept
{
    return expr::SubExpr< L, R >( l, r );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto operator-( const E& e ) noexcept
{
    return expr::NegExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto operator-( const E& e, typename std::remove_cvref_t< E >::Scalar c ) noexcept
{
    return expr::ScalarAddExpr< E >( e, -c );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto operator-( typename std::remove_cvref_t< E >::Scalar c, const E& e ) noexcept
{
    // c - e = (-e) + c.  NegExpr<E> is constructed as a prvalue and copied
    // into ScalarAddExpr's by-value child slot via etstore_t.
    return expr::ScalarAddExpr< expr::NegExpr< E > >( expr::NegExpr< E >( e ), c );
}

// ---- * ---------------------------------------------------------------
template < class L, class R >
    requires SameKindExpression< L, R >
[[nodiscard]] auto operator*( const L& l, const R& r ) noexcept
{
    return expr::MulExpr< L, R >( l, r );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto operator*( const E& e, typename std::remove_cvref_t< E >::Scalar c ) noexcept
{
    return expr::ScalarMulExpr< E >( e, c );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto operator*( typename std::remove_cvref_t< E >::Scalar c, const E& e ) noexcept
{
    return expr::ScalarMulExpr< E >( e, c );
}

// ---- / ---------------------------------------------------------------
template < class L, class R >
    requires SameKindExpression< L, R >
[[nodiscard]] auto operator/( const L& l, const R& r ) noexcept
{
    return expr::DivExpr< L, R >( l, r );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto operator/( const E& e, typename std::remove_cvref_t< E >::Scalar c ) noexcept
{
    using S = typename std::remove_cvref_t< E >::Scalar;
    return expr::ScalarMulExpr< E >( e, S{ 1 } / c );
}

}  // namespace tax
