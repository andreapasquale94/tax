// SPDX-License-Identifier: BSD-3-Clause
//
// Math free functions: sin, cos, exp, log, sqrt, square, cube, ...
//
// All return ET nodes from `expr::`.  Buffered nodes (sin/cos/exp/log/sqrt)
// allocate; view-like compositions (cube via x * x * x) propagate through
// the multiplicative ET path.

#pragma once

#include "tax/expr/buffered_nodes.hpp"
#include "tax/expr/view_nodes.hpp"
#include "tax/ops/arithmetic.hpp"

namespace tax
{

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto sin( const E& e )
{
    return expr::SinExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto cos( const E& e )
{
    return expr::CosExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto sinh( const E& e )
{
    return expr::SinhExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto cosh( const E& e )
{
    return expr::CoshExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto exp( const E& e )
{
    return expr::ExpExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto log( const E& e )
{
    return expr::LogExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto sqrt( const E& e )
{
    return expr::SqrtExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto square( const E& e )
{
    return expr::SquareExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto cube( const E& e )
{
    // cube = square(e) * e via composition; both operands collapse into the
    // same buffered Mul node which fills slice-by-slice.
    return expr::MulExpr< expr::SquareExpr< E >, E >( expr::SquareExpr< E >( e ), e );
}

// tan = sin / cos.
template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto tan( const E& e )
{
    return expr::DivExpr< expr::SinExpr< E >, expr::CosExpr< E > >(
        expr::SinExpr< E >( e ), expr::CosExpr< E >( e ) );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto tanh( const E& e )
{
    return expr::DivExpr< expr::SinhExpr< E >, expr::CoshExpr< E > >(
        expr::SinhExpr< E >( e ), expr::CoshExpr< E >( e ) );
}

}  // namespace tax
