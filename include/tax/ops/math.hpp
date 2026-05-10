// SPDX-License-Identifier: BSD-3-Clause
//
// Math free functions.
//
// All return ET nodes from `expr::`.  Buffered nodes own a coefficient
// buffer; view-like compositions (cube via x * x * x, hypot via sqrt
// of sums of squares, etc.) propagate through the multiplicative ET
// path.

#pragma once

#include <cmath>
#include <type_traits>

#include "tax/expr/buffered_nodes.hpp"
#include "tax/expr/view_nodes.hpp"
#include "tax/ops/arithmetic.hpp"

namespace tax
{

// ---- single-branch trig / hyper -------------------------------------

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

// ---- paired trig / hyper --------------------------------------------
//
// `sincos(x)` and `sinhcosh(x)` return owner objects that hold a single
// shared buffered node.  The `.sin() / .cos()` (resp. `.sinh() /
// .cosh()`) accessors hand back lightweight view ETs that read the
// shared buffers.  Realising one side of the pair is enough to fill
// both — the second `<<=` is a buffer copy.

template < class E >
class SinCosPair
{
  public:
    using Node = expr::SinCosNodeExpr< E >;

    explicit SinCosPair( const E& e ) : node_( e )
    {
    }

    [[nodiscard]] auto sin() const noexcept
    {
        return expr::SinCosPairView< Node, true >( node_ );
    }
    [[nodiscard]] auto cos() const noexcept
    {
        return expr::SinCosPairView< Node, false >( node_ );
    }

  private:
    Node node_;
};

template < class E >
class SinhCoshPair
{
  public:
    using Node = expr::SinhCoshNodeExpr< E >;

    explicit SinhCoshPair( const E& e ) : node_( e )
    {
    }

    [[nodiscard]] auto sinh() const noexcept
    {
        return expr::SinhCoshPairView< Node, true >( node_ );
    }
    [[nodiscard]] auto cosh() const noexcept
    {
        return expr::SinhCoshPairView< Node, false >( node_ );
    }

  private:
    Node node_;
};

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto sincos( const E& e )
{
    return SinCosPair< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto sinhcosh( const E& e )
{
    return SinhCoshPair< E >( e );
}

// ---- inverse trig / hyperbolic --------------------------------------

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto atan( const E& e )
{
    return expr::AtanExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto atanh( const E& e )
{
    return expr::AtanhExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto asin( const E& e )
{
    return expr::AsinExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto acos( const E& e )
{
    return expr::AcosExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto asinh( const E& e )
{
    return expr::AsinhExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto acosh( const E& e )
{
    return expr::AcoshExpr< E >( e );
}

// ---- exp / log family -----------------------------------------------

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

// log10(x) = log(x) / log(10).  View-like scaled composition.
template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto log10( const E& e )
{
    using S = typename std::remove_cvref_t< E >::Scalar;
    constexpr S inv_log10 = static_cast< S >( 0.43429448190325182765112891891660508 );
    return expr::ScalarMulExpr< expr::LogExpr< E > >( expr::LogExpr< E >( e ), inv_log10 );
}

// ---- roots and powers ------------------------------------------------

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto sqrt( const E& e )
{
    return expr::SqrtExpr< E >( e );
}

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto cbrt( const E& e )
{
    return expr::CbrtExpr< E >( e );
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
    return expr::MulExpr< expr::SquareExpr< E >, E >( expr::SquareExpr< E >( e ), e );
}

// pow with compile-time integer exponent.  Resolved by template recursion
// into chains of MulExpr / SquareExpr / DivExpr — the result type
// changes with N but the runtime path is exactly what the user would
// have written by hand.
template < int N, class E >
    requires TaxExpression< E >
[[nodiscard]] auto pow( const E& e )
{
    if constexpr ( N == 0 )
    {
        // x^0 = 1.  Use a constant scaled view (e * 0 + 1) so the
        // resulting type still depends on E (no ambiguous overload).
        return expr::ScalarAddExpr< expr::ScalarMulExpr< E > >(
            expr::ScalarMulExpr< E >( e, typename std::remove_cvref_t< E >::Scalar{ 0 } ),
            typename std::remove_cvref_t< E >::Scalar{ 1 } );
    }
    else if constexpr ( N == 1 )
    {
        return e;
    }
    else if constexpr ( N == 2 )
    {
        return expr::SquareExpr< E >( e );
    }
    else if constexpr ( N < 0 )
    {
        // x^(-N) = 1 / x^N.
        using S = typename std::remove_cvref_t< E >::Scalar;
        auto pos = pow< -N >( e );
        return expr::DivExpr< expr::ScalarAddExpr< expr::ScalarMulExpr< E > >,
                              decltype( pos ) >(
            expr::ScalarAddExpr< expr::ScalarMulExpr< E > >(
                expr::ScalarMulExpr< E >( e, S{ 0 } ), S{ 1 } ),
            std::move( pos ) );
    }
    else if constexpr ( N % 2 == 0 )
    {
        // Repeated squaring: x^(2k) = (x^k)^2.  The inner pow<N/2>(e)
        // yields some buffered ET; SquareExpr wraps it.
        auto half = pow< N / 2 >( e );
        return expr::SquareExpr< decltype( half ) >( std::move( half ) );
    }
    else
    {
        // Odd N: x^N = x * x^(N-1).
        auto rest = pow< N - 1 >( e );
        return expr::MulExpr< E, decltype( rest ) >( e, std::move( rest ) );
    }
}

// pow with a runtime real exponent.
template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto pow( const E& e, typename std::remove_cvref_t< E >::Scalar p )
{
    return expr::PowRealExpr< E >( e, p );
}

// ---- atan2, hypot ----------------------------------------------------

template < class Y, class X >
    requires SameKindExpression< Y, X >
[[nodiscard]] auto atan2( const Y& y, const X& x )
{
    return expr::Atan2Expr< Y, X >( y, x );
}

// hypot(x, y) = sqrt(x^2 + y^2).  Composed entirely from existing
// nodes, no new kernel needed.
template < class X, class Y >
    requires SameKindExpression< X, Y >
[[nodiscard]] auto hypot( const X& x, const Y& y )
{
    auto sum = expr::AddExpr< expr::SquareExpr< X >, expr::SquareExpr< Y > >(
        expr::SquareExpr< X >( x ), expr::SquareExpr< Y >( y ) );
    return expr::SqrtExpr< decltype( sum ) >( sum );
}

// 3-argument hypot(x, y, z) = sqrt(x^2 + y^2 + z^2).
template < class X, class Y, class Z >
    requires( SameKindExpression< X, Y > && SameKindExpression< Y, Z > )
[[nodiscard]] auto hypot( const X& x, const Y& y, const Z& z )
{
    auto xy = expr::AddExpr< expr::SquareExpr< X >, expr::SquareExpr< Y > >(
        expr::SquareExpr< X >( x ), expr::SquareExpr< Y >( y ) );
    auto xyz =
        expr::AddExpr< decltype( xy ), expr::SquareExpr< Z > >( xy, expr::SquareExpr< Z >( z ) );
    return expr::SqrtExpr< decltype( xyz ) >( xyz );
}

// ---- erf -------------------------------------------------------------

template < class E >
    requires TaxExpression< E >
[[nodiscard]] auto erf( const E& e )
{
    return expr::ErfExpr< E >( e );
}

}  // namespace tax
