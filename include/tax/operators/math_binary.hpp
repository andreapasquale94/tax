#pragma once

#include <tax/expr/func_expr.hpp>
#include <tax/operators/common.hpp>
#include <tax/operators/math_unary.hpp>

namespace tax
{

template < typename L, typename R >
    requires CompatibleTTE< L, R >
[[nodiscard]] constexpr auto atan2( const ExprBase< L >& y, const ExprBase< R >& x ) noexcept
{
    return detail::BinFuncExpr< L, R, detail::OpAtan2< L::order, L::nvars > >{ y.self(), x.self() };
}

template < typename L, typename R >
    requires CompatibleTTE< L, R >
[[nodiscard]] constexpr auto hypot( const ExprBase< L >& x, const ExprBase< R >& y ) noexcept
{
    return detail::BinFuncExpr< L, R, detail::OpHypot< L::order, L::nvars > >{ x.self(), y.self() };
}

template < typename A, typename B, typename C >
    requires CompatibleTTE< A, B > && CompatibleTTE< B, C >
[[nodiscard]] constexpr auto hypot( const ExprBase< A >& x, const ExprBase< B >& y,
                                    const ExprBase< C >& z ) noexcept
{
    return detail::TerFuncExpr< A, B, C, detail::OpHypot3< A::order, A::nvars > >{
        x.self(), y.self(), z.self() };
}

template < typename E >
[[nodiscard]] constexpr auto ipow( const ExprBase< E >& e, int n ) noexcept
{
    return detail::ParamFuncExpr< E, detail::OpIPow< E::order, E::nvars >, int >{ e.self(), n };
}

template < typename E >
[[nodiscard]] constexpr auto dpow( const ExprBase< E >& e, typename E::scalar_type c ) noexcept
{
    return detail::ParamFuncExpr< E, detail::OpDPow< E::order, E::nvars >,
                                  typename E::scalar_type >{ e.self(), c };
}

template < typename L, typename R >
    requires CompatibleTTE< L, R >
[[nodiscard]] constexpr auto tpow( const ExprBase< L >& l, const ExprBase< R >& r ) noexcept
{
    return detail::BinFuncExpr< L, R, detail::OpTPow< L::order, L::nvars > >{ l.self(), r.self() };
}

// -- pow overloads dispatching to ipow / dpow / tpow -------------------------

template < typename E >
[[nodiscard]] constexpr auto pow( const ExprBase< E >& e, int n ) noexcept
{
    return ipow( e, n );
}

template < typename E >
[[nodiscard]] constexpr auto pow( const ExprBase< E >& e, typename E::scalar_type c ) noexcept
{
    return dpow( e, c );
}

template < typename L, typename R >
    requires CompatibleTTE< L, R >
[[nodiscard]] constexpr auto pow( const ExprBase< L >& l, const ExprBase< R >& r ) noexcept
{
    return tpow( l, r );
}

}  // namespace tax
