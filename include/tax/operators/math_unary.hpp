#pragma once

#include <tax/expr/func_expr.hpp>
#include <tax/operators/common.hpp>

namespace tax
{

template < typename E >
[[nodiscard]] constexpr auto abs( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAbs< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto square( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpSquare< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto cube( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpCube< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto sqrt( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpSqrt< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto sin( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpSin< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto cos( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpCos< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto tan( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpTan< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto asin( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAsin< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto acos( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAcos< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto atan( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAtan< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto sinh( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpSinh< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto cosh( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpCosh< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto tanh( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpTanh< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto log( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpLog< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto log10( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpLog10< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto exp( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpExp< E::order, E::nvars > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto erf( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpErf< E::order, E::nvars > >{ e.self() };
}

/**
 * @brief Compositional inverse (series reversion).
 * @details Given `f` with `f(0) == 0` and `f'(0) != 0`, returns `g` such that `f(g(y)) = y`.
 *          Univariate only (`M == 1`).
 */
template < typename E >
[[nodiscard]] constexpr auto inv( const ExprBase< E >& e ) noexcept
{
    static_assert( E::nvars == 1, "inv requires univariate DA (M==1)" );
    return detail::FuncExpr< E, detail::OpInv< E::order, E::nvars > >{ e.self() };
}

}  // namespace tax
