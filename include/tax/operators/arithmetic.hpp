#pragma once

#include <tax/expr/bin_expr.hpp>
#include <tax/expr/func_expr.hpp>
#include <tax/expr/product_expr.hpp>
#include <tax/expr/scalar_expr.hpp>
#include <tax/expr/sum_expr.hpp>
#include <tax/expr/unary_expr.hpp>
#include <tax/operators/common.hpp>

namespace tax
{

// All overloads return lazy expression nodes; evaluation happens on materialization.

// -- DA + DA (overloads for SumExpr flattening) ------------------------------

template < typename L, typename R >
    requires CompatibleDA< L, R >
[[nodiscard]] constexpr auto operator+( const ExprBase< L >& l, const ExprBase< R >& r ) noexcept
{
    return detail::SumExpr< L, R >{ l.self(), r.self() };
}

template < typename... Ls, typename R >
    requires CompatibleDA< detail::SumExpr< Ls... >, R >
[[nodiscard]] constexpr auto operator+( const detail::SumExpr< Ls... >& l,
                                        const ExprBase< R >& r ) noexcept
{
    return l.template append< R >( r.self() );
}

template < typename L, typename... Rs >
    requires CompatibleDA< L, detail::SumExpr< Rs... > >
[[nodiscard]] constexpr auto operator+( const ExprBase< L >& l,
                                        const detail::SumExpr< Rs... >& r ) noexcept
{
    return r.template prepend< L >( l.self() );
}

template < typename... Ls, typename... Rs >
    requires CompatibleDA< detail::SumExpr< Ls... >, detail::SumExpr< Rs... > >
[[nodiscard]] constexpr auto operator+( const detail::SumExpr< Ls... >& l,
                                        const detail::SumExpr< Rs... >& r ) noexcept
{
    return l.concat( r );
}

// -- DA - DA -----------------------------------------------------------------

template < typename L, typename R >
    requires CompatibleDA< L, R >
[[nodiscard]] constexpr auto operator-( const ExprBase< L >& l, const ExprBase< R >& r ) noexcept
{
    return detail::BinExpr< L, R, detail::OpSub >{ l.self(), r.self() };
}

// -- DA * DA (overloads for ProductExpr flattening) --------------------------

template < typename L, typename R >
    requires CompatibleDA< L, R >
[[nodiscard]] constexpr auto operator*( const ExprBase< L >& l, const ExprBase< R >& r ) noexcept
{
    return detail::ProductExpr< L, R >{ l.self(), r.self() };
}

template < typename... Ls, typename R >
    requires CompatibleDA< detail::ProductExpr< Ls... >, R >
[[nodiscard]] constexpr auto operator*( const detail::ProductExpr< Ls... >& l,
                                        const ExprBase< R >& r ) noexcept
{
    return l.template append< R >( r.self() );
}

// -- DA / DA -----------------------------------------------------------------

template < typename L, typename R >
    requires CompatibleDA< L, R >
[[nodiscard]] constexpr auto operator/( const ExprBase< L >& l, const ExprBase< R >& r ) noexcept
{
    return detail::BinExpr< L, R, detail::OpDiv< L::order, L::nvars > >{ l.self(), r.self() };
}

// -- Unary negation ----------------------------------------------------------

template < typename E >
[[nodiscard]] constexpr auto operator-( const ExprBase< E >& e ) noexcept
{
    return detail::UnaryExpr< E, detail::OpNeg >{ e.self() };
}

// -- DA op scalar -------------------------------------------------------------

template < typename E >
[[nodiscard]] constexpr auto operator+( const ExprBase< E >& e, typename E::scalar_type s ) noexcept
{
    return detail::ScalarExpr< E, detail::OpScalarAdd >{ e.self(), s };
}

template < typename E >
[[nodiscard]] constexpr auto operator-( const ExprBase< E >& e, typename E::scalar_type s ) noexcept
{
    return detail::ScalarExpr< E, detail::OpScalarSubR >{ e.self(), s };
}

template < typename E >
[[nodiscard]] constexpr auto operator*( const ExprBase< E >& e, typename E::scalar_type s ) noexcept
{
    return detail::ScalarExpr< E, detail::OpScalarMul >{ e.self(), s };
}

template < typename E >
[[nodiscard]] constexpr auto operator/( const ExprBase< E >& e, typename E::scalar_type s ) noexcept
{
    return detail::ScalarExpr< E, detail::OpScalarDivR >{ e.self(), s };
}

// -- scalar op DA -------------------------------------------------------------

template < typename E >
[[nodiscard]] constexpr auto operator+( typename E::scalar_type s, const ExprBase< E >& e ) noexcept
{
    return detail::ScalarExpr< E, detail::OpScalarAdd >{ e.self(), s };
}

template < typename E >
[[nodiscard]] constexpr auto operator-( typename E::scalar_type s, const ExprBase< E >& e ) noexcept
{
    return detail::ScalarExpr< E, detail::OpScalarSubL >{ e.self(), s };
}

template < typename E >
[[nodiscard]] constexpr auto operator*( typename E::scalar_type s, const ExprBase< E >& e ) noexcept
{
    return detail::ScalarExpr< E, detail::OpScalarMul >{ e.self(), s };
}

template < typename E >
[[nodiscard]] constexpr auto operator/( typename E::scalar_type s, const ExprBase< E >& e ) noexcept
{
    using Recip = detail::FuncExpr< E, detail::OpReciprocal< E::order, E::nvars > >;
    return detail::ScalarExpr< Recip, detail::OpScalarMul >{ Recip{ e.self() }, s };
}

}  // namespace tax
