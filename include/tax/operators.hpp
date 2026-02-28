#pragma once

#include <tax/da.hpp>
#include <tax/expr/bin_expr.hpp>
#include <tax/expr/unary_expr.hpp>
#include <tax/expr/scalar_expr.hpp>
#include <tax/expr/func_expr.hpp>
#include <tax/expr/sum_expr.hpp>
#include <tax/expr/product_expr.hpp>

namespace da {

// =============================================================================
// Operator overloads and math free functions
// =============================================================================

template <typename L, typename R>
concept CompatibleDA =
    (L::order == R::order) && (L::nvars == R::nvars) &&
    std::is_same_v<typename L::scalar_type, typename R::scalar_type>;

#define DA_BASE(E) DAExpr<E, typename E::scalar_type, E::order, E::nvars>

// -- DA + DA (four overloads for SumExpr flattening) --------------------------

template <typename L, typename R> requires CompatibleDA<L, R>
[[nodiscard]] constexpr auto operator+(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::SumExpr<L, R>{l.self(), r.self()}; }

template <typename... Ls, typename R>
requires CompatibleDA<detail::SumExpr<Ls...>, R>
[[nodiscard]] constexpr auto operator+(const detail::SumExpr<Ls...>& l,
                                       const DA_BASE(R)& r) noexcept
{ return l.template append<R>(r.self()); }

template <typename L, typename... Rs>
requires CompatibleDA<L, detail::SumExpr<Rs...>>
[[nodiscard]] constexpr auto operator+(const DA_BASE(L)& l,
                                       const detail::SumExpr<Rs...>& r) noexcept
{ return r.template prepend<L>(l.self()); }

template <typename... Ls, typename... Rs>
requires CompatibleDA<detail::SumExpr<Ls...>, detail::SumExpr<Rs...>>
[[nodiscard]] constexpr auto operator+(const detail::SumExpr<Ls...>& l,
                                       const detail::SumExpr<Rs...>& r) noexcept
{ return l.concat(r); }

// -- DA - DA ------------------------------------------------------------------

template <typename L, typename R> requires CompatibleDA<L, R>
[[nodiscard]] constexpr auto operator-(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::BinExpr<L, R, detail::OpSub>{l.self(), r.self()}; }

// -- DA * DA (two overloads for ProductExpr flattening) -----------------------

template <typename L, typename R> requires CompatibleDA<L, R>
[[nodiscard]] constexpr auto operator*(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::ProductExpr<L, R>{l.self(), r.self()}; }

template <typename... Ls, typename R>
requires CompatibleDA<detail::ProductExpr<Ls...>, R>
[[nodiscard]] constexpr auto operator*(const detail::ProductExpr<Ls...>& l,
                                       const DA_BASE(R)& r) noexcept
{ return l.template append<R>(r.self()); }

// -- DA / DA ------------------------------------------------------------------

template <typename L, typename R> requires CompatibleDA<L, R>
[[nodiscard]] constexpr auto operator/(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::BinExpr<L, R, detail::OpDiv<L::order, L::nvars>>{l.self(), r.self()}; }

// -- Unary negation -----------------------------------------------------------

template <typename E>
[[nodiscard]] constexpr auto operator-(const DA_BASE(E)& e) noexcept
{ return detail::UnaryExpr<E, detail::OpNeg>{e.self()}; }

// -- DA op scalar -------------------------------------------------------------

template <typename E>
[[nodiscard]] constexpr auto operator+(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarAdd>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator-(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarSubR>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator*(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarMul>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator/(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarDivR>{e.self(), s}; }

// -- scalar op DA -------------------------------------------------------------

template <typename E>
[[nodiscard]] constexpr auto operator+(typename E::scalar_type s, const DA_BASE(E)& e) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarAdd>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator-(typename E::scalar_type s, const DA_BASE(E)& e) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarSubL>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator*(typename E::scalar_type s, const DA_BASE(E)& e) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarMul>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator/(typename E::scalar_type s, const DA_BASE(E)& e) noexcept {
    using Recip = detail::FuncExpr<E, detail::OpReciprocal<E::order, E::nvars>>;
    return detail::ScalarExpr<Recip, detail::OpScalarMul>{Recip{e.self()}, s};
}

// -- Math free functions ------------------------------------------------------

template <typename E>
[[nodiscard]] constexpr auto square(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpSquare<E::order, E::nvars>>{e.self()}; }

template <typename E>
[[nodiscard]] constexpr auto cube(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpCube<E::order, E::nvars>>{e.self()}; }

template <typename E>
[[nodiscard]] constexpr auto sqrt(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpSqrt<E::order, E::nvars>>{e.self()}; }

template <typename E>
[[nodiscard]] constexpr auto sin(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpSin<E::order, E::nvars>>{e.self()}; }

template <typename E>
[[nodiscard]] constexpr auto cos(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpCos<E::order, E::nvars>>{e.self()}; }

template <typename E>
[[nodiscard]] constexpr auto tan(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpTan<E::order, E::nvars>>{e.self()}; }

template <typename E>
[[nodiscard]] constexpr auto sincos(const DA_BASE(E)& e) noexcept {
    using T = typename E::scalar_type;
    constexpr int N = E::order, M = E::nvars;
    using DAType = DA<T, N, M>;
    using coeff_array = std::array<T, detail::numMonomials(N, M)>;

    coeff_array sa{}, ca{};
    if constexpr (detail::is_leaf_v<E>) {
        detail::seriesSinCos<T, N, M>(sa, ca, e.self().coeffs());
    } else {
        coeff_array a{};
        e.self().evalTo(a);
        detail::seriesSinCos<T, N, M>(sa, ca, a);
    }
    return std::pair<DAType, DAType>{DAType{sa}, DAType{ca}};
}

#undef DA_BASE

} // namespace da
