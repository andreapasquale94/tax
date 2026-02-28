#pragma once

#include <tax/da.hpp>
#include <tax/expr/bin_expr.hpp>
#include <tax/expr/unary_expr.hpp>
#include <tax/expr/scalar_expr.hpp>
#include <tax/expr/func_expr.hpp>
#include <tax/expr/sum_expr.hpp>
#include <tax/expr/product_expr.hpp>

namespace tax {
// All overloads return lazy expression nodes; evaluation happens on materialization.

/**
 * @brief Constraint requiring same order, variable count, and scalar type.
 * @details Disallows mixing DA objects with incompatible truncation spaces.
 */
template <typename L, typename R>
concept CompatibleDA =
    (L::order == R::order) && (L::nvars == R::nvars) &&
    std::is_same_v<typename L::scalar_type, typename R::scalar_type>;

#define DA_BASE(E) DAExpr<E, typename E::scalar_type, E::order, E::nvars>

// -- DA + DA (four overloads for SumExpr flattening) --------------------------

template <typename L, typename R> requires CompatibleDA<L, R>
/// @brief Build lazy sum expression `l + r`.
[[nodiscard]] constexpr auto operator+(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::SumExpr<L, R>{l.self(), r.self()}; }

template <typename... Ls, typename R>
requires CompatibleDA<detail::SumExpr<Ls...>, R>
/// @brief Append operand to an existing flattened sum.
[[nodiscard]] constexpr auto operator+(const detail::SumExpr<Ls...>& l,
                                       const DA_BASE(R)& r) noexcept
{ return l.template append<R>(r.self()); }

template <typename L, typename... Rs>
requires CompatibleDA<L, detail::SumExpr<Rs...>>
/// @brief Prepend operand to an existing flattened sum.
[[nodiscard]] constexpr auto operator+(const DA_BASE(L)& l,
                                       const detail::SumExpr<Rs...>& r) noexcept
{ return r.template prepend<L>(l.self()); }

template <typename... Ls, typename... Rs>
requires CompatibleDA<detail::SumExpr<Ls...>, detail::SumExpr<Rs...>>
/// @brief Concatenate two flattened sums.
[[nodiscard]] constexpr auto operator+(const detail::SumExpr<Ls...>& l,
                                       const detail::SumExpr<Rs...>& r) noexcept
{ return l.concat(r); }

// -- DA - DA ------------------------------------------------------------------

template <typename L, typename R> requires CompatibleDA<L, R>
/// @brief Build lazy subtraction expression `l - r`.
[[nodiscard]] constexpr auto operator-(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::BinExpr<L, R, detail::OpSub>{l.self(), r.self()}; }

// -- DA * DA (two overloads for ProductExpr flattening) -----------------------

template <typename L, typename R> requires CompatibleDA<L, R>
/// @brief Build lazy product expression `l * r`.
[[nodiscard]] constexpr auto operator*(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::ProductExpr<L, R>{l.self(), r.self()}; }

template <typename... Ls, typename R>
requires CompatibleDA<detail::ProductExpr<Ls...>, R>
/// @brief Append operand to an existing flattened product.
[[nodiscard]] constexpr auto operator*(const detail::ProductExpr<Ls...>& l,
                                       const DA_BASE(R)& r) noexcept
{ return l.template append<R>(r.self()); }

// -- DA / DA ------------------------------------------------------------------

template <typename L, typename R> requires CompatibleDA<L, R>
/// @brief Build lazy division expression `l / r`.
[[nodiscard]] constexpr auto operator/(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::BinExpr<L, R, detail::OpDiv<L::order, L::nvars>>{l.self(), r.self()}; }

// -- Unary negation -----------------------------------------------------------

template <typename E>
/// @brief Build lazy unary negation expression `-e`.
[[nodiscard]] constexpr auto operator-(const DA_BASE(E)& e) noexcept
{ return detail::UnaryExpr<E, detail::OpNeg>{e.self()}; }

// -- DA op scalar -------------------------------------------------------------

template <typename E>
/// @brief Build expression `e + s`.
[[nodiscard]] constexpr auto operator+(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarAdd>{e.self(), s}; }
template <typename E>
/// @brief Build expression `e - s`.
[[nodiscard]] constexpr auto operator-(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarSubR>{e.self(), s}; }
template <typename E>
/// @brief Build expression `e * s`.
[[nodiscard]] constexpr auto operator*(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarMul>{e.self(), s}; }
template <typename E>
/// @brief Build expression `e / s`.
[[nodiscard]] constexpr auto operator/(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarDivR>{e.self(), s}; }

// -- scalar op DA -------------------------------------------------------------

template <typename E>
/// @brief Build expression `s + e`.
[[nodiscard]] constexpr auto operator+(typename E::scalar_type s, const DA_BASE(E)& e) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarAdd>{e.self(), s}; }
template <typename E>
/// @brief Build expression `s - e`.
[[nodiscard]] constexpr auto operator-(typename E::scalar_type s, const DA_BASE(E)& e) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarSubL>{e.self(), s}; }
template <typename E>
/// @brief Build expression `s * e`.
[[nodiscard]] constexpr auto operator*(typename E::scalar_type s, const DA_BASE(E)& e) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarMul>{e.self(), s}; }
template <typename E>
/// @brief Build expression `s / e` using reciprocal series.
[[nodiscard]] constexpr auto operator/(typename E::scalar_type s, const DA_BASE(E)& e) noexcept {
    using Recip = detail::FuncExpr<E, detail::OpReciprocal<E::order, E::nvars>>;
    return detail::ScalarExpr<Recip, detail::OpScalarMul>{Recip{e.self()}, s};
}

// -- Math free functions ------------------------------------------------------

template <typename E>
/// @brief Build lazy `square(e)` expression.
[[nodiscard]] constexpr auto square(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpSquare<E::order, E::nvars>>{e.self()}; }

template <typename E>
/// @brief Build lazy `cube(e)` expression.
[[nodiscard]] constexpr auto cube(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpCube<E::order, E::nvars>>{e.self()}; }

template <typename E>
/// @brief Build lazy `sqrt(e)` expression.
[[nodiscard]] constexpr auto sqrt(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpSqrt<E::order, E::nvars>>{e.self()}; }

template <typename E>
/// @brief Build lazy `sin(e)` expression.
[[nodiscard]] constexpr auto sin(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpSin<E::order, E::nvars>>{e.self()}; }

template <typename E>
/// @brief Build lazy `cos(e)` expression.
[[nodiscard]] constexpr auto cos(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpCos<E::order, E::nvars>>{e.self()}; }

template <typename E>
/// @brief Build lazy `tan(e)` expression.
[[nodiscard]] constexpr auto tan(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpTan<E::order, E::nvars>>{e.self()}; }

template <typename E>
/// @brief Build lazy natural logarithm expression `log(e)`.
[[nodiscard]] constexpr auto log(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpLog<E::order, E::nvars>>{e.self()}; }

template <typename E>
/// @brief Build lazy base-10 logarithm expression `log10(e)`.
[[nodiscard]] constexpr auto log10(const DA_BASE(E)& e) noexcept
{ return detail::FuncExpr<E, detail::OpLog10<E::order, E::nvars>>{e.self()}; }

template <typename E>
/**
 * @brief Compute `sin(e)` and `cos(e)` together, returning `{sin, cos}`.
 * @details This evaluates the operand once and runs a coupled recurrence.
 */
[[nodiscard]] constexpr auto sincos(const DA_BASE(E)& e) noexcept {
    using T = typename E::scalar_type;
    constexpr int N = E::order, M = E::nvars;
    using DAType = TDA<T, N, M>;
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

} // namespace tax
