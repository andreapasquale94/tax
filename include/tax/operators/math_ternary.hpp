#pragma once

#include <tax/expr/func_expr.hpp>
#include <tax/operators/common.hpp>

namespace tax
{

/**
 * @brief Fused multiply-add: `fma(x, y, z) = x * y + z`.
 */
template < typename A, typename B, typename C >
    requires CompatibleTTE< A, B > && CompatibleTTE< B, C >
[[nodiscard]] constexpr auto fma( const ExprBase< A >& x, const ExprBase< B >& y,
                                  const ExprBase< C >& z ) noexcept
{
    return detail::TerFuncExpr< A, B, C, detail::OpFMA< A::order, A::nvars > >{ x.self(), y.self(),
                                                                                z.self() };
}

}  // namespace tax
