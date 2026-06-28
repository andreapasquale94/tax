#pragma once

// Binary math surface for NamedTaylorExpansion (pow, atan2). Forwarded to the
// inner expansion; atan2 merges the two operands' axis sets into their union.
// Mirrors operators/math_binary.hpp for the unnamed dense type.

#include <cmath>
#include <tax/core/named.hpp>
#include <tax/operators/math_binary.hpp>
#include <type_traits>

namespace tax::named
{

/// `x^n` for an integer exponent (axis set preserved).
template < typename T, int N, typename... A >
[[nodiscard]] NamedTaylorExpansion< T, N, A... > pow( const NamedTaylorExpansion< T, N, A... >& x,
                                                      int n ) noexcept
{
    return NamedTaylorExpansion< T, N, A... >{ tax::pow( x.inner(), n ) };
}

/// `x^p` for a real exponent (axis set preserved; requires x.value() != 0).
template < typename T, int N, typename... A >
[[nodiscard]] NamedTaylorExpansion< T, N, A... > pow( const NamedTaylorExpansion< T, N, A... >& x,
                                                      std::type_identity_t< T > p ) noexcept
{
    return NamedTaylorExpansion< T, N, A... >{ tax::pow( x.inner(), p ) };
}

/// `atan2(y, x)` over the union of the two operands' axis sets.
template < typename T, int N, typename... A, typename... B >
[[nodiscard]] auto atan2( const NamedTaylorExpansion< T, N, A... >& y,
                          const NamedTaylorExpansion< T, N, B... >& x ) noexcept
{
    using R = detail::MergedNamedExpansion< T, TaylorBasis, N, detail::TypeList< A... >,
                                            detail::TypeList< B... > >;
    return R{ tax::atan2( y.template embed< R >().inner(), x.template embed< R >().inner() ) };
}

}  // namespace tax::named

// ---------------------------------------------------------------------------
// Re-exports: qualified `tax::pow` / `tax::atan2` for named expansions and
// plain scalars (see named_math_unary.hpp for the rationale).
// ---------------------------------------------------------------------------

namespace tax
{
using named::atan2;
using named::pow;
using std::atan2;
using std::pow;
}  // namespace tax
