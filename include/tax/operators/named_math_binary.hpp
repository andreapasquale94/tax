#pragma once

// Binary math surface for NamedExpansion (any basis): pow, atan2. Forwarded to
// the inner expansion; atan2 merges the two operands' axis sets into their union.
// Mirrors operators/math_binary.hpp for the unnamed dense type and is basis-
// generic like operators/named_arithmetic.hpp. The inner call is unqualified
// after a `using tax::FN;` so ADL augments the overload set with the inner
// basis' own math at instantiation (see named_math_unary.hpp for the rationale).

#include <cmath>
#include <tax/core/named.hpp>
#include <tax/operators/math_binary.hpp>
#include <type_traits>

namespace tax::named
{

/// `x^n` for an integer exponent (axis set preserved).
template < typename T, typename Basis, int N, typename... A >
[[nodiscard]] NamedExpansion< T, Basis, N, A... > pow( const NamedExpansion< T, Basis, N, A... >& x,
                                                       int n ) noexcept
{
    using tax::pow;
    return NamedExpansion< T, Basis, N, A... >{ pow( x.inner(), n ) };
}

/// `x^p` for a real exponent (axis set preserved; requires x.value() != 0).
template < typename T, typename Basis, int N, typename... A >
[[nodiscard]] NamedExpansion< T, Basis, N, A... > pow( const NamedExpansion< T, Basis, N, A... >& x,
                                                       std::type_identity_t< T > p ) noexcept
{
    using tax::pow;
    return NamedExpansion< T, Basis, N, A... >{ pow( x.inner(), p ) };
}

/// `atan2(y, x)` over the union of the two operands' axis sets.
template < typename T, typename Basis, int N, typename... A, typename... B >
[[nodiscard]] auto atan2( const NamedExpansion< T, Basis, N, A... >& y,
                          const NamedExpansion< T, Basis, N, B... >& x ) noexcept
{
    using R = detail::MergedNamedExpansion< T, Basis, N, detail::TypeList< A... >,
                                            detail::TypeList< B... > >;
    using tax::atan2;
    return R{ atan2( y.template embed< R >().inner(), x.template embed< R >().inner() ) };
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
