#pragma once

// Binary math surface for MixedTaylorExpansion: pow, atan2. Mirrors
// operators/named_math_binary.hpp; the inner call is unqualified after
// `using tax::FN` so ADL reaches the dense math. Reached from user code via ADL
// only (e.g. `pow(mte, n)`); there is no `tax::pow` re-export for the mixed
// surface, matching the mixed unary-math convention.

#include <cmath>
#include <tax/expansion/mixed_named.hpp>
#include <tax/expansion/ops/math_binary.hpp>
#include <type_traits>

namespace tax::named
{

/// `x^n` for an integer exponent (axis set preserved).
template < typename T, typename... A >
[[nodiscard]] MixedTaylorExpansion< T, A... > pow( const MixedTaylorExpansion< T, A... >& x,
                                                   int n ) noexcept
{
    using tax::pow;
    return MixedTaylorExpansion< T, A... >{ pow( x.inner(), n ) };
}

/// `x^p` for a real exponent (axis set preserved; requires x.value() > 0).
template < typename T, typename... A >
[[nodiscard]] MixedTaylorExpansion< T, A... > pow( const MixedTaylorExpansion< T, A... >& x,
                                                   std::type_identity_t< T > p ) noexcept
{
    using tax::pow;
    return MixedTaylorExpansion< T, A... >{ pow( x.inner(), p ) };
}

/// `x^p` for a Taylor-valued exponent over the union of the operands' axes.
template < typename T, typename... A, typename... B >
[[nodiscard]] auto pow( const MixedTaylorExpansion< T, A... >& x,
                        const MixedTaylorExpansion< T, B... >& p ) noexcept
{
    using R = detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    using tax::pow;
    return R{ pow( x.template embed< R >().inner(), p.template embed< R >().inner() ) };
}

/// `s^x` for a scalar base (axis set unchanged).
template < typename T, typename... A >
[[nodiscard]] MixedTaylorExpansion< T, A... > pow( std::type_identity_t< T > s,
                                                   const MixedTaylorExpansion< T, A... >& x ) noexcept
{
    using tax::pow;
    return MixedTaylorExpansion< T, A... >{ pow( s, x.inner() ) };
}

/// `atan2(y, x)` over the union of the two operands' axis sets.
template < typename T, typename... A, typename... B >
[[nodiscard]] auto atan2( const MixedTaylorExpansion< T, A... >& y,
                          const MixedTaylorExpansion< T, B... >& x ) noexcept
{
    using R = detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    using tax::atan2;
    return R{ atan2( y.template embed< R >().inner(), x.template embed< R >().inner() ) };
}

/// `atan2(y, x)` with a constant `x` (axis set unchanged).
template < typename T, typename... A >
[[nodiscard]] MixedTaylorExpansion< T, A... > atan2( const MixedTaylorExpansion< T, A... >& y,
                                                     std::type_identity_t< T > x ) noexcept
{
    using tax::atan2;
    return MixedTaylorExpansion< T, A... >{ atan2( y.inner(), x ) };
}

/// `atan2(y, x)` with a constant `y` (axis set unchanged).
template < typename T, typename... A >
[[nodiscard]] MixedTaylorExpansion< T, A... > atan2( std::type_identity_t< T > y,
                                                     const MixedTaylorExpansion< T, A... >& x ) noexcept
{
    using tax::atan2;
    return MixedTaylorExpansion< T, A... >{ atan2( y, x.inner() ) };
}

}  // namespace tax::named
