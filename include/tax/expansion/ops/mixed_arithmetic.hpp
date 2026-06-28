#pragma once

// Free-function arithmetic surface for MixedTaylorExpansion: operands embed into
// the union (max-order per shared axis) before delegating to the inner
// TaylorExpansion operators. Mirrors operators/named_arithmetic.hpp.

#include <tax/expansion/mixed_named.hpp>
#include <tax/expansion/ops/arithmetic.hpp>
#include <type_traits>

namespace tax::named
{

// Composition operators (union axis set, max order per shared axis). Each binary
// op embeds both operands into the merged mixed type, then delegates to the
// underlying TaylorExpansion operator on the (now box-compatible) inners.
#define TAX_MIXED_BINARY_OP( OP )                                                                 \
    template < typename T, typename... A, typename... B >                                         \
    [[nodiscard]] constexpr auto operator OP( const MixedTaylorExpansion< T, A... >& a,           \
                                              const MixedTaylorExpansion< T, B... >& b ) noexcept \
    {                                                                                             \
        using R = detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >,                \
                                                      detail::TypeList< B... > >;                 \
        return R{ a.template embed< R >().inner() OP b.template embed< R >().inner() };           \
    }

TAX_MIXED_BINARY_OP( +)
TAX_MIXED_BINARY_OP( -)
TAX_MIXED_BINARY_OP( * )
TAX_MIXED_BINARY_OP( / )

#undef TAX_MIXED_BINARY_OP

// Scalar combinations (axis set unchanged).

#define TAX_MIXED_SCALAR_OP( OP )                                                        \
    template < typename T, typename... A >                                               \
    [[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator OP(                 \
        const MixedTaylorExpansion< T, A... >& a, std::type_identity_t< T > s ) noexcept \
    {                                                                                    \
        return MixedTaylorExpansion< T, A... >{ a.inner() OP s };                        \
    }

TAX_MIXED_SCALAR_OP( +)
TAX_MIXED_SCALAR_OP( -)
TAX_MIXED_SCALAR_OP( * )
TAX_MIXED_SCALAR_OP( / )

#undef TAX_MIXED_SCALAR_OP

template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator+(
    std::type_identity_t< T > s, const MixedTaylorExpansion< T, A... >& a ) noexcept
{
    return a + s;
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator*(
    std::type_identity_t< T > s, const MixedTaylorExpansion< T, A... >& a ) noexcept
{
    return a * s;
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator-(
    std::type_identity_t< T > s, const MixedTaylorExpansion< T, A... >& a ) noexcept
{
    return MixedTaylorExpansion< T, A... >{ s - a.inner() };
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator/(
    std::type_identity_t< T > s, const MixedTaylorExpansion< T, A... >& a ) noexcept
{
    return MixedTaylorExpansion< T, A... >{ s / a.inner() };
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > operator-(
    const MixedTaylorExpansion< T, A... >& a ) noexcept
{
    return MixedTaylorExpansion< T, A... >{ -a.inner() };
}

}  // namespace tax::named
