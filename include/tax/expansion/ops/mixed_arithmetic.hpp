#pragma once

// Free-function arithmetic surface for MixedExpansion: operands embed into
// the union (max-order per shared axis) before delegating to the inner
// TaylorExpansion operators. Mirrors expansion/ops/named_arithmetic.hpp.

#include <tax/expansion/mixed.hpp>
#include <tax/expansion/ops/arithmetic.hpp>
#include <type_traits>

namespace tax::mixed
{

// Composition operators (union axis set, max order per shared axis). Each binary
// op embeds both operands into the merged mixed type, then delegates to the
// underlying TaylorExpansion operator on the (now box-compatible) inners.
#define TAX_MIXED_BINARY_OP( OP )                                                           \
    template < typename T, typename... A, typename... B >                                   \
    [[nodiscard]] constexpr auto operator OP( const MixedExpansion< T, A... >& a,           \
                                              const MixedExpansion< T, B... >& b ) noexcept \
    {                                                                                       \
        using R = detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >,          \
                                                      detail::TypeList< B... > >;           \
        return R{ a.template embed< R >().inner() OP b.template embed< R >().inner() };     \
    }

TAX_MIXED_BINARY_OP( +)
TAX_MIXED_BINARY_OP( -)
TAX_MIXED_BINARY_OP( * )
TAX_MIXED_BINARY_OP( / )

#undef TAX_MIXED_BINARY_OP

// Scalar combinations (axis set unchanged).

#define TAX_MIXED_SCALAR_OP( OP )                                                  \
    template < typename T, typename... A >                                         \
    [[nodiscard]] constexpr MixedExpansion< T, A... > operator OP(                 \
        const MixedExpansion< T, A... >& a, std::type_identity_t< T > s ) noexcept \
    {                                                                              \
        return MixedExpansion< T, A... >{ a.inner() OP s };                        \
    }

TAX_MIXED_SCALAR_OP( +)
TAX_MIXED_SCALAR_OP( -)
TAX_MIXED_SCALAR_OP( * )
TAX_MIXED_SCALAR_OP( / )

#undef TAX_MIXED_SCALAR_OP

template < typename T, typename... A >
[[nodiscard]] constexpr MixedExpansion< T, A... > operator+(
    std::type_identity_t< T > s, const MixedExpansion< T, A... >& a ) noexcept
{
    return a + s;
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedExpansion< T, A... > operator*(
    std::type_identity_t< T > s, const MixedExpansion< T, A... >& a ) noexcept
{
    return a * s;
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedExpansion< T, A... > operator-(
    std::type_identity_t< T > s, const MixedExpansion< T, A... >& a ) noexcept
{
    return MixedExpansion< T, A... >{ s - a.inner() };
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedExpansion< T, A... > operator/(
    std::type_identity_t< T > s, const MixedExpansion< T, A... >& a ) noexcept
{
    return MixedExpansion< T, A... >{ s / a.inner() };
}

template < typename T, typename... A >
[[nodiscard]] constexpr MixedExpansion< T, A... > operator-(
    const MixedExpansion< T, A... >& a ) noexcept
{
    return MixedExpansion< T, A... >{ -a.inner() };
}

}  // namespace tax::mixed
