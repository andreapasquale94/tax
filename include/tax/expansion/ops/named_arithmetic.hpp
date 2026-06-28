#pragma once

// Free-function arithmetic surface for NamedExpansion (any basis): operands over
// different axis sets are embedded into the union before the dense kernels run,
// so the result type tracks the union of axes. Mirrors expansion/ops/arithmetic.hpp
// for the unnamed dense type. The inner operators are basis-dispatched, so this
// surface serves Taylor, Chebyshev, … alike (each op is available
// wherever the inner expansion defines it).

#include <tax/expansion/bases/operators.hpp>
#include <tax/expansion/named.hpp>
#include <tax/expansion/ops/arithmetic.hpp>
#include <type_traits>

namespace tax::named
{

// ---------------------------------------------------------------------------
// Composition operators (axis sets merged into their union)
// ---------------------------------------------------------------------------

#define TAX_NAMED_BINARY_OP( OP )                                                       \
    template < typename T, typename Basis, int N, typename... A, typename... B >        \
    [[nodiscard]] constexpr auto operator OP(                                           \
        const NamedExpansion< T, Basis, N, A... >& a,                                   \
        const NamedExpansion< T, Basis, N, B... >& b ) noexcept                         \
    {                                                                                   \
        using R = detail::MergedNamedExpansion< T, Basis, N, detail::TypeList< A... >,  \
                                                detail::TypeList< B... > >;             \
        return R{ a.template embed< R >().inner() OP b.template embed< R >().inner() }; \
    }

TAX_NAMED_BINARY_OP( +)
TAX_NAMED_BINARY_OP( -)
TAX_NAMED_BINARY_OP( * )
TAX_NAMED_BINARY_OP( / )

#undef TAX_NAMED_BINARY_OP

// --- Scalar combinations (axis set unchanged) ------------------------------

#define TAX_NAMED_SCALAR_OP( OP )                                                            \
    template < typename T, typename Basis, int N, typename... A >                            \
    [[nodiscard]] constexpr NamedExpansion< T, Basis, N, A... > operator OP(                 \
        const NamedExpansion< T, Basis, N, A... >& a, std::type_identity_t< T > s ) noexcept \
    {                                                                                        \
        return NamedExpansion< T, Basis, N, A... >{ a.inner() OP s };                        \
    }

TAX_NAMED_SCALAR_OP( +)
TAX_NAMED_SCALAR_OP( -)
TAX_NAMED_SCALAR_OP( * )
TAX_NAMED_SCALAR_OP( / )

#undef TAX_NAMED_SCALAR_OP

template < typename T, typename Basis, int N, typename... A >
[[nodiscard]] constexpr NamedExpansion< T, Basis, N, A... > operator+(
    std::type_identity_t< T > s, const NamedExpansion< T, Basis, N, A... >& a ) noexcept
{
    return a + s;
}

template < typename T, typename Basis, int N, typename... A >
[[nodiscard]] constexpr NamedExpansion< T, Basis, N, A... > operator*(
    std::type_identity_t< T > s, const NamedExpansion< T, Basis, N, A... >& a ) noexcept
{
    return a * s;
}

template < typename T, typename Basis, int N, typename... A >
[[nodiscard]] constexpr NamedExpansion< T, Basis, N, A... > operator-(
    std::type_identity_t< T > s, const NamedExpansion< T, Basis, N, A... >& a ) noexcept
{
    return NamedExpansion< T, Basis, N, A... >{ s - a.inner() };
}

template < typename T, typename Basis, int N, typename... A >
[[nodiscard]] constexpr NamedExpansion< T, Basis, N, A... > operator/(
    std::type_identity_t< T > s, const NamedExpansion< T, Basis, N, A... >& a ) noexcept
{
    return NamedExpansion< T, Basis, N, A... >{ s / a.inner() };
}

template < typename T, typename Basis, int N, typename... A >
[[nodiscard]] constexpr NamedExpansion< T, Basis, N, A... > operator-(
    const NamedExpansion< T, Basis, N, A... >& a ) noexcept
{
    return NamedExpansion< T, Basis, N, A... >{ -a.inner() };
}

}  // namespace tax::named
