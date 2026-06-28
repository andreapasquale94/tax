#pragma once

#include <tax/expansion/detail/algebra.hpp>
#include <tax/expansion/detail/transcendental.hpp>
#include <tax/expansion/detail/trigonometric.hpp>
#include <tax/expansion/expansion.hpp>

namespace tax
{

// ===========================================================================
// Dense unary math wrappers
//
// Generated from one macro: TAX_UNARY_OP_CE is constexpr (pure recurrence);
// TAX_UNARY_OP is runtime-only (the kernel evaluates std::exp/sin/... at the
// constant term).
//
// Domain preconditions on x.value() (violations yield inf/nan; no throw):
//   sqrt: x0 > 0   reciprocal/cbrt: x0 != 0   log: x0 > 0
//   acosh: x0 > 1   atanh/asin/acos: |x0| < 1
// ===========================================================================

#define TAX_UNARY_OP_CE( NAME, KERNEL )                                             \
    template < typename T, IndexScheme Scheme >                                     \
    [[nodiscard]] constexpr TaylorExpansion< T, Scheme > NAME(                      \
        const TaylorExpansion< T, Scheme >& x ) noexcept                            \
    {                                                                               \
        TaylorExpansion< T, Scheme > r;                                             \
        detail::kernels::KERNEL< T, Scheme >( r.coefficients(), x.coefficients() ); \
        return r;                                                                   \
    }

#define TAX_UNARY_OP( NAME, KERNEL )                                                \
    template < typename T, IndexScheme Scheme >                                     \
    [[nodiscard]] TaylorExpansion< T, Scheme > NAME(                                \
        const TaylorExpansion< T, Scheme >& x ) noexcept                            \
    {                                                                               \
        TaylorExpansion< T, Scheme > r;                                             \
        detail::kernels::KERNEL< T, Scheme >( r.coefficients(), x.coefficients() ); \
        return r;                                                                   \
    }

#define TAX_UNARY_CE( NAME, KERNEL ) TAX_UNARY_OP_CE( NAME, KERNEL )
#define TAX_UNARY_RT( NAME, KERNEL ) TAX_UNARY_OP( NAME, KERNEL )
#include <tax/expansion/ops/unary_functions.def>
#undef TAX_UNARY_CE
#undef TAX_UNARY_RT

#undef TAX_UNARY_OP
#undef TAX_UNARY_OP_CE

}  // namespace tax
