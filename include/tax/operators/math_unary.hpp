#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/transcendental.hpp>
#include <tax/kernels/trigonometric.hpp>

namespace tax
{

// Dense unary math wrappers, generated from two macros: TAX_UNARY_OP_CE is
// constexpr (pure recurrence); TAX_UNARY_OP is runtime-only (the kernel
// evaluates std::exp/sin/... at the constant term).
//
// Domain preconditions on x.value() (violations yield inf/nan; no throw):
//   sqrt: x0 > 0   reciprocal/cbrt: x0 != 0   log: x0 > 0
//   acosh: x0 > 1   atanh/asin/acos: |x0| < 1

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

// Pure-polynomial recurrences (constexpr).
TAX_UNARY_OP_CE( square, seriesSquare )
TAX_UNARY_OP_CE( cube, seriesCube )
TAX_UNARY_OP_CE( reciprocal, seriesReciprocal )

TAX_UNARY_OP( sqrt, seriesSqrt )
TAX_UNARY_OP( cbrt, seriesCbrt )
TAX_UNARY_OP( exp, seriesExp )
TAX_UNARY_OP( log, seriesLog )
TAX_UNARY_OP( sinh, seriesSinh )
TAX_UNARY_OP( cosh, seriesCosh )
TAX_UNARY_OP( tanh, seriesTanh )
TAX_UNARY_OP( asinh, seriesAsinh )
TAX_UNARY_OP( acosh, seriesAcosh )
TAX_UNARY_OP( atanh, seriesAtanh )
TAX_UNARY_OP( erf, seriesErf )
TAX_UNARY_OP( sin, seriesSin )
TAX_UNARY_OP( cos, seriesCos )
TAX_UNARY_OP( tan, seriesTan )
TAX_UNARY_OP( asin, seriesAsin )
TAX_UNARY_OP( acos, seriesAcos )
TAX_UNARY_OP( atan, seriesAtan )

#undef TAX_UNARY_OP
#undef TAX_UNARY_OP_CE

}  // namespace tax
