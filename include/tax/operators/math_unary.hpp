#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/sparse_subs.hpp>
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

// Sparse overloads.

/// Sparse `sqrt(f)` via support-set forward substitution.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > sqrt(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesSqrtSparse< T, N, M >( r.container(), x.container() );
    return r;
}

/// Sparse `1/f` via support-set forward substitution.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > reciprocal(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesReciprocalSparse< T, N, M >( r.container(), x.container() );
    return r;
}

// Remaining unary functions for sparse storage. A transcendental (or the
// polynomial square/cube/cbrt) turns a sparse operand into a result whose
// support is the additive closure of the input's — effectively dense — so we
// evaluate the dense recurrence and re-sparsify, which is both exact and about
// as fast as a bespoke sparse kernel would be. `sqrt` and `reciprocal` keep
// their native forward-substitution kernels above.

#define TAX_UNARY_SPARSE_BRIDGE( NAME )                                                \
    template < typename T, int N, int M >                                              \
    [[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > NAME( \
        const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )      \
    {                                                                                  \
        return sparse( NAME( x.dense() ) );                                            \
    }

TAX_UNARY_SPARSE_BRIDGE( square )
TAX_UNARY_SPARSE_BRIDGE( cube )
TAX_UNARY_SPARSE_BRIDGE( cbrt )
TAX_UNARY_SPARSE_BRIDGE( exp )
TAX_UNARY_SPARSE_BRIDGE( log )
TAX_UNARY_SPARSE_BRIDGE( sinh )
TAX_UNARY_SPARSE_BRIDGE( cosh )
TAX_UNARY_SPARSE_BRIDGE( tanh )
TAX_UNARY_SPARSE_BRIDGE( asinh )
TAX_UNARY_SPARSE_BRIDGE( acosh )
TAX_UNARY_SPARSE_BRIDGE( atanh )
TAX_UNARY_SPARSE_BRIDGE( erf )
TAX_UNARY_SPARSE_BRIDGE( sin )
TAX_UNARY_SPARSE_BRIDGE( cos )
TAX_UNARY_SPARSE_BRIDGE( tan )
TAX_UNARY_SPARSE_BRIDGE( asin )
TAX_UNARY_SPARSE_BRIDGE( acos )
TAX_UNARY_SPARSE_BRIDGE( atan )

#undef TAX_UNARY_SPARSE_BRIDGE

}  // namespace tax
