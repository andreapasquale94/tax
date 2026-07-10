#pragma once

#include <cmath>
#include <numbers>
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

// Remaining unary functions for sparse storage. Each runs a native sparse
// recurrence: the forward-substitution drivers in sparse_subs.hpp walk only the
// additive-closure support of the input (never the full dense monomial set),
// and the auxiliary series (e.g. sqrt(1 - x^2) for asin) are themselves built
// from the native sparse operators. `sqrt` and `reciprocal` above use the same
// pattern.

/// `x^2` via the symmetric sparse self-product.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > square(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::sparseCauchySelfProduct< T, N, M >( r.container(), x.container() );
    return r;
}

/// `x^3` = x^2 * x, both sparse products.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > cube(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    return square( x ) * x;
}

/// `cbrt(x)` (real branch) via the real-power recurrence at exponent 1/3.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > cbrt(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    using std::cbrt;
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesPowFromSeedSparse< T, N, M >( r.container(), cbrt( x.value() ),
                                                         x.container(), T{ 1 } / T{ 3 } );
    return r;
}

/// `exp(x)` via the native product recurrence.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > exp(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesExpSparse< T, N, M >( r.container(), x.container() );
    return r;
}

/// `log(x)` via `x * out' = x'`. Requires `x.value() > 0`.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > log(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    using std::log;
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesDerivQuotientSparse< 1, T, N, M >( r.container(), log( x.value() ),
                                                              x.container(), x.container() );
    return r;
}

/// `sinh(x)`, `cosh(x)`, `tanh(x)` from a shared native `exp(x)` / `exp(-x)` pair.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > sinh(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    return ( exp( x ) - exp( -x ) ) * T{ 0.5 };
}

template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > cosh(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    return ( exp( x ) + exp( -x ) ) * T{ 0.5 };
}

template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > tanh(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    const auto ep = exp( x );
    const auto em = exp( -x );
    return ( ep - em ) / ( ep + em );
}

/// `asinh(x)` = ∫ x' / sqrt(1 + x^2).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > asinh(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    using std::asinh;
    const auto h = sqrt( T{ 1 } + square( x ) );
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesDerivQuotientSparse< 1, T, N, M >( r.container(), asinh( x.value() ),
                                                              x.container(), h.container() );
    return r;
}

/// `acosh(x)` = ∫ x' / sqrt(x^2 - 1). Requires `x.value() > 1`.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > acosh(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    using std::acosh;
    const auto h = sqrt( square( x ) - T{ 1 } );
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesDerivQuotientSparse< 1, T, N, M >( r.container(), acosh( x.value() ),
                                                              x.container(), h.container() );
    return r;
}

/// `atanh(x)` = ∫ x' / (1 - x^2). Requires `|x.value()| < 1`.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > atanh(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    using std::atanh;
    const auto h = T{ 1 } - square( x );
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesDerivQuotientSparse< 1, T, N, M >( r.container(), atanh( x.value() ),
                                                              x.container(), h.container() );
    return r;
}

/// `erf(x)` = ∫ x' * (2/sqrt(pi)) exp(-x^2).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > erf(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    using std::erf;
    using R = real_scalar_t< T >;
    const T two_over_sqrtpi = T( R{ 2 } * std::numbers::inv_sqrtpi_v< R > );
    const auto h = exp( -square( x ) ) * two_over_sqrtpi;
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesDerivProductSparse< T, N, M >( r.container(), erf( x.value() ),
                                                          x.container(), h.container() );
    return r;
}

/// `sin(x)`, `cos(x)`, `tan(x)` via the native coupled recurrence.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > sin(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > s, c;
    detail::kernels::seriesSinCosSparse< T, N, M >( s.container(), c.container(), x.container() );
    return s;
}

template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > cos(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > s, c;
    detail::kernels::seriesSinCosSparse< T, N, M >( s.container(), c.container(), x.container() );
    return c;
}

template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > tan(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > s, c;
    detail::kernels::seriesSinCosSparse< T, N, M >( s.container(), c.container(), x.container() );
    return s / c;
}

/// `asin(x)` = ∫ x' / sqrt(1 - x^2). Requires `|x.value()| < 1`.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > asin(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    using std::asin;
    const auto h = sqrt( T{ 1 } - square( x ) );
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesDerivQuotientSparse< 1, T, N, M >( r.container(), asin( x.value() ),
                                                              x.container(), h.container() );
    return r;
}

/// `acos(x)` = -∫ x' / sqrt(1 - x^2). Requires `|x.value()| < 1`.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > acos(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    using std::acos;
    const auto h = sqrt( T{ 1 } - square( x ) );
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesDerivQuotientSparse< -1, T, N, M >( r.container(), acos( x.value() ),
                                                               x.container(), h.container() );
    return r;
}

/// `atan(x)` = ∫ x' / (1 + x^2).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > atan(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    using std::atan;
    const auto h = T{ 1 } + square( x );
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesDerivQuotientSparse< 1, T, N, M >( r.container(), atan( x.value() ),
                                                              x.container(), h.container() );
    return r;
}

}  // namespace tax
