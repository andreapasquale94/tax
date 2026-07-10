#pragma once

// Fused math surface: operations that compute two coupled results in a single
// recurrence pass (see <tax/kernels/fused.hpp> for provenance and benchmarks).
// Pair-returning functions order the pair as spelled in the name:
//   sinCos(x)        {sin(x), cos(x)} — one coupled pass, the price of one.
//   sinhCosh(x)      {sinh(x), cosh(x)} — one shared exp(x)/exp(-x) pair.
//   sqrtInvSqrt(x)   {sqrt(x), 1/sqrt(x)} — use only when BOTH are consumed.
//   expSin(v, u)     exp(v)*sin(u)  — one coupled pass, ~1.4x vs exp(v)*sin(u).
//   expCos(v, u)     exp(v)*cos(u)  — likewise.
//   expSinCos(v, u)  {exp(v)*sin(u), exp(v)*cos(u)} — both for the price of one.
//
// Named and mixed-order named overloads live in tax::named below and are
// re-exported into tax::; two-operand forms compose in the union of the
// operands' axis sets, exactly like operator* / atan2.

#include <tax/core/mixed_named.hpp>
#include <tax/core/named.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/fused.hpp>
#include <tax/kernels/transcendental.hpp>
#include <tax/kernels/trigonometric.hpp>
#include <utility>

namespace tax
{

template < typename T, IndexScheme Scheme >
[[nodiscard]] auto sinCos( const TaylorExpansion< T, Scheme >& x ) noexcept
{
    std::pair< TaylorExpansion< T, Scheme >, TaylorExpansion< T, Scheme > > r;
    detail::kernels::seriesSinCos< T, Scheme >( r.first.coefficients(), r.second.coefficients(),
                                                x.coefficients() );
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] auto sinhCosh( const TaylorExpansion< T, Scheme >& x ) noexcept
{
    std::pair< TaylorExpansion< T, Scheme >, TaylorExpansion< T, Scheme > > r;
    detail::kernels::seriesSinhCosh< T, Scheme >( r.first.coefficients(), r.second.coefficients(),
                                                  x.coefficients() );
    return r;
}

/// Requires `x.value() > 0`. Worth calling only when both results are consumed
/// (e.g. r and 1/r^3); a single-output caller should use sqrt() or pow().
template < typename T, IndexScheme Scheme >
[[nodiscard]] auto sqrtInvSqrt( const TaylorExpansion< T, Scheme >& x ) noexcept
{
    std::pair< TaylorExpansion< T, Scheme >, TaylorExpansion< T, Scheme > > r;
    detail::kernels::seriesSqrtInvSqrt< T, Scheme >( r.first.coefficients(),
                                                     r.second.coefficients(), x.coefficients() );
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > expSin( const TaylorExpansion< T, Scheme >& v,
                                                   const TaylorExpansion< T, Scheme >& u ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesExpSin< T, Scheme >( r.coefficients(), v.coefficients(),
                                                u.coefficients() );
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > expCos( const TaylorExpansion< T, Scheme >& v,
                                                   const TaylorExpansion< T, Scheme >& u ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesExpCos< T, Scheme >( r.coefficients(), v.coefficients(),
                                                u.coefficients() );
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] auto expSinCos( const TaylorExpansion< T, Scheme >& v,
                              const TaylorExpansion< T, Scheme >& u ) noexcept
{
    std::pair< TaylorExpansion< T, Scheme >, TaylorExpansion< T, Scheme > > r;
    detail::kernels::seriesExpSinCos< T, Scheme >( r.first.coefficients(), r.second.coefficients(),
                                                   v.coefficients(), u.coefficients() );
    return r;
}

// Fused surface for sparse storage: the coupled results densify like the plain
// transcendentals, so each runs the dense fused pass and re-sparsifies both
// outputs. See math_unary.hpp for the rationale.

#define TAX_FUSED_SPARSE_UNARY_PAIR( FN )                                         \
    template < typename T, int N, int M >                                         \
    [[nodiscard]] auto FN(                                                        \
        const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x ) \
    {                                                                             \
        auto p = FN( x.dense() );                                                 \
        return std::pair{ sparse( p.first ), sparse( p.second ) };                \
    }

TAX_FUSED_SPARSE_UNARY_PAIR( sinCos )
TAX_FUSED_SPARSE_UNARY_PAIR( sinhCosh )
TAX_FUSED_SPARSE_UNARY_PAIR( sqrtInvSqrt )

#undef TAX_FUSED_SPARSE_UNARY_PAIR

/// Sparse `exp(v)*sin(u)`.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > expSin(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& v,
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& u )
{
    return sparse( expSin( v.dense(), u.dense() ) );
}

/// Sparse `exp(v)*cos(u)`.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > expCos(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& v,
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& u )
{
    return sparse( expCos( v.dense(), u.dense() ) );
}

/// Sparse `{exp(v)*sin(u), exp(v)*cos(u)}`.
template < typename T, int N, int M >
[[nodiscard]] auto expSinCos(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& v,
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& u )
{
    auto p = expSinCos( v.dense(), u.dense() );
    return std::pair{ sparse( p.first ), sparse( p.second ) };
}

}  // namespace tax

// Named (single-order) and mixed-order named overloads.

namespace tax::named
{

#define TAX_NAMED_FUSED_PAIR( FN )                                                \
    template < typename T, int N, typename... A >                                 \
    [[nodiscard]] auto FN( const NamedTaylorExpansion< T, N, A... >& a ) noexcept \
    {                                                                             \
        using R = NamedTaylorExpansion< T, N, A... >;                             \
        auto p = tax::FN( a.inner() );                                            \
        return std::pair{ R{ p.first }, R{ p.second } };                          \
    }                                                                             \
    template < typename T, typename... A >                                        \
    [[nodiscard]] auto FN( const MixedTaylorExpansion< T, A... >& a ) noexcept    \
    {                                                                             \
        using R = MixedTaylorExpansion< T, A... >;                                \
        auto p = tax::FN( a.inner() );                                            \
        return std::pair{ R{ p.first }, R{ p.second } };                          \
    }

TAX_NAMED_FUSED_PAIR( sinCos )
TAX_NAMED_FUSED_PAIR( sinhCosh )
TAX_NAMED_FUSED_PAIR( sqrtInvSqrt )

#undef TAX_NAMED_FUSED_PAIR

/// Fused `exp(v) * sin(u)` over the union of the two operands' axis sets.
template < typename T, int N, typename... A, typename... B >
[[nodiscard]] auto expSin( const NamedTaylorExpansion< T, N, A... >& v,
                           const NamedTaylorExpansion< T, N, B... >& u ) noexcept
{
    using R = detail::MergedNamedTaylorExpansion< T, N, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    return R{ tax::expSin( v.template embed< R >().inner(), u.template embed< R >().inner() ) };
}

/// Fused `exp(v) * cos(u)` over the union of the two operands' axis sets.
template < typename T, int N, typename... A, typename... B >
[[nodiscard]] auto expCos( const NamedTaylorExpansion< T, N, A... >& v,
                           const NamedTaylorExpansion< T, N, B... >& u ) noexcept
{
    using R = detail::MergedNamedTaylorExpansion< T, N, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    return R{ tax::expCos( v.template embed< R >().inner(), u.template embed< R >().inner() ) };
}

/// `{exp(v)*sin(u), exp(v)*cos(u)}` over the union of the two operands' axis sets.
template < typename T, int N, typename... A, typename... B >
[[nodiscard]] auto expSinCos( const NamedTaylorExpansion< T, N, A... >& v,
                              const NamedTaylorExpansion< T, N, B... >& u ) noexcept
{
    using R = detail::MergedNamedTaylorExpansion< T, N, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    auto p = tax::expSinCos( v.template embed< R >().inner(), u.template embed< R >().inner() );
    return std::pair{ R{ p.first }, R{ p.second } };
}

/// Fused `exp(v) * sin(u)` over the union of the two operands' (ordered) axis sets.
template < typename T, typename... A, typename... B >
[[nodiscard]] auto expSin( const MixedTaylorExpansion< T, A... >& v,
                           const MixedTaylorExpansion< T, B... >& u ) noexcept
{
    using R =
        detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >, detail::TypeList< B... > >;
    return R{ tax::expSin( v.template embed< R >().inner(), u.template embed< R >().inner() ) };
}

/// Fused `exp(v) * cos(u)` over the union of the two operands' (ordered) axis sets.
template < typename T, typename... A, typename... B >
[[nodiscard]] auto expCos( const MixedTaylorExpansion< T, A... >& v,
                           const MixedTaylorExpansion< T, B... >& u ) noexcept
{
    using R =
        detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >, detail::TypeList< B... > >;
    return R{ tax::expCos( v.template embed< R >().inner(), u.template embed< R >().inner() ) };
}

/// `{exp(v)*sin(u), exp(v)*cos(u)}` over the union of the two operands' (ordered) axis sets.
template < typename T, typename... A, typename... B >
[[nodiscard]] auto expSinCos( const MixedTaylorExpansion< T, A... >& v,
                              const MixedTaylorExpansion< T, B... >& u ) noexcept
{
    using R =
        detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >, detail::TypeList< B... > >;
    auto p = tax::expSinCos( v.template embed< R >().inner(), u.template embed< R >().inner() );
    return std::pair{ R{ p.first }, R{ p.second } };
}

}  // namespace tax::named

namespace tax
{
using named::expCos;
using named::expSin;
using named::expSinCos;
using named::sinCos;
using named::sinhCosh;
using named::sqrtInvSqrt;
}  // namespace tax
