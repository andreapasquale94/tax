#pragma once

#include <cmath>
#include <concepts>
#include <tax/expansion/detail/algebra.hpp>
#include <tax/expansion/detail/trigonometric.hpp>
#include <tax/expansion/expansion.hpp>
#include <tax/expansion/ops/math_unary.hpp>
#include <type_traits>

namespace tax
{

/// Compute `out = x^n` for an integer exponent via binary exponentiation.
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > pow( const TaylorExpansion< T, Scheme >& x,
                                                          int n ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesPowInt< T, Scheme >( r.coefficients(), x.coefficients(), n );
    return r;
}

/// Real-exponent power `out = x^p`. Requires `x.value() != 0`; not constexpr.
template < typename T, IndexScheme Scheme, std::floating_point P >
[[nodiscard]] TaylorExpansion< T, Scheme > pow( const TaylorExpansion< T, Scheme >& x,
                                                P p ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesPow< T, Scheme >( r.coefficients(), x.coefficients(), T( p ) );
    return r;
}

/// Taylor-valued exponent `out = a^b = exp(b*log(a))`. Requires `a.value() > 0`; not constexpr.
template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > pow( const TaylorExpansion< T, Scheme >& a,
                                                const TaylorExpansion< T, Scheme >& b ) noexcept
{
    return exp( b * log( a ) );
}

/// Scalar base, Taylor exponent `out = s^b = exp(b*log(s))`. Requires `s > 0`; not constexpr.
template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > pow( std::type_identity_t< T > s,
                                                const TaylorExpansion< T, Scheme >& b ) noexcept
{
    using std::log;
    return exp( b * log( s ) );
}

/// Compute `out = atan2(y, x)` using the two-argument arctangent series kernel.
template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > atan2( const TaylorExpansion< T, Scheme >& y,
                                                  const TaylorExpansion< T, Scheme >& x ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesAtan2< T, Scheme >( r.coefficients(), y.coefficients(),
                                               x.coefficients() );
    return r;
}

/// `atan2(y, x)` with a constant `x` (promoted to a flat expansion).
template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > atan2( const TaylorExpansion< T, Scheme >& y,
                                                  std::type_identity_t< T > x ) noexcept
{
    return atan2( y, TaylorExpansion< T, Scheme >{ x } );
}

/// `atan2(y, x)` with a constant `y` (promoted to a flat expansion).
template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > atan2( std::type_identity_t< T > y,
                                                  const TaylorExpansion< T, Scheme >& x ) noexcept
{
    return atan2( TaylorExpansion< T, Scheme >{ y }, x );
}

}  // namespace tax
