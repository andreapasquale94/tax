#pragma once

#include <cmath>
#include <concepts>
#include <numeric>
#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/trigonometric.hpp>
#include <tax/operators/math_unary.hpp>
#include <type_traits>

namespace tax
{

/// Integer power `out = x^n` via binary exponentiation.
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > pow( const TaylorExpansion< T, Scheme >& x,
                                                          int n ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesPowInt< T, Scheme >( r.coefficients(), x.coefficients(), n );
    return r;
}

/// Compile-time integer power `out = x^N` (constexpr). Prefer to `pow(x, n)`
/// whenever the exponent is a constant.
template < int N, typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > pow(
    const TaylorExpansion< T, Scheme >& x ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesPowInt< T, Scheme >( r.coefficients(), x.coefficients(), N );
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

/// Half-integer power `out = x^(K/2)`. Even `K` dispatches to the integer-power
/// chain (constexpr; requires `x.value() != 0` for negative K, valid for a
/// negative base); odd `K` runs the single real-exponent recurrence (requires
/// `x.value() > 0`). One seriesPow pass is the fastest spelling when only one
/// output is consumed; a caller needing sqrt(x) alongside x^(-K/2) should
/// combine sqrtInvSqrt with pow instead.
template < int K, typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > halfPow( const TaylorExpansion< T, Scheme >& x ) noexcept
{
    if constexpr ( K % 2 == 0 )
    {
        return pow< K / 2 >( x );  // integer power (constexpr, exact)
    } else
    {
        TaylorExpansion< T, Scheme > r;
        detail::kernels::seriesPow< T, Scheme >( r.coefficients(), x.coefficients(),
                                                 T( K ) / T( 2 ) );
        return r;
    }
}

/// Inverse square-root power `out = x^(-K/2) = 1/sqrt(x)^K` (K >= 1).
/// `invSqrtPow<3>(r2)` is the classic 1/r^3 of a squared radius.
/// Requires `x.value() > 0`.
template < int K, typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > invSqrtPow(
    const TaylorExpansion< T, Scheme >& x ) noexcept
{
    static_assert( K >= 1, "invSqrtPow<K>: K must be >= 1 (computes x^(-K/2))" );
    return halfPow< -K >( x );
}

/// Compile-time rational power `out = x^(Num/Den)`. The exponent is reduced to
/// lowest terms at compile time and bound to the cheapest kernel:
///   * integer exponent (Den | Num)  -> the integer-power chain (constexpr);
///   * denominator 2                  -> the sqrt / invsqrt chain (halfPow);
///   * otherwise                      -> one real-exponent recurrence.
/// So `pow<3,2>(x)` == `halfPow<3>(x)`, `pow<-3,2>` == `invSqrtPow<3>`,
/// `pow<6,3>` reduces to `pow<2>` (integer), and `pow<2,5>` is x^(2/5).
/// Requires `x.value() > 0` unless the reduced exponent is a non-negative
/// integer. Runtime-only except when it reduces to an integer power.
template < int Num, int Den, typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > pow(
    const TaylorExpansion< T, Scheme >& x ) noexcept
{
    static_assert( Den != 0, "pow<Num, Den>: denominator must be non-zero" );
    // Reduce Num/Den to lowest terms with a positive denominator.
    constexpr int g = std::gcd( Num < 0 ? -Num : Num, Den < 0 ? -Den : Den );
    constexpr int sign = Den < 0 ? -1 : 1;
    constexpr int n = sign * Num / g;                // signed numerator
    constexpr int m = ( Den < 0 ? -Den : Den ) / g;  // positive denominator

    if constexpr ( m == 1 )
        return pow< n >( x );  // integer power
    else if constexpr ( m == 2 )
        return halfPow< n >( x );  // sqrt / invsqrt chain
    else
    {
        TaylorExpansion< T, Scheme > r;
        detail::kernels::seriesPow< T, Scheme >( r.coefficients(), x.coefficients(),
                                                 T( n ) / T( m ) );
        return r;
    }
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

/// `atan2(y, x)` via the two-argument arctangent series kernel.
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
