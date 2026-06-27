#pragma once

#include <array>
#include <cstddef>
#include <tax/core/scheme/concept.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/transcendental.hpp>
#include <tax/kernels/trigonometric.hpp>
#include <tax/series/series.hpp>
#include <type_traits>

namespace tax
{

// ===========================================================================
// Basis-independent linear-space operators (every basis, every scheme)
// ===========================================================================

template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = a[k] + b[k];
    return r;
}

template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = a[k] - b[k];
    return r;
}

template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    const Expansion< T, B, Scheme >& a ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r = a;
    r[0] += s;
    return r;
}
template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    return a + s;
}

template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r = a;
    r[0] -= s;
    return r;
}
template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    return ( -a ) + s;
}

template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = a[k] * s;
    return r;
}
template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    return a * s;
}

template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator/( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    return a * ( T{ 1 } / s );
}

// ---------------------------------------------------------------------------
// Expansion * Expansion — the basis-defined bilinear product
// ---------------------------------------------------------------------------

template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    B::template product< T, Scheme >( r.coefficients(), a.coefficients(), b.coefficients() );
    return r;
}

template < typename T, typename B, typename Scheme >
constexpr Expansion< T, B, Scheme >& operator+=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] += b[k];
    return a;
}
template < typename T, typename B, typename Scheme >
constexpr Expansion< T, B, Scheme >& operator-=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] -= b[k];
    return a;
}
template < typename T, typename B, typename Scheme >
constexpr Expansion< T, B, Scheme >& operator*=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    a = a * b;
    return a;
}
template < typename T, typename B, typename Scheme >
constexpr Expansion< T, B, Scheme >& operator*=( Expansion< T, B, Scheme >& a,
                                                 std::type_identity_t< T > s ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] *= s;
    return a;
}

// ===========================================================================
// Taylor-basis transcendental surface (scheme-generic — works multivariate)
//
// Reuses the library's existing `series*` recurrence kernels directly: an
// Expansion over TaylorBasis carries the same coefficient layout the kernels
// expect for that scheme.
// ===========================================================================

/// Expansion / Expansion (Taylor basis): reciprocal then Cauchy product.
template < typename T, typename Scheme >
[[nodiscard]] constexpr Expansion< T, TaylorBasis, Scheme > operator/(
    const Expansion< T, TaylorBasis, Scheme >& a,
    const Expansion< T, TaylorBasis, Scheme >& b ) noexcept
{
    std::array< T, Scheme::nCoeff > inv_b{};
    detail::kernels::seriesReciprocal< T, Scheme >( inv_b, b.coefficients() );
    Expansion< T, TaylorBasis, Scheme > r;
    tax::cauchyProduct< T, Scheme >( r.coefficients(), a.coefficients(), inv_b );
    return r;
}

template < typename T, typename Scheme >
[[nodiscard]] constexpr Expansion< T, TaylorBasis, Scheme > operator/(
    std::type_identity_t< T > s, const Expansion< T, TaylorBasis, Scheme >& a ) noexcept
{
    Expansion< T, TaylorBasis, Scheme > inv_a;
    detail::kernels::seriesReciprocal< T, Scheme >( inv_a.coefficients(), a.coefficients() );
    return inv_a * s;
}

#define TAX_SERIES_UNARY( NAME, KERNEL )                                            \
    template < typename T, typename Scheme >                                        \
    [[nodiscard]] Expansion< T, TaylorBasis, Scheme > NAME(                         \
        const Expansion< T, TaylorBasis, Scheme >& x ) noexcept                     \
    {                                                                               \
        Expansion< T, TaylorBasis, Scheme > r;                                      \
        detail::kernels::KERNEL< T, Scheme >( r.coefficients(), x.coefficients() ); \
        return r;                                                                   \
    }

TAX_SERIES_UNARY( square, seriesSquare )
TAX_SERIES_UNARY( cube, seriesCube )
TAX_SERIES_UNARY( reciprocal, seriesReciprocal )
TAX_SERIES_UNARY( sqrt, seriesSqrt )
TAX_SERIES_UNARY( cbrt, seriesCbrt )
TAX_SERIES_UNARY( exp, seriesExp )
TAX_SERIES_UNARY( log, seriesLog )
TAX_SERIES_UNARY( sinh, seriesSinh )
TAX_SERIES_UNARY( cosh, seriesCosh )
TAX_SERIES_UNARY( tanh, seriesTanh )
TAX_SERIES_UNARY( erf, seriesErf )
TAX_SERIES_UNARY( sin, seriesSin )
TAX_SERIES_UNARY( cos, seriesCos )
TAX_SERIES_UNARY( tan, seriesTan )
TAX_SERIES_UNARY( asin, seriesAsin )
TAX_SERIES_UNARY( acos, seriesAcos )
TAX_SERIES_UNARY( atan, seriesAtan )

#undef TAX_SERIES_UNARY

// ===========================================================================
// Powers (any basis with a product): integer power by repeated squaring.
// ===========================================================================

template < typename T, typename B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > pow( const Expansion< T, B, Scheme >& x,
                                                       int n ) noexcept
{
    if ( n == 0 ) return Expansion< T, B, Scheme >::constant( T{ 1 } );
    Expansion< T, B, Scheme > base = x;
    Expansion< T, B, Scheme > acc = Expansion< T, B, Scheme >::constant( T{ 1 } );
    int e = n < 0 ? -n : n;
    while ( e > 0 )
    {
        if ( e & 1 ) acc = acc * base;
        e >>= 1;
        if ( e ) base = base * base;
    }
    if ( n < 0 )
    {
        // Reciprocal exists only for bases that define series division (Taylor).
        if constexpr ( requires( Expansion< T, B, Scheme > z ) { T{ 1 } / z; } )
            return T{ 1 } / acc;
    }
    return acc;
}

}  // namespace tax
