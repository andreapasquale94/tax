#pragma once

#include <array>
#include <cstddef>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/transcendental.hpp>
#include <tax/kernels/trigonometric.hpp>
#include <tax/series/series.hpp>
#include <type_traits>

namespace tax
{

// ===========================================================================
// Basis-independent linear-space operators (work for every basis)
// ===========================================================================

template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator+( const Series< B, N, T >& a,
                                                     const Series< B, N, T >& b ) noexcept
{
    Series< B, N, T > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = a[k] + b[k];
    return r;
}

template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator-( const Series< B, N, T >& a,
                                                     const Series< B, N, T >& b ) noexcept
{
    Series< B, N, T > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = a[k] - b[k];
    return r;
}

template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator-( const Series< B, N, T >& a ) noexcept
{
    Series< B, N, T > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator+( const Series< B, N, T >& a,
                                                     std::type_identity_t< T > s ) noexcept
{
    Series< B, N, T > r = a;
    r[0] += s;
    return r;
}
template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator+( std::type_identity_t< T > s,
                                                     const Series< B, N, T >& a ) noexcept
{
    return a + s;
}

template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator-( const Series< B, N, T >& a,
                                                     std::type_identity_t< T > s ) noexcept
{
    Series< B, N, T > r = a;
    r[0] -= s;
    return r;
}
template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator-( std::type_identity_t< T > s,
                                                     const Series< B, N, T >& a ) noexcept
{
    return ( -a ) + s;
}

template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator*( const Series< B, N, T >& a,
                                                     std::type_identity_t< T > s ) noexcept
{
    Series< B, N, T > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = a[k] * s;
    return r;
}
template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator*( std::type_identity_t< T > s,
                                                     const Series< B, N, T >& a ) noexcept
{
    return a * s;
}

template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator/( const Series< B, N, T >& a,
                                                     std::type_identity_t< T > s ) noexcept
{
    return a * ( T{ 1 } / s );
}

// ---------------------------------------------------------------------------
// Series * Series — the basis-defined bilinear product
// ---------------------------------------------------------------------------

template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > operator*( const Series< B, N, T >& a,
                                                     const Series< B, N, T >& b ) noexcept
{
    Series< B, N, T > r;
    B::template product< T, N >( r.coefficients(), a.coefficients(), b.coefficients() );
    return r;
}

template < typename B, int N, typename T >
constexpr Series< B, N, T >& operator+=( Series< B, N, T >& a, const Series< B, N, T >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] += b[k];
    return a;
}
template < typename B, int N, typename T >
constexpr Series< B, N, T >& operator-=( Series< B, N, T >& a, const Series< B, N, T >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] -= b[k];
    return a;
}
template < typename B, int N, typename T >
constexpr Series< B, N, T >& operator*=( Series< B, N, T >& a, const Series< B, N, T >& b ) noexcept
{
    a = a * b;
    return a;
}
template < typename B, int N, typename T >
constexpr Series< B, N, T >& operator*=( Series< B, N, T >& a,
                                         std::type_identity_t< T > s ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] *= s;
    return a;
}

// ===========================================================================
// Taylor-basis transcendental surface
//
// The monomial basis composes elementary functions with a series via the
// classical ODE recurrences. We reuse the library's existing univariate
// kernels directly — `Series< TaylorBasis, N, T >` carries the same
// std::array< T, N + 1 > coefficient layout as `IsotropicScheme< N, 1 >`.
// ===========================================================================

namespace detail
{
template < typename T, int N >
using IsoUni = tax::IsotropicScheme< N, 1 >;
}  // namespace detail

/// Series / Series division (Taylor basis only): reciprocal then Cauchy product.
template < int N, typename T >
[[nodiscard]] constexpr Series< TaylorBasis, N, T > operator/(
    const Series< TaylorBasis, N, T >& a, const Series< TaylorBasis, N, T >& b ) noexcept
{
    std::array< T, std::size_t( N ) + 1 > inv_b{};
    detail::kernels::seriesReciprocal< T, detail::IsoUni< T, N > >( inv_b, b.coefficients() );
    Series< TaylorBasis, N, T > r;
    detail::kernels::cauchyProduct< T, N, 1 >( r.coefficients(), a.coefficients(), inv_b );
    return r;
}

template < int N, typename T >
[[nodiscard]] constexpr Series< TaylorBasis, N, T > operator/(
    std::type_identity_t< T > s, const Series< TaylorBasis, N, T >& a ) noexcept
{
    Series< TaylorBasis, N, T > inv_a;
    detail::kernels::seriesReciprocal< T, detail::IsoUni< T, N > >( inv_a.coefficients(),
                                                                    a.coefficients() );
    return inv_a * s;
}

#define TAX_SERIES_UNARY( NAME, KERNEL )                                          \
    template < int N, typename T >                                                \
    [[nodiscard]] Series< TaylorBasis, N, T > NAME(                               \
        const Series< TaylorBasis, N, T >& x ) noexcept                           \
    {                                                                             \
        Series< TaylorBasis, N, T > r;                                            \
        detail::kernels::KERNEL< T, detail::IsoUni< T, N > >( r.coefficients(),   \
                                                              x.coefficients() ); \
        return r;                                                                 \
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

template < typename B, int N, typename T >
[[nodiscard]] constexpr Series< B, N, T > pow( const Series< B, N, T >& x, int n ) noexcept
{
    if ( n == 0 ) return Series< B, N, T >::constant( T{ 1 } );
    Series< B, N, T > base = x;
    Series< B, N, T > acc = Series< B, N, T >::constant( T{ 1 } );
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
        // The branch is compiled only when that operator is available.
        if constexpr ( requires( Series< B, N, T > z ) { T{ 1 } / z; } ) return T{ 1 } / acc;
    }
    return acc;
}

}  // namespace tax
