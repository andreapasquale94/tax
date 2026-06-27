#pragma once

#include <cstddef>
#include <tax/series/series.hpp>
#include <tax/series/taylor_basis.hpp>
#include <type_traits>

namespace tax
{

// ===========================================================================
// Basis-generic operator surface for NON-Taylor bases.
//
// The Taylor (monomial) basis already has a full, specialised operator surface
// (operators/arithmetic.hpp, math_unary.hpp, math_binary.hpp) reached through
// the `TaylorExpansion` alias. To avoid overload ambiguity those legacy
// operators stay authoritative for `TaylorBasis`; everything here is gated on
// `!is_same_v< B, TaylorBasis >` and serves Chebyshev (and future families).
// ===========================================================================

template < typename B >
concept NonTaylorBasis = Basis< B > && !std::is_same_v< B, TaylorBasis >;

// ---------------------------------------------------------------------------
// Linear-space operators
// ---------------------------------------------------------------------------

template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = a[k] + b[k];
    return r;
}

template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = a[k] - b[k];
    return r;
}

template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    const Expansion< T, B, Scheme >& a ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r = a;
    r[0] += s;
    return r;
}
template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    return a + s;
}

template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r = a;
    r[0] -= s;
    return r;
}
template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    return ( -a ) + s;
}

template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < r.nCoefficients; ++k ) r[k] = a[k] * s;
    return r;
}
template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    return a * s;
}

template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator/( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    return a * ( T{ 1 } / s );
}

// ---------------------------------------------------------------------------
// Expansion * Expansion — the basis-defined bilinear product
// ---------------------------------------------------------------------------

template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    B::template product< T, Scheme >( r.coefficients(), a.coefficients(), b.coefficients() );
    return r;
}

template < typename T, NonTaylorBasis B, typename Scheme >
constexpr Expansion< T, B, Scheme >& operator+=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] += b[k];
    return a;
}
template < typename T, NonTaylorBasis B, typename Scheme >
constexpr Expansion< T, B, Scheme >& operator-=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] -= b[k];
    return a;
}
template < typename T, NonTaylorBasis B, typename Scheme >
constexpr Expansion< T, B, Scheme >& operator*=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    a = a * b;
    return a;
}
template < typename T, NonTaylorBasis B, typename Scheme >
constexpr Expansion< T, B, Scheme >& operator*=( Expansion< T, B, Scheme >& a,
                                                 std::type_identity_t< T > s ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] *= s;
    return a;
}

// ---------------------------------------------------------------------------
// Integer power by repeated squaring (any product-bearing basis).
// ---------------------------------------------------------------------------

template < typename T, NonTaylorBasis B, typename Scheme >
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
        if constexpr ( requires( Expansion< T, B, Scheme > z ) { T{ 1 } / z; } )
            return T{ 1 } / acc;
    }
    return acc;
}

}  // namespace tax
