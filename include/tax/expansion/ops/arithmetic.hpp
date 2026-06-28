#pragma once

#include <tax/expansion/detail/algebra.hpp>
#include <tax/expansion/detail/cauchy.hpp>
#include <tax/expansion/expansion.hpp>
#include <tax/expansion/scheme/isotropic.hpp>
#include <type_traits>

namespace tax
{

// ===========================================================================
// Dense arithmetic — basis-generic over Expansion< T, Basis, Scheme >.
//
// Every linear-space operation, and the bilinear product (which delegates to
// the basis' own B::product), is identical for every basis, so each is
// written once here over `BasisPolicy B`. Division is the exception: the
// expansion/expansion quotient and the scalar/expansion reciprocal use the
// Taylor recurrence kernels and stay TaylorBasis-specific — other families
// supply their own division where it is defined (e.g. bases/chebyshev_math.hpp).
//
// `TaylorExpansion< T, Scheme >` is the B = TaylorBasis instance of Expansion,
// so these templates cover the Taylor hot path as well as Chebyshev/Legendre/…
// ===========================================================================

// ---------------------------------------------------------------------------
// Addition
// ---------------------------------------------------------------------------

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] + b[k];
    return r;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r = a;
    r[0] += s;
    return r;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    return a + s;
}

// ---------------------------------------------------------------------------
// Subtraction
// ---------------------------------------------------------------------------

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] - b[k];
    return r;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r = a;
    r[0] -= s;
    return r;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    Expansion< T, B, Scheme > r;
    r[0] = s - a[0];
    for ( std::size_t k = 1; k < a.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

// ---------------------------------------------------------------------------
// Unary negation
// ---------------------------------------------------------------------------

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    const Expansion< T, B, Scheme >& a ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

// ---------------------------------------------------------------------------
// Scalar multiplication / division
// ---------------------------------------------------------------------------

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] * s;
    return r;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    return a * s;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator/( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    return a * ( T( 1 ) / s );
}

// ---------------------------------------------------------------------------
// Bilinear product (the basis-defined Cauchy/convolution product)
// ---------------------------------------------------------------------------

template < typename T, BasisPolicy B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    B::template product< T, Scheme >( r.coefficients(), a.coefficients(), b.coefficients() );
    return r;
}

// ---------------------------------------------------------------------------
// Compound assignment (dense)
// ---------------------------------------------------------------------------

template < typename T, BasisPolicy B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator+=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] += b[k];
    return a;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator-=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] -= b[k];
    return a;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator+=( Expansion< T, B, Scheme >& a,
                                                 std::type_identity_t< T > s ) noexcept
{
    a[0] += s;
    return a;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator-=( Expansion< T, B, Scheme >& a,
                                                 std::type_identity_t< T > s ) noexcept
{
    a[0] -= s;
    return a;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator*=( Expansion< T, B, Scheme >& a,
                                                 std::type_identity_t< T > s ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] *= s;
    return a;
}

template < typename T, BasisPolicy B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator/=( Expansion< T, B, Scheme >& a,
                                                 std::type_identity_t< T > s ) noexcept
{
    return a *= ( T( 1 ) / s );
}

/// In-place bilinear product.
template < typename T, BasisPolicy B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator*=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    a = a * b;
    return a;
}

// ===========================================================================
// Division (TaylorBasis only): the quotient and scalar reciprocal use the
// Taylor recurrence kernels. Other bases provide their own where defined.
// ===========================================================================

/// Scalar / expansion: `s / a = s * (1 / a)`.
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator/(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& a ) noexcept
{
    TaylorExpansion< T, Scheme > inv_a;
    detail::kernels::seriesReciprocal< T, Scheme >( inv_a.coefficients(), a.coefficients() );
    return inv_a * s;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator/(
    const TaylorExpansion< T, Scheme >& a, const TaylorExpansion< T, Scheme >& b ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesDivide< T, Scheme >( r.coefficients(), a.coefficients(),
                                                b.coefficients() );
    return r;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator/=( TaylorExpansion< T, Scheme >& a,
                                                    const TaylorExpansion< T, Scheme >& b ) noexcept
{
    std::array< T, Scheme::nCoeff > inv_b{};
    detail::kernels::seriesReciprocal< T, Scheme >( inv_b, b.coefficients() );
    std::array< T, Scheme::nCoeff > tmp{};
    tax::cauchyProduct< T, Scheme >( tmp, a.coefficients(), inv_b );
    a.coefficients() = tmp;
    return a;
}

}  // namespace tax
