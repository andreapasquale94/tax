#pragma once

#include <tax/core/scheme/isotropic.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/cauchy.hpp>
#include <type_traits>

namespace tax
{

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator+(
    const TaylorExpansion< T, Scheme >& a, const TaylorExpansion< T, Scheme >& b ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] + b[k];
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator+(
    const TaylorExpansion< T, Scheme >& a, std::type_identity_t< T > s ) noexcept
{
    TaylorExpansion< T, Scheme > r = a;
    r[0] += s;
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator+(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& a ) noexcept
{
    return a + s;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator-(
    const TaylorExpansion< T, Scheme >& a, const TaylorExpansion< T, Scheme >& b ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] - b[k];
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator-(
    const TaylorExpansion< T, Scheme >& a, std::type_identity_t< T > s ) noexcept
{
    TaylorExpansion< T, Scheme > r = a;
    r[0] -= s;
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator-(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& a ) noexcept
{
    TaylorExpansion< T, Scheme > r = -a;
    r[0] += s;
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator-(
    const TaylorExpansion< T, Scheme >& a ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator*(
    const TaylorExpansion< T, Scheme >& a, std::type_identity_t< T > s ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] * s;
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator*(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& a ) noexcept
{
    return a * s;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator/(
    const TaylorExpansion< T, Scheme >& a, std::type_identity_t< T > s ) noexcept
{
    return a * ( T( 1 ) / s );
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator/(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& a ) noexcept
{
    TaylorExpansion< T, Scheme > inv_a;
    detail::kernels::seriesReciprocal< T, Scheme >( inv_a.coefficients(), a.coefficients() );
    return inv_a * s;
}

// Cauchy (TE x TE) product.
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator*(
    const TaylorExpansion< T, Scheme >& a, const TaylorExpansion< T, Scheme >& b ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    tax::cauchyProduct< T, Scheme >( r.coefficients(), a.coefficients(), b.coefficients() );
    return r;
}

// TE / TE division via reciprocal.
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
constexpr TaylorExpansion< T, Scheme >& operator+=( TaylorExpansion< T, Scheme >& a,
                                                    const TaylorExpansion< T, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] += b[k];
    return a;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator-=( TaylorExpansion< T, Scheme >& a,
                                                    const TaylorExpansion< T, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] -= b[k];
    return a;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator+=( TaylorExpansion< T, Scheme >& a,
                                                    std::type_identity_t< T > s ) noexcept
{
    a[0] += s;
    return a;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator-=( TaylorExpansion< T, Scheme >& a,
                                                    std::type_identity_t< T > s ) noexcept
{
    a[0] -= s;
    return a;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator*=( TaylorExpansion< T, Scheme >& a,
                                                    std::type_identity_t< T > s ) noexcept
{
    for ( T& ak : a.coefficients() ) ak *= s;
    return a;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator/=( TaylorExpansion< T, Scheme >& a,
                                                    std::type_identity_t< T > s ) noexcept
{
    return a *= ( T( 1 ) / s );
}

/// In-place Cauchy product.
template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator*=( TaylorExpansion< T, Scheme >& a,
                                                    const TaylorExpansion< T, Scheme >& b ) noexcept
{
    std::array< T, Scheme::nCoeff > tmp{};
    tax::cauchyProduct< T, Scheme >( tmp, a.coefficients(), b.coefficients() );
    a.coefficients() = tmp;
    return a;
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
