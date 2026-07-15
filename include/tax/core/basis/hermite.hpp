// Monomial <-> (probabilists') Hermite basis conversion for dense, isotropic
// `TaylorExpansion`s. The probabilists' Hermite polynomials `He_n` are the
// orthogonal family for the standard-normal weight (`E[He_n(X)] = 0` for n>=1,
// `X ~ N(0,1)`), which is exactly what makes them the right basis for
// extracting statistical moments of a polynomial in Gaussian variables
// (see `tax/la/moments.hpp`).

#pragma once

#include <tax/core/basis/connection.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/core/taylor_expansion.hpp>

namespace tax::detail::basis
{

/// Connection coefficient for `x^n = sum_m hermiteForwardCoeff(n, m) * He_{n-2m}(x)`:
/// `coeff(n, m) = n! / (m! (n-2m)! 2^m)`.
template < typename T >
constexpr T hermiteForwardCoeff( int n, int m ) noexcept
{
    T pow2m{ 1 };
    for ( int i = 0; i < m; ++i ) pow2m *= T( 2 );
    return factorial< T >( n ) / ( factorial< T >( m ) * factorial< T >( n - 2 * m ) * pow2m );
}

/// Connection coefficient for `He_n(x) = sum_m hermiteInverseCoeff(n, m) * x^{n-2m}`:
/// `(-1)^m` times the forward coefficient.
template < typename T >
constexpr T hermiteInverseCoeff( int n, int m ) noexcept
{
    const T c = hermiteForwardCoeff< T >( n, m );
    return ( m % 2 == 0 ) ? c : -c;
}

}  // namespace tax::detail::basis

namespace tax
{

/// Coefficients of an order-`N`, `M`-variate expansion expressed in the
/// (probabilists') Hermite product basis `He_alpha(x) = prod_i He_{alpha_i}(x_i)`,
/// rather than the monomial basis. Same graded-lex layout and flat indexing as
/// the monomial coefficient array, but a distinct type so it can't be fed
/// directly into monomial-basis arithmetic (`+`, `*`, ...) by mistake.
template < typename T, int N, int M >
struct HermiteCoefficients
{
    using Data = Coeffs< T, N, M >;
    Data data{};

    /// Runtime multi-index Hermite-coefficient lookup.
    [[nodiscard]] constexpr T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return data[flatIndex< M >( alpha )];
    }

    /// Compile-time multi-index Hermite-coefficient lookup.
    template < int... Alpha >
    [[nodiscard]] constexpr T coeff() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ),
                       "coeff<Alpha...>(): arity must match variable count" );
        constexpr MultiIndex< M > a{ Alpha... };
        return data[flatIndex< M >( a )];
    }
};

/// Convert monomial-basis coefficients to the (probabilists') Hermite basis:
/// `f(x) = sum_alpha a_alpha x^alpha = sum_beta h_beta He_beta(x)`.
template < typename T, int N, int M >
[[nodiscard]] HermiteCoefficients< T, N, M > toHermite(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense >& f ) noexcept
{
    return HermiteCoefficients< T, N, M >{ detail::basis::separableBasisTransform< N, M, T >(
        f.coefficients(), detail::basis::hermiteForwardCoeff< T > ) };
}

/// Convert Hermite-basis coefficients back to the monomial basis.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense > fromHermite(
    const HermiteCoefficients< T, N, M >& h ) noexcept
{
    using TE = TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense >;
    return TE{ detail::basis::separableBasisTransform< N, M, T >(
        h.data, detail::basis::hermiteInverseCoeff< T > ) };
}

}  // namespace tax
