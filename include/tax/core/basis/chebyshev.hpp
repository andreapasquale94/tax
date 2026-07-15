// Monomial <-> Chebyshev (first-kind, T_n) basis conversion for dense,
// isotropic `TaylorExpansion`s. Useful independently of statistical moments —
// e.g. for range/enclosure analysis, since |T_n(x)| <= 1 on [-1, 1] gives
// tight, easily-bounded coefficients compared to the monomial basis.

#pragma once

#include <cmath>
#include <tax/core/basis/connection.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/core/taylor_expansion.hpp>

namespace tax::detail::basis
{

/// Connection coefficient for `x^n = 2^{1-n} sum_m w(n,m) C(n,m) T_{n-2m}(x)`,
/// where `w(n,m) = 1/2` when `n` is even and `m == n/2` (the middle term is
/// otherwise double-counted), else `1`. (DLMF 18.5.10 / Abramowitz & Stegun 22.3.)
template < typename T >
T chebyshevForwardCoeff( int n, int m ) noexcept
{
    const T binomVal = factorial< T >( n ) / ( factorial< T >( m ) * factorial< T >( n - m ) );
    const T w = ( n % 2 == 0 && m == n / 2 ) ? T( 0.5 ) : T( 1 );
    return w * binomVal * T( std::ldexp( 1.0, 1 - n ) );
}

/// Connection coefficient for `T_n(x) = sum_m chebyshevInverseCoeff(n,m) * x^{n-2m}`
/// (n >= 1): `(n/2) * (-1)^m * (n-m-1)! / (m! (n-2m)!) * 2^{n-2m}`; `T_0(x) = 1`.
/// (DLMF 18.5.11 / Abramowitz & Stegun 22.3.6.)
template < typename T >
T chebyshevInverseCoeff( int n, int m ) noexcept
{
    if ( n == 0 ) return T( 1 );
    T val = ( T( n ) / T( 2 ) ) * factorial< T >( n - m - 1 ) /
            ( factorial< T >( m ) * factorial< T >( n - 2 * m ) );
    val *= T( std::ldexp( 1.0, n - 2 * m ) );
    return ( m % 2 == 0 ) ? val : -val;
}

}  // namespace tax::detail::basis

namespace tax
{

/// Coefficients of an order-`N`, `M`-variate expansion expressed in the
/// Chebyshev (first-kind) product basis `T_alpha(x) = prod_i T_{alpha_i}(x_i)`.
/// Same graded-lex layout as the monomial coefficient array, but a distinct
/// type so it can't be fed directly into monomial-basis arithmetic by mistake.
template < typename T, int N, int M >
struct ChebyshevCoefficients
{
    using Data = Coeffs< T, N, M >;
    Data data{};

    /// Runtime multi-index Chebyshev-coefficient lookup.
    [[nodiscard]] constexpr T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return data[flatIndex< M >( alpha )];
    }

    /// Compile-time multi-index Chebyshev-coefficient lookup.
    template < int... Alpha >
    [[nodiscard]] constexpr T coeff() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ),
                       "coeff<Alpha...>(): arity must match variable count" );
        constexpr MultiIndex< M > a{ Alpha... };
        return data[flatIndex< M >( a )];
    }
};

/// Convert monomial-basis coefficients to the Chebyshev (first-kind) basis:
/// `f(x) = sum_alpha a_alpha x^alpha = sum_beta c_beta T_beta(x)`.
template < typename T, int N, int M >
[[nodiscard]] ChebyshevCoefficients< T, N, M > toChebyshev(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense >& f ) noexcept
{
    return ChebyshevCoefficients< T, N, M >{ detail::basis::separableBasisTransform< N, M, T >(
        f.coefficients(), detail::basis::chebyshevForwardCoeff< T > ) };
}

/// Convert Chebyshev-basis coefficients back to the monomial basis.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense > fromChebyshev(
    const ChebyshevCoefficients< T, N, M >& c ) noexcept
{
    using TE = TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense >;
    return TE{ detail::basis::separableBasisTransform< N, M, T >(
        c.data, detail::basis::chebyshevInverseCoeff< T > ) };
}

}  // namespace tax
