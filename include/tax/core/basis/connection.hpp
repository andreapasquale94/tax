// Shared machinery for monomial <-> classical-orthogonal-polynomial basis
// conversion (Hermite, Chebyshev, ...). Every basis considered here has the
// same even/odd symmetry as the monomials themselves, so a degree-n basis
// polynomial only ever mixes with monomials/basis polynomials of degree
// n, n-2, n-4, .... That shared "two steps down, same parity" structure is
// what `separableBasisTransform` walks; each concrete basis only supplies its
// own connection-coefficient function.

#pragma once

#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>

namespace tax::detail::basis
{

/// `n!`, accumulated in `T` (matches `TaylorExpansion::derivative`'s convention of
/// accumulating factorials in the scalar type rather than `std::size_t`, which
/// overflows at 21!).
template < typename T >
constexpr T factorial( int n ) noexcept
{
    T r{ 1 };
    for ( int i = 2; i <= n; ++i ) r *= T( i );
    return r;
}

/// Separable multivariate basis conversion.
///
/// Given monomial-degree-graded input coefficients `in` and a connection
/// coefficient function `coeff(n, m)` such that (for the univariate case)
/// `x^n = sum_{m=0}^{floor(n/2)} coeff(n, m) * Basis_{n-2m}(x)`, computes the
/// multivariate transform
///
///   out[beta] = sum_{m: |beta| + 2|m| <= N} in[beta + 2m] * prod_i coeff(beta_i + 2m_i, m_i)
///
/// which holds because both the monomial basis and any basis sharing this
/// parity structure factor over axes: `x^alpha = prod_i x_i^{alpha_i}` and
/// `Basis_alpha(x) = prod_i Basis_{alpha_i}(x_i)` for independent axes.
/// Passing the inverse connection coefficient function performs the reverse
/// (basis -> monomial) conversion.
template < int N, int M, typename T, typename CoeffFn >
[[nodiscard]] Coeffs< T, N, M > separableBasisTransform( const Coeffs< T, N, M >& in,
                                                         CoeffFn&& coeff )
{
    Coeffs< T, N, M > out{};
    constexpr int mMax = N / 2;
    forEachMonomial< M, N >( [&]( const MultiIndex< M >& beta ) {
        const int betaDeg = totalDegree( beta );
        T acc{};
        forEachMonomial< M, mMax >( [&]( const MultiIndex< M >& m ) {
            if ( betaDeg + 2 * totalDegree( m ) > N ) return;
            MultiIndex< M > alpha{};
            T weight{ 1 };
            for ( int i = 0; i < M; ++i )
            {
                alpha[std::size_t( i )] = beta[std::size_t( i )] + 2 * m[std::size_t( i )];
                weight *= coeff( alpha[std::size_t( i )], m[std::size_t( i )] );
            }
            acc += in[flatIndex< M >( alpha )] * weight;
        } );
        out[flatIndex< M >( beta )] = acc;
    } );
    return out;
}

}  // namespace tax::detail::basis
