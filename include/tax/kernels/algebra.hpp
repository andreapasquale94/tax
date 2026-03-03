#pragma once

#include <cmath>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/ops.hpp>

namespace tax::detail
{

template < typename T, int N, int M >
/**
 * @brief Reciprocal series solve `a * out = 1`.
 * @details Requires `a[0] != 0`.
 */
constexpr void seriesReciprocal( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
{
    out = {};
    const T inv_a0 = T{ 1 } / a[0];

    if constexpr ( M == 1 )
    {
        out[0] = inv_a0;
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k <= d; ++k ) rhs -= a[k] * out[d - k];
            out[d] = rhs * inv_a0;
        }
    } else
    {
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = ( d == 0 ) ? T{ 1 } : T{ 0 };
                forEachSubIndex< M >( alpha, 1, d, [&]( auto bi, auto gi, int ) {
                    rhs -= a[bi] * out[gi];
                } );
                out[ai] = rhs * inv_a0;
            } );
        }
    }
}

template < typename T, int N, int M >
/// @brief Square series `out = a^2`.
constexpr void seriesSquare( std::array< T, numMonomials( N, M ) >& out,
                             const std::array< T, numMonomials( N, M ) >& a ) noexcept
{
    cauchyProduct< T, N, M >( out, a, a );
}

template < typename T, int N, int M >
/// @brief Cube series `out = a^3`.
constexpr void seriesCube( std::array< T, numMonomials( N, M ) >& out,
                           const std::array< T, numMonomials( N, M ) >& a ) noexcept
{
    if constexpr ( M == 1 )
    {
        for ( int d = 0; d <= N; ++d )
            for ( int j = 0; j <= d; ++j )
                for ( int k = 0; k <= d - j; ++k ) out[d] += a[j] * a[k] * a[d - j - k];
    } else
    {
        constexpr auto S = numMonomials( N, M );
        std::array< T, S > tmp;
        cauchyProduct< T, N, M >( tmp, a, a );
        cauchyProduct< T, N, M >( out, tmp, a );
    }
}

template < typename T, int N, int M >
/**
 * @brief Square-root series solve `out * out = a`.
 * @details Uses the principal branch from `sqrt(a[0])`.
 */
constexpr void seriesSqrt( std::array< T, numMonomials( N, M ) >& out,
                           const std::array< T, numMonomials( N, M ) >& a ) noexcept
{
    using std::sqrt;
    out = {};
    out[0] = sqrt( a[0] );
    const T inv2g0 = T{ 1 } / ( T{ 2 } * out[0] );

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = a[d];
            for ( int k = 1; k < d; ++k ) rhs -= out[k] * out[d - k];
            out[d] = rhs * inv2g0;
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = a[ai];
                forEachSubIndex< M >( alpha, 1, d - 1, [&]( auto bi, auto gi, int ) {
                    rhs -= out[bi] * out[gi];
                } );
                out[ai] = rhs * inv2g0;
            } );
        }
    }
}

template < typename T, int N, int M >
/**
 * @brief Compositional inverse (series reversion).
 * @details Given `f(x) = a[1]*x + a[2]*x^2 + ...` with `a[0] == 0` and `a[1] != 0`,
 *          computes `g(y)` such that `f(g(y)) = y`, truncated to order `N`.
 *          Univariate only (`M == 1`).
 */
constexpr void seriesInv( std::array< T, numMonomials( N, M ) >& out,
                          const std::array< T, numMonomials( N, M ) >& a ) noexcept
{
    static_assert( M == 1, "Compositional inverse is only supported for univariate series (M==1)" );

    out = {};
    if constexpr ( N == 0 ) return;

    const T inv_a1 = T{ 1 } / a[1];
    out[1] = inv_a1;

    if constexpr ( N >= 2 )
    {
        // gp[k][d] = coefficient of y^d in g(y)^k
        // We build these degree-by-degree.
        // gp[0] = [1, 0, ...], gp[1] = g (= out), gp[k] = g^k
        std::array< std::array< T, N + 1 >, N + 1 > gp{};
        gp[0][0] = T{ 1 };
        gp[1][1] = inv_a1;

        for ( int d = 2; d <= N; ++d )
        {
            // Update gp[k][d] for k = 2..d using: gp[k][d] = sum_{j=1}^{d-1} gp[k-1][d-j] * out[j]
            // (j=0 skipped since out[0]=0; j=d skipped since gp[k-1][0]=0 for k>=2)
            for ( int k = 2; k <= d; ++k )
            {
                T s = T{ 0 };
                for ( int j = 1; j <= d - 1; ++j ) s += gp[k - 1][d - j] * out[j];
                gp[k][d] = s;
            }

            // S = sum_{k=2}^{min(d,N)} a[k] * gp[k][d]
            T S = T{ 0 };
            const int kmax = d < N ? d : N;
            for ( int k = 2; k <= kmax; ++k ) S += a[k] * gp[k][d];

            out[d] = -S * inv_a1;
            gp[1][d] = out[d];
        }
    }
}

}  // namespace tax::detail
