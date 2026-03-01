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

}  // namespace tax::detail
