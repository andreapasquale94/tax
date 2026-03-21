#pragma once

#include <tax/utils/enumeration.hpp>

namespace tax::detail
{

template < typename T, int N, int M >
/**
 * @brief Truncated multivariate Cauchy product `out = f * g`.
 * @details Output is truncated to total degree `N`.
 */
constexpr void cauchyProduct( std::array< T, numMonomials( N, M ) >& out,
                              const std::array< T, numMonomials( N, M ) >& f,
                              const std::array< T, numMonomials( N, M ) >& g ) noexcept
{
    out = {};

    if constexpr ( M == 1 )
    {
        for ( int d = 0; d <= N; ++d )
            for ( int k = 0; k <= d; ++k ) out[d] += f[k] * g[d - k];
    } else
    {
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                forEachSubIndex< M >( alpha,
                                      [&]( auto bi, auto gi ) { out[ai] += f[bi] * g[gi]; } );
            } );
        }
    }
}

template < typename T, int N, int M >
/**
 * @brief Truncated self-product `out = f * f`, exploiting symmetry.
 * @details Enumerates each unordered pair {beta, gamma} with beta+gamma=alpha only once,
 *          doubling the off-diagonal contribution. Yields ~2x fewer multiplications than
 *          a general cauchyProduct call.
 */
constexpr void cauchySelfProduct( std::array< T, numMonomials( N, M ) >& out,
                                  const std::array< T, numMonomials( N, M ) >& f ) noexcept
{
    out = {};

    if constexpr ( M == 1 )
    {
        for ( int d = 0; d <= N; ++d )
        {
            // Enumerate only k <= d-k, i.e. k <= d/2.
            for ( int k = 0; k + k < d; ++k ) out[d] += T{ 2 } * f[k] * f[d - k];
            if ( d % 2 == 0 ) out[d] += f[d / 2] * f[d / 2];
        }
    } else
    {
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                forEachSubIndex< M >( alpha, [&]( auto bi, auto gi ) {
                    if ( bi < gi )
                        out[ai] += T{ 2 } * f[bi] * f[gi];
                    else if ( bi == gi )
                        out[ai] += f[bi] * f[bi];
                } );
            } );
        }
    }
}

template < typename T, int N, int M >
/**
 * @brief Truncated multivariate Cauchy accumulate `out += f * g`.
 * @details Contribution is truncated to total degree `N`.
 */
constexpr void cauchyAccumulate( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& f,
                                 const std::array< T, numMonomials( N, M ) >& g ) noexcept
{
    if constexpr ( M == 1 )
    {
        for ( int d = 0; d <= N; ++d )
            for ( int k = 0; k <= d; ++k ) out[d] += f[k] * g[d - k];
    } else
    {
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                forEachSubIndex< M >( alpha,
                                      [&]( auto bi, auto gi ) { out[ai] += f[bi] * g[gi]; } );
            } );
        }
    }
}

}  // namespace tax::detail
