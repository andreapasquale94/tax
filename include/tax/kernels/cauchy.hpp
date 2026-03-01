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
                forEachSubIndex< M >( alpha, [&]( auto bi, auto gi ) {
                    out[ai] += f[bi] * g[gi];
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
                forEachSubIndex< M >( alpha, [&]( auto bi, auto gi ) {
                    out[ai] += f[bi] * g[gi];
                } );
            } );
        }
    }
}

}  // namespace tax::detail
