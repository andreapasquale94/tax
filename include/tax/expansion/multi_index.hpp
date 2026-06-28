#pragma once

#include <array>
#include <cstddef>

namespace tax
{

/// Exponent vector `(a_0, ..., a_{M-1})` for multivariate monomials.
template < int M >
using MultiIndex = std::array< int, static_cast< std::size_t >( M ) >;

namespace detail
{

/// Binomial coefficient `n choose k`; returns 0 when arguments are out of range.
constexpr std::size_t binom( int n, int k ) noexcept
{
    if ( k < 0 || n < 0 || k > n ) return 0;
    if ( k == 0 || k == n ) return 1;
    if ( k > n - k ) k = n - k;
    std::size_t r = 1;
    for ( int i = 0; i < k; ++i )
    {
        r *= std::size_t( n - i );
        r /= std::size_t( i + 1 );
    }
    return r;
}

}  // namespace detail

/// Number of monomials with total degree `<= N` in `M` variables.
constexpr std::size_t numMonomials( int N, int M ) noexcept
{
    return detail::binom( N + M, M );
}

/// Storage type for `numMonomials(N, M)` coefficients.
template < typename T, int N, int M >
using Coeffs = std::array< T, numMonomials( N, M ) >;

/// Total degree `|a| = sum_i a[i]` of a multi-index.
template < std::size_t N >
constexpr int totalDegree( const std::array< int, N >& a ) noexcept
{
    int d = 0;
    for ( std::size_t i = 0; i < N; ++i ) d += a[i];
    return d;
}

/// Map a multi-index to the internal flat storage index.
template < int M >
constexpr std::size_t flatIndex( const MultiIndex< M >& alpha ) noexcept
{
    static_assert( M >= 1 );
    const int d = totalDegree( alpha );
    std::size_t idx = detail::binom( d + M - 1, M );
    int rem = d;
    for ( int i = 0; i < M - 1; ++i )
    {
        idx += detail::binom(
            rem - alpha[static_cast< std::size_t >( i )] + ( M - 2 - i ), M - 1 - i );
        rem -= alpha[static_cast< std::size_t >( i )];
    }
    return idx;
}

/// Map a flat storage index back to the corresponding multi-index.
template < int M >
constexpr MultiIndex< M > unflatIndex( std::size_t k ) noexcept
{
    static_assert( M >= 1 );
    MultiIndex< M > alpha{};

    int d = 0;
    while ( detail::binom( d + M, M ) <= k ) ++d;

    std::size_t rank = k - detail::binom( d + M - 1, M );
    int rem = d;
    for ( int i = 0; i < M - 1; ++i )
    {
        const int vars_left = M - i;
        for ( int ai = rem; ai >= 0; --ai )
        {
            const std::size_t block =
                detail::binom( rem - ai + vars_left - 2, vars_left - 2 );
            if ( rank < block )
            {
                alpha[static_cast< std::size_t >( i )] = ai;
                rem -= ai;
                break;
            }
            rank -= block;
        }
    }
    alpha[static_cast< std::size_t >( M - 1 )] = rem;
    return alpha;
}

/// Compile-time table mapping flat index k to its total degree.
template < int N, int M >
struct DegreeOf
{
    static constexpr std::size_t size = numMonomials( N, M );
    std::array< int, size > value{};

    constexpr DegreeOf() noexcept
    {
        for ( std::size_t k = 0; k < size; ++k )
        {
            value[k] = totalDegree( unflatIndex< M >( k ) );
        }
    }
};

}  // namespace tax
