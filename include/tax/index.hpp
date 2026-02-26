#pragma once

#include "tax/utils.hpp"

namespace tax::detail
{

// #multiindices with total degree exactly d in D dims: C(d + D - 1, D - 1)
template < std::size_t D >
constexpr std::size_t totalDegree( std::size_t d ) noexcept
{
    if constexpr ( D == 1 ) return 1;
    return binom( d + D - 1, D - 1 );
}

// #multiindices with total degree <= N in D dims: C(N + D, D)
template < std::size_t N, std::size_t D >
constexpr std::size_t countMonomials() noexcept
{
    if constexpr ( D == 1 ) return N + 1;
    return binom( N + D, D );
}

// graded-lex findIndex within fixed degree
template < std::size_t D >
constexpr std::size_t findIndexWithinDegree( const std::array< std::size_t, D >& alpha ) noexcept
{
    static_assert( D >= 1 );
    const std::size_t deg = degree< D >( alpha );

    if constexpr ( D == 1 )
    {
        (void)deg;
        return 0;
    } else
    {
        std::size_t r = 0;
        for ( std::size_t a0 = 0; a0 < alpha[0]; ++a0 )
        {
            r += totalDegree< D - 1 >( deg - a0 );
        }
        std::array< std::size_t, D - 1 > tail{};
        for ( std::size_t i = 1; i < D; ++i ) tail[i - 1] = alpha[i];
        r += findIndexWithinDegree< D - 1 >( tail );
        return r;
    }
}

template < std::size_t N, std::size_t D >
constexpr std::size_t findIndex( const std::array< std::size_t, D >& alpha ) noexcept
{
    if constexpr ( D == 1 )
    {
        return alpha[0];
    } else
    {
        const std::size_t deg = degree< D >( alpha );
        std::size_t prefix = 0;
        for ( std::size_t d = 0; d < deg; ++d ) prefix += totalDegree< D >( d );
        return prefix + findIndexWithinDegree< D >( alpha );
    }
}

// ---- precompute alphas in graded-lex order ---------------------------------

template < std::size_t D, class AlphaArr >
constexpr void makeCompositions( std::size_t deg, std::array< std::size_t, D >& cur,
                                 std::size_t pos, AlphaArr& out, std::size_t& idx )
{
    if ( pos + 1 == D )
    {
        cur[pos] = deg;
        out[idx++] = cur;
        return;
    }
    for ( std::size_t a = 0; a <= deg; ++a )
    {
        cur[pos] = a;
        makeCompositions< D >( deg - a, cur, pos + 1, out, idx );
    }
}

template < std::size_t N, std::size_t D >
constexpr auto makeIndices()
{
    using Index = std::array< std::size_t, D >;
    constexpr std::size_t M = countMonomials< N, D >();
    std::array< Index, M > out{};
    std::size_t idx = 0;
    Index cur{};
    for ( std::size_t deg = 0; deg <= N; ++deg )
    {
        makeCompositions< D >( deg, cur, 0, out, idx );
    }
    return out;
}

template < std::size_t N, std::size_t D >
constexpr auto makeIndexOffset()
{
    std::array< std::size_t, N + 2 > start{};  // start[d], start[N+1]=M
    std::size_t s = 0;
    for ( std::size_t d = 0; d <= N; ++d )
    {
        start[d] = s;
        s += totalDegree< D >( d );
    }
    start[N + 1] = s;
    return start;
}

// Total-degree (≤N) index set, graded-lex, with precomputed blocks.
template < std::size_t N, std::size_t D >
struct IndexSet
{
    static constexpr std::size_t order = N;
    static constexpr std::size_t dim = D;
    static constexpr std::size_t size = countMonomials< N, D >();
    using Index = std::array< std::size_t, D >;

    static constexpr auto data = makeIndices< N, D >();
    static constexpr auto offset = makeIndexOffset< N, D >();

    static constexpr std::size_t indexOf( const Index& a ) noexcept
    {
        return findIndex< N, D >( a );
    }

    static constexpr std::size_t degreeOf( const Index& a ) noexcept { return degree< D >( a ); }
};

// Specialization for D == 1
template < std::size_t N >
struct IndexSet< N, 1 >
{
    static constexpr std::size_t order = N;
    static constexpr std::size_t dim = 1;
    static constexpr std::size_t size = N + 1;

    // In 1D, the "multi-index" is just the exponent k
    using Index = std::size_t;

    // List of indices (0,1,...,N)
    static constexpr std::array< Index, size > data = [] {
        std::array< Index, size > a{};
        for ( std::size_t k = 0; k < size; ++k ) a[k] = k;
        return a;
    }();

    // Degree block starts: deg d starts at d
    static constexpr std::array< std::size_t, N + 2 > offset = [] {
        std::array< std::size_t, N + 2 > s{};
        for ( std::size_t d = 0; d <= N + 1; ++d ) s[d] = d;
        return s;
    }();

    static constexpr std::size_t indexOf( Index k ) noexcept { return k; }
    static constexpr std::size_t degreeOf( Index k ) noexcept { return k; }
};

}  // namespace tax::detail