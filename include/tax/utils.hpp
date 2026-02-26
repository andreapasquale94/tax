#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tax::detail
{

constexpr std::size_t binom( std::size_t n, std::size_t k ) noexcept
{
    if ( k > n ) return 0;
    if ( k == 0 || k == n ) return 1;
    if ( k > n - k ) k = n - k;
    std::size_t r = 1;
    for ( std::size_t i = 1; i <= k; ++i ) r = ( r * ( n - k + i ) ) / i;
    return r;
}

constexpr std::size_t factorial( std::size_t k ) noexcept
{
    std::size_t f = 1;
    for ( std::size_t i = 2; i <= k; ++i ) f *= i;
    return f;
}

template < std::size_t D >
constexpr std::size_t degree( const std::array< std::size_t, D >& a ) noexcept
{
    std::size_t s = 0;
    for ( auto v : a ) s += v;
    return s;
}

template < std::size_t D >
constexpr bool leq( const std::array< std::size_t, D >& a,
                    const std::array< std::size_t, D >& b ) noexcept
{
    for ( std::size_t i = 0; i < D; ++i )
        if ( a[i] > b[i] ) return false;
    return true;
}

template < std::size_t D >
constexpr std::size_t multifactorial( const std::array< std::size_t, D >& a ) noexcept
{
    std::size_t f = 1;
    for ( std::size_t i = 0; i < D; ++i ) f *= factorial( a[i] );
    return f;
}

}  // namespace tax::detail