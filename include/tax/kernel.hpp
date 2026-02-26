#pragma once

#include "tax/utils.hpp"

namespace tax::detail
{

template < typename T, std::size_t M >
constexpr void add( std::array< T, M >& out, const std::array< T, M >& rhs ) noexcept
{
    for ( std::size_t i = 0; i < M; ++i ) out[i] += rhs[i];
}

template < typename T, std::size_t M >
constexpr void sub( std::array< T, M >& out, const std::array< T, M >& rhs ) noexcept
{
    for ( std::size_t i = 0; i < M; ++i ) out[i] -= rhs[i];
}

template < typename T, std::size_t M >
constexpr void negate( std::array< T, M >& out ) noexcept
{
    for ( auto& v : out ) v = -v;
}

template < typename T, std::size_t M >
constexpr void scale( std::array< T, M >& out, T s ) noexcept
{
    for ( auto& v : out ) v *= s;
}

template < typename T, std::size_t M >
constexpr void scaleInv( std::array< T, M >& out, T s ) noexcept
{
    const T d = 1 / s;
    for ( auto& v : out ) v *= d;
}

template < typename T, std::size_t N >
[[nodiscard]] inline constexpr T truncConvolution( const std::array< T, N + 1 >& a,
                                                   const std::array< T, N + 1 >& b,
                                                   std::size_t until,
                                                   std::size_t from = 0 ) noexcept
{
    T acc{};
    for ( std::size_t j = from; j <= until; ++j ) acc += a[j] * b[until - j];
    return acc;
}

template < typename T, std::size_t N >
constexpr void seriesProduct1D( std::array< T, N + 1 >& out, const std::array< T, N + 1 >& a,
                                const std::array< T, N + 1 >& b ) noexcept
{
    for ( std::size_t k = 0; k <= N; ++k ) out[k] += truncConvolution< T, N >( a, b, k );
}

template < typename T, std::size_t N >
constexpr void seriesDivision1D( std::array< T, N + 1 >& out, const std::array< T, N + 1 >& num,
                                 const std::array< T, N + 1 >& den ) noexcept
{
    // Preconditions: den[0] != 0
    out[0] = num[0] / den[0];

    for ( std::size_t k = 1; k <= N; ++k )
    {
        T s = truncConvolution< T, N >( den, out, k, /*from*/ 1 );
        out[k] = ( num[k] - s ) / den[0];
    }
}

template < typename T, std::size_t N >
constexpr void seriesReciprocal1D( std::array< T, N + 1 >& out,
                                   const std::array< T, N + 1 >& a ) noexcept
{
    out[0] = T{ 1 } / a[0];
    for ( std::size_t k = 1; k <= N; ++k )
    {
        T s = truncConvolution< T, N >( a, out, k, /*from*/ 1 );
        out[k] = -s / a[0];
    }
}

}  // namespace tax::detail
