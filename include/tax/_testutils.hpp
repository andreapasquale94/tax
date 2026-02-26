#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>

template < typename T >
bool expectNear( T a, T b, T atol = T( 1e-12 ), T rtol = T( 1e-12 ) )
{
    if constexpr ( std::is_floating_point_v< T > )
    {
        const T diff = std::abs( a - b );
        const T tol = atol + rtol * std::max( std::abs( a ), std::abs( b ) );
        return diff <= tol;
    } else
    {
        return a == b;
    }
}

template < typename T, std::size_t M >
void expectArrayNear( const std::array< T, M >& x, const std::array< T, M >& y,
                      T atol = T( 1e-12 ), T rtol = T( 1e-12 ) )
{
    for ( std::size_t i = 0; i < M; ++i )
    {
        EXPECT_TRUE( expectNear( x[i], y[i], atol, rtol ) )
            << "i=" << i << " computed=" << x[i] << " expected=" << y[i];
    }
}

template < std::size_t D >
inline std::ostream& operator<<( std::ostream& os, const std::array< std::size_t, D >& a )
{
    os << '(';
    for ( std::size_t i = 0; i < D; ++i )
    {
        if ( i ) os << ", ";
        os << a[i];
    }
    os << ')';
    return os;
}
