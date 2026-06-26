#pragma once

#include <ostream>
#include <sstream>
#include <string>
#include <tax/series/series.hpp>

namespace tax
{

/// Human-readable series in its own basis, e.g. "1 + 2*x + 3*x^2" or
/// "1 + 2*T_1 + 3*T_2". Zero coefficients are omitted; the zero series prints
/// as "0".
template < typename B, int N, typename T >
[[nodiscard]] std::string to_string( const Series< B, N, T >& f )
{
    std::ostringstream os;
    bool first = true;
    for ( int k = 0; k <= N; ++k )
    {
        const T ck = f[std::size_t( k )];
        if ( ck == T{ 0 } ) continue;
        if ( !first ) os << " + ";
        first = false;
        if ( k == 0 )
            os << ck;
        else
            os << ck << "*" << B::term( k );
    }
    if ( first ) os << T{ 0 };
    return os.str();
}

template < typename B, int N, typename T >
std::ostream& operator<<( std::ostream& os, const Series< B, N, T >& f )
{
    return os << to_string( f );
}

}  // namespace tax
