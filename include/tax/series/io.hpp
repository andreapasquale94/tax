#pragma once

#include <ostream>
#include <sstream>
#include <string>
#include <tax/series/series.hpp>
#include <tax/series/taylor_basis.hpp>
#include <type_traits>

namespace tax
{

/// Human-readable univariate series in its own basis, e.g. "1 + 2*x + 3*x^2" or
/// "1 + 2*T_1 + 3*T_2". Zero coefficients are omitted; the zero series prints
/// as "0".
template < typename T, typename B, typename Scheme >
    requires( Scheme::isUnivariate && !std::is_same_v< B, TaylorBasis > )
[[nodiscard]] std::string to_string( const Expansion< T, B, Scheme >& f )
{
    std::ostringstream os;
    bool first = true;
    for ( int k = 0; k <= Scheme::order; ++k )
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

template < typename T, typename B, typename Scheme >
    requires( Scheme::isUnivariate && !std::is_same_v< B, TaylorBasis > )
std::ostream& operator<<( std::ostream& os, const Expansion< T, B, Scheme >& f )
{
    return os << to_string( f );
}

}  // namespace tax
