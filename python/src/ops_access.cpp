// ---------------------------------------------------------------------------
// ops_access.cpp — evaluation, differentiation, coefficient access.
// ---------------------------------------------------------------------------

#include <string>
#include "operations.hpp"

namespace tax::py
{

double evalAt( const PyTE& a, const std::vector< double >& dx )
{
    if ( static_cast< int >( dx.size() ) != a.nvars )
        throw std::invalid_argument( "tax: eval point size must equal M" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        typename tax::TEn< N, M >::Input p{};
        for ( std::size_t i = 0; i < p.size(); ++i ) p[i] = dx[i];
        return ta.eval( p );
    } );
}

PyTE derivVar( const PyTE& a, int var )
{
    if ( var < 0 || var >= a.nvars )
        throw std::out_of_range( "tax: deriv variable out of range" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = ta.deriv( var );
        return fromTE< N, M >( r );
    } );
}

PyTE integVar( const PyTE& a, int var )
{
    if ( var < 0 || var >= a.nvars )
        throw std::out_of_range( "tax: integ variable out of range" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = ta.integ( var );
        return fromTE< N, M >( r );
    } );
}

double derivativeAt( const PyTE& a, const std::vector< int >& alpha )
{
    if ( static_cast< int >( alpha.size() ) != a.nvars )
        throw std::invalid_argument( "tax: derivative multi-index size must equal M" );
    int total = 0;
    for ( int v : alpha )
    {
        if ( v < 0 ) throw std::invalid_argument( "tax: derivative order must be non-negative" );
        total += v;
    }
    if ( total > a.order )
        throw std::invalid_argument( "tax: derivative total order exceeds truncation order" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        MultiIndex< M > mi{};
        for ( std::size_t i = 0; i < mi.size(); ++i ) mi[i] = alpha[i];
        return ta.derivative( mi );
    } );
}

double coeffAt( const PyTE& a, const std::vector< int >& alpha )
{
    if ( static_cast< int >( alpha.size() ) != a.nvars )
        throw std::invalid_argument( "tax: coefficient multi-index size must equal M" );
    int total = 0;
    for ( int v : alpha )
    {
        if ( v < 0 ) throw std::invalid_argument( "tax: exponent must be non-negative" );
        total += v;
    }
    if ( total > a.order )
        throw std::invalid_argument( "tax: total exponent exceeds truncation order" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        MultiIndex< M > mi{};
        for ( std::size_t i = 0; i < mi.size(); ++i ) mi[i] = alpha[i];
        return ta.coeff( mi );
    } );
}

std::vector< int > unflat( const PyTE& a, std::size_t k )
{
    if ( k >= a.coeffs.size() ) throw std::out_of_range( "tax: flat index out of range" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        MultiIndex< M > mi = tax::detail::unflatIndex< M >( k );
        std::vector< int > out( M );
        for ( std::size_t i = 0; i < mi.size(); ++i ) out[i] = mi[i];
        return out;
    } );
}

std::string repr( const PyTE& a )
{
    return "TE(N=" + std::to_string( a.order ) + ", M=" + std::to_string( a.nvars ) +
           ", value=" + std::to_string( a.value() ) + ")";
}

}  // namespace tax::py
