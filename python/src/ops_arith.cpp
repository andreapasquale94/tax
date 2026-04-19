// ---------------------------------------------------------------------------
// ops_arith.cpp — factories and TE-vs-TE arithmetic.
// ---------------------------------------------------------------------------

#include "operations.hpp"

namespace tax::py
{

PyTE makeConstant( int n, int m, double value )
{
    return dispatchNM( n, m, [&]< int N, int M >() {
        auto te = tax::TEn< N, M >::constant( value );
        return fromTE< N, M >( te );
    } );
}

PyTE makeVariable( int n, int m, int i, double x0 )
{
    if ( i < 0 || i >= m ) throw std::out_of_range( "tax: variable index out of range" );
    return dispatchNM( n, m, [&]< int N, int M >() {
        tax::TEn< N, M > te;
        te.coeffs()[0] = x0;
        if constexpr ( N >= 1 )
        {
            MultiIndex< M > ei{};
            ei[static_cast< std::size_t >( i )] = 1;
            te.coeffs()[tax::detail::flatIndex< M >( ei )] = 1.0;
        }
        return fromTE< N, M >( te );
    } );
}

std::vector< PyTE > makeVariables( int n, int m, const std::vector< double >& x0 )
{
    if ( static_cast< int >( x0.size() ) != m )
        throw std::invalid_argument( "tax: expansion point size must equal M" );
    std::vector< PyTE > out;
    out.reserve( static_cast< std::size_t >( m ) );
    for ( int i = 0; i < m; ++i )
        out.push_back( makeVariable( n, m, i, x0[static_cast< std::size_t >( i )] ) );
    return out;
}

PyTE add( const PyTE& a, const PyTE& b )
{
    checkCompatible( a, b, "+" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        auto tb = toTE< N, M >( b );
        tax::TEn< N, M > r = ta + tb;
        return fromTE< N, M >( r );
    } );
}

PyTE sub( const PyTE& a, const PyTE& b )
{
    checkCompatible( a, b, "-" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        auto tb = toTE< N, M >( b );
        tax::TEn< N, M > r = ta - tb;
        return fromTE< N, M >( r );
    } );
}

PyTE mul( const PyTE& a, const PyTE& b )
{
    checkCompatible( a, b, "*" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        auto tb = toTE< N, M >( b );
        tax::TEn< N, M > r = ta * tb;
        return fromTE< N, M >( r );
    } );
}

PyTE div( const PyTE& a, const PyTE& b )
{
    checkCompatible( a, b, "/" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        auto tb = toTE< N, M >( b );
        tax::TEn< N, M > r = ta / tb;
        return fromTE< N, M >( r );
    } );
}

PyTE neg( const PyTE& a )
{
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = -ta;
        return fromTE< N, M >( r );
    } );
}

}  // namespace tax::py
