// ---------------------------------------------------------------------------
// ops_pow.cpp — power functions (integer, real, TE^TE).
// ---------------------------------------------------------------------------

#include "operations.hpp"

namespace tax::py
{

PyTE powInt( const PyTE& a, int p )
{
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = tax::ipow( ta, p );
        return fromTE< N, M >( r );
    } );
}

PyTE powDouble( const PyTE& a, double p )
{
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = tax::dpow( ta, p );
        return fromTE< N, M >( r );
    } );
}

PyTE powTE( const PyTE& a, const PyTE& b )
{
    checkCompatible( a, b, "pow" );
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        auto tb = toTE< N, M >( b );
        tax::TEn< N, M > r = tax::tpow( ta, tb );
        return fromTE< N, M >( r );
    } );
}

}  // namespace tax::py
