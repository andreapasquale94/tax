// ---------------------------------------------------------------------------
// ops_scalar.cpp — TE ↔ scalar arithmetic.
// ---------------------------------------------------------------------------

#include "operations.hpp"

namespace tax::py
{

PyTE addScalar( const PyTE& a, double s )
{
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = ta + s;
        return fromTE< N, M >( r );
    } );
}

PyTE subScalar( const PyTE& a, double s )
{
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = ta - s;
        return fromTE< N, M >( r );
    } );
}

PyTE mulScalar( const PyTE& a, double s )
{
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = ta * s;
        return fromTE< N, M >( r );
    } );
}

PyTE divScalar( const PyTE& a, double s )
{
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = ta / s;
        return fromTE< N, M >( r );
    } );
}

PyTE scalarSub( double s, const PyTE& a )
{
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = s - ta;
        return fromTE< N, M >( r );
    } );
}

PyTE scalarDiv( double s, const PyTE& a )
{
    return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {
        auto ta = toTE< N, M >( a );
        tax::TEn< N, M > r = s / ta;
        return fromTE< N, M >( r );
    } );
}

}  // namespace tax::py
