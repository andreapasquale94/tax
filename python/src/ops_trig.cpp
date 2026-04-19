// ---------------------------------------------------------------------------
// ops_trig.cpp — trigonometric functions.
// ---------------------------------------------------------------------------

#include "operations.hpp"

namespace tax::py
{

#define TAX_PY_UNARY( NAME )                                                                       \
    PyTE NAME( const PyTE& a )                                                                     \
    {                                                                                              \
        return dispatchNM( a.order, a.nvars, [&]< int N, int M >() {                               \
            auto ta = toTE< N, M >( a );                                                           \
            tax::TEn< N, M > r = tax::NAME( ta );                                                  \
            return fromTE< N, M >( r );                                                            \
        } );                                                                                       \
    }

TAX_PY_UNARY( sin )
TAX_PY_UNARY( cos )
TAX_PY_UNARY( tan )
TAX_PY_UNARY( asin )
TAX_PY_UNARY( acos )
TAX_PY_UNARY( atan )

#undef TAX_PY_UNARY

PyTE atan2( const PyTE& y, const PyTE& x )
{
    checkCompatible( y, x, "atan2" );
    return dispatchNM( y.order, y.nvars, [&]< int N, int M >() {
        auto ty = toTE< N, M >( y );
        auto tx = toTE< N, M >( x );
        tax::TEn< N, M > r = tax::atan2( ty, tx );
        return fromTE< N, M >( r );
    } );
}

}  // namespace tax::py
