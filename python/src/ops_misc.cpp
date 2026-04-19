// ---------------------------------------------------------------------------
// ops_misc.cpp — exp/log, sqrt/cbrt, square/cube, abs, erf, hypot.
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

TAX_PY_UNARY( exp )
TAX_PY_UNARY( log )
TAX_PY_UNARY( log10 )
TAX_PY_UNARY( sqrt )
TAX_PY_UNARY( cbrt )
TAX_PY_UNARY( square )
TAX_PY_UNARY( cube )
TAX_PY_UNARY( abs )
TAX_PY_UNARY( erf )

#undef TAX_PY_UNARY

PyTE hypot2( const PyTE& x, const PyTE& y )
{
    checkCompatible( x, y, "hypot" );
    return dispatchNM( x.order, x.nvars, [&]< int N, int M >() {
        auto tx = toTE< N, M >( x );
        auto ty = toTE< N, M >( y );
        tax::TEn< N, M > r = tax::hypot( tx, ty );
        return fromTE< N, M >( r );
    } );
}

}  // namespace tax::py
