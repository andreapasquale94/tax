// ---------------------------------------------------------------------------
// ops_hyper.cpp — hyperbolic functions.
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

TAX_PY_UNARY( sinh )
TAX_PY_UNARY( cosh )
TAX_PY_UNARY( tanh )
TAX_PY_UNARY( asinh )
TAX_PY_UNARY( acosh )
TAX_PY_UNARY( atanh )

#undef TAX_PY_UNARY

}  // namespace tax::py
