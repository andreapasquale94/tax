#pragma once

// Unary math surface for MixedTaylorExpansion: applies tax::FN to the inner
// expansion, rewrapping with the same axes. Mirrors operators/named_math_unary.hpp.

#include <cmath>
#include <tax/core/mixed_named.hpp>
#include <tax/operators/math_unary.hpp>

namespace tax::named
{

// Unary math functions (forwarded to the inner expansion, axis set preserved).

#define TAX_MIXED_UNARY_FN( FN )                                        \
    template < typename T, typename... A >                              \
    [[nodiscard]] MixedTaylorExpansion< T, A... > FN(                   \
        const MixedTaylorExpansion< T, A... >& a ) noexcept             \
    {                                                                   \
        return MixedTaylorExpansion< T, A... >{ tax::FN( a.inner() ) }; \
    }

TAX_MIXED_UNARY_FN( square )
TAX_MIXED_UNARY_FN( cube )
TAX_MIXED_UNARY_FN( sqrt )
TAX_MIXED_UNARY_FN( cbrt )
TAX_MIXED_UNARY_FN( reciprocal )
TAX_MIXED_UNARY_FN( exp )
TAX_MIXED_UNARY_FN( log )
TAX_MIXED_UNARY_FN( sin )
TAX_MIXED_UNARY_FN( cos )
TAX_MIXED_UNARY_FN( tan )
TAX_MIXED_UNARY_FN( asin )
TAX_MIXED_UNARY_FN( acos )
TAX_MIXED_UNARY_FN( atan )
TAX_MIXED_UNARY_FN( sinh )
TAX_MIXED_UNARY_FN( cosh )
TAX_MIXED_UNARY_FN( tanh )
TAX_MIXED_UNARY_FN( asinh )
TAX_MIXED_UNARY_FN( acosh )
TAX_MIXED_UNARY_FN( atanh )
TAX_MIXED_UNARY_FN( erf )

#undef TAX_MIXED_UNARY_FN

}  // namespace tax::named
