#pragma once

// Unary math surface for MixedExpansion: applies tax::FN to the inner
// expansion, rewrapping with the same axes. Mirrors expansion/ops/named_math_unary.hpp.

#include <cmath>
#include <tax/expansion/mixed.hpp>
#include <tax/expansion/ops/math_unary.hpp>

namespace tax::mixed
{

// Unary math functions (forwarded to the inner expansion, axis set preserved).

#define TAX_MIXED_UNARY_FN( FN )                                                              \
    template < typename T, typename... A >                                                    \
    [[nodiscard]] MixedExpansion< T, A... > FN( const MixedExpansion< T, A... >& a ) noexcept \
    {                                                                                         \
        return MixedExpansion< T, A... >{ tax::FN( a.inner() ) };                             \
    }

#define TAX_UNARY_CE( NAME, KERNEL ) TAX_MIXED_UNARY_FN( NAME )
#define TAX_UNARY_RT( NAME, KERNEL ) TAX_MIXED_UNARY_FN( NAME )
#include <tax/expansion/ops/unary_functions.def>
#undef TAX_UNARY_CE
#undef TAX_UNARY_RT

#undef TAX_MIXED_UNARY_FN

}  // namespace tax::mixed
