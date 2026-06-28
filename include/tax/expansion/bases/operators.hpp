#pragma once

#include <tax/expansion/bases/aliases.hpp>
#include <tax/expansion/bases/taylor_basis.hpp>
#include <tax/expansion/ops/arithmetic.hpp>
#include <type_traits>

namespace tax
{

// ===========================================================================
// Basis-generic operator surface.
//
// The linear-space operators and the bilinear product are basis-generic and
// live in <tax/expansion/ops/arithmetic.hpp> (one set of templates over `BasisPolicy B`,
// the product delegating to `B::product`). This header only adds the
// integer power, which a Taylor expansion already gets a recurrence-based form
// of via <tax/expansion/ops/math_binary.hpp>; here it is provided by repeated
// squaring for the other (non-Taylor) product-bearing families.
// ===========================================================================

template < typename B >
concept NonTaylorBasis = BasisPolicy< B > && !std::is_same_v< B, TaylorBasis >;

/// Integer power by repeated squaring (any non-Taylor product-bearing basis).
template < typename T, NonTaylorBasis B, typename Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > pow( const Expansion< T, B, Scheme >& x,
                                                       int n ) noexcept
{
    if ( n == 0 ) return Expansion< T, B, Scheme >::constant( T{ 1 } );
    Expansion< T, B, Scheme > base = x;
    Expansion< T, B, Scheme > acc = Expansion< T, B, Scheme >::constant( T{ 1 } );
    int e = n < 0 ? -n : n;
    while ( e > 0 )
    {
        if ( e & 1 ) acc = acc * base;
        e >>= 1;
        if ( e ) base = base * base;
    }
    if ( n < 0 )
    {
        if constexpr ( requires( Expansion< T, B, Scheme > z ) { T{ 1 } / z; } )
            return T{ 1 } / acc;
    }
    return acc;
}

}  // namespace tax
