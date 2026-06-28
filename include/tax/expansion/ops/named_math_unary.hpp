#pragma once

// Unary math surface for NamedExpansion (any basis). Each wrapper applies the
// corresponding `tax::` math function to the inner anonymous expansion and
// rewraps the result with the same axis list, so transcendental functions of
// named expansions keep their named structure. Mirrors expansion/ops/math_unary.hpp
// and is basis-generic like expansion/ops/named_arithmetic.hpp.
//
// The inner call is unqualified after a `using tax::FN;`: ordinary lookup picks
// up the Taylor overloads visible here, and ADL augments the set at the point of
// instantiation with the inner basis' own math (e.g. the Chebyshev overloads in
// bases/chebyshev_math.hpp, which the umbrella includes after this header). A
// plain `tax::FN(...)` would suppress that ADL and only see TaylorBasis.

#include <cmath>
#include <tax/expansion/named.hpp>
#include <tax/expansion/ops/math_unary.hpp>

namespace tax::named
{

#define TAX_NAMED_UNARY_FN( FN )                                       \
    template < typename T, typename Basis, int N, typename... A >      \
    [[nodiscard]] NamedExpansion< T, Basis, N, A... > FN(              \
        const NamedExpansion< T, Basis, N, A... >& a ) noexcept        \
    {                                                                  \
        using tax::FN;                                                 \
        return NamedExpansion< T, Basis, N, A... >{ FN( a.inner() ) }; \
    }

TAX_NAMED_UNARY_FN( square )
TAX_NAMED_UNARY_FN( cube )
TAX_NAMED_UNARY_FN( sqrt )
TAX_NAMED_UNARY_FN( cbrt )
TAX_NAMED_UNARY_FN( reciprocal )
TAX_NAMED_UNARY_FN( exp )
TAX_NAMED_UNARY_FN( log )
TAX_NAMED_UNARY_FN( sin )
TAX_NAMED_UNARY_FN( cos )
TAX_NAMED_UNARY_FN( tan )
TAX_NAMED_UNARY_FN( asin )
TAX_NAMED_UNARY_FN( acos )
TAX_NAMED_UNARY_FN( atan )
TAX_NAMED_UNARY_FN( sinh )
TAX_NAMED_UNARY_FN( cosh )
TAX_NAMED_UNARY_FN( tanh )
TAX_NAMED_UNARY_FN( asinh )
TAX_NAMED_UNARY_FN( acosh )
TAX_NAMED_UNARY_FN( atanh )
TAX_NAMED_UNARY_FN( erf )

#undef TAX_NAMED_UNARY_FN

}  // namespace tax::named

// ---------------------------------------------------------------------------
// Re-exports: make a *qualified* `tax::fn(...)` resolve for every supported
// argument type. A qualified call suppresses argument-dependent lookup, so
// without these the named-expansion overloads (in `tax::named`) and the scalar
// overloads (in `std`) are invisible to `tax::fn`. The dense / sparse
// TaylorExpansion overloads already live directly in `tax`.
//   using named::fn -> NamedTaylorExpansion
//   using std::fn   -> float / double / long double + the integral overloads
// ---------------------------------------------------------------------------

namespace tax
{

#define TAX_REEXPORT_UNARY( FN ) \
    using named::FN;             \
    using std::FN;

TAX_REEXPORT_UNARY( sqrt )
TAX_REEXPORT_UNARY( cbrt )
TAX_REEXPORT_UNARY( exp )
TAX_REEXPORT_UNARY( log )
TAX_REEXPORT_UNARY( sin )
TAX_REEXPORT_UNARY( cos )
TAX_REEXPORT_UNARY( tan )
TAX_REEXPORT_UNARY( asin )
TAX_REEXPORT_UNARY( acos )
TAX_REEXPORT_UNARY( atan )
TAX_REEXPORT_UNARY( sinh )
TAX_REEXPORT_UNARY( cosh )
TAX_REEXPORT_UNARY( tanh )
TAX_REEXPORT_UNARY( asinh )
TAX_REEXPORT_UNARY( acosh )
TAX_REEXPORT_UNARY( atanh )
TAX_REEXPORT_UNARY( erf )

#undef TAX_REEXPORT_UNARY

// square / cube / reciprocal have no `std` (scalar) analogue: re-export the
// named overloads only. The TaylorExpansion overloads are already in `tax`.
using named::cube;
using named::reciprocal;
using named::square;

}  // namespace tax
