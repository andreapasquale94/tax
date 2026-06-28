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

#define TAX_UNARY_CE( NAME, KERNEL ) TAX_NAMED_UNARY_FN( NAME )
#define TAX_UNARY_RT( NAME, KERNEL ) TAX_NAMED_UNARY_FN( NAME )
#include <tax/expansion/ops/unary_functions.def>
#undef TAX_UNARY_CE
#undef TAX_UNARY_RT

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

#define TAX_UNARY_CE( NAME, KERNEL )  // square/cube/reciprocal: no std analogue
#define TAX_UNARY_RT( NAME, KERNEL ) TAX_REEXPORT_UNARY( NAME )
#include <tax/expansion/ops/unary_functions.def>
#undef TAX_UNARY_CE
#undef TAX_UNARY_RT

#undef TAX_REEXPORT_UNARY

// square / cube / reciprocal have no `std` (scalar) analogue: re-export the
// named overloads only. The TaylorExpansion overloads are already in `tax`.
using named::cube;
using named::reciprocal;
using named::square;

}  // namespace tax
