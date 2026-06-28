// Fixed-size Eigen column vectors of Taylor-expansion scalars. Each alias bakes
// the expansion construction into a `VecNT< D, Scalar >`: `D` is the vector
// length, the remaining parameters forward to the underlying expansion alias
// (TE / NE / MTE). Defined after the expansion types so it can name them.

#pragma once

#include <tax/expansion/mixed_named.hpp>
#include <tax/expansion/named.hpp>
#include <tax/expansion/expansion.hpp>
#include <tax/la/types.hpp>

namespace tax::la
{

/// `D`-row column vector of dense `TE< N, M >` expansions.
template < int D, int N, int M = 1 >
using TEVec = VecNT< D, TE< N, M > >;

/// `D`-row column vector of named `NE< N, Axes... >` expansions.
template < int D, int N, typename... Axes >
using NEVec = VecNT< D, NE< N, Axes... > >;

/// `D`-row column vector of mixed-order `MTE< Axes... >` expansions.
template < int D, typename... Axes >
using MTEVec = VecNT< D, MTE< Axes... > >;

}  // namespace tax::la
