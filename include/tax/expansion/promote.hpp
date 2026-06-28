// promote_t — the common Taylor-expansion type a set of operands promote into.
//
// For named / mixed expansions this is the expansion over the *union* of their
// axes (exactly the result type the arithmetic operators produce); a narrower
// expansion promotes implicitly into it. Plain scalars promote into the
// accompanying expansion. Useful for declaring a homogeneous container (e.g. an
// Eigen vector) whose element type must hold the result of mixing operands that
// live over different axis sets.

#pragma once

#include <type_traits>
#include <utility>

namespace tax
{

/// The common expansion type that every `Ts...` promotes into — i.e. the type
/// of `t0 + t1 + ...`. Mirrors the union-of-axes semantics of the operators, so
/// `promote_t< NE< N, A >, NE< N, B > >` is `NE< N, A, B >` (sorted/unique), and
/// `promote_t< NE< N, A >, double >` is `NE< N, A >`. A single argument yields
/// itself.
template < typename... Ts >
using promote_t = std::remove_cvref_t< decltype( ( ... + std::declval< Ts >() ) ) >;

}  // namespace tax
