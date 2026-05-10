// SPDX-License-Identifier: BSD-3-Clause
//
// Concepts shared by both storage paths and the streaming expression layer.

#pragma once

#include <concepts>
#include <cstddef>
#include <span>

#include "tax/fwd.hpp"

namespace tax
{

// Anything that holds a complete (Order, Vars) jet of T-coefficients.  The
// unified `TaylorExpansionT` template (in either of its IsStatic / dynamic
// configurations) models this.
template < class E >
concept TaylorExpansion = requires( const E& e, std::span< const std::size_t > alpha ) {
    typename E::Scalar;
    { e.order() } -> std::convertible_to< std::size_t >;
    { e.nvars() } -> std::convertible_to< std::size_t >;
    { e.value() } -> std::convertible_to< typename E::Scalar >;
    { e.coeff( alpha ) } -> std::convertible_to< typename E::Scalar >;
    { e.derivative( alpha ) } -> std::convertible_to< typename E::Scalar >;
};

// All ET nodes (and the storage types) advance their coefficient buffer
// degree-by-degree.  The driver loop calls `advanceTo(d)` for d = 0, 1, ...
// and then reads `slice(d)`.
template < class E >
concept StreamingExpression = requires( E& e, std::size_t d ) {
    typename E::Scalar;
    { e.order() } -> std::convertible_to< std::size_t >;
    { e.nvars() } -> std::convertible_to< std::size_t >;
    e.advanceTo( d );
    e.slice( d );
};

// Tag dispatching: distinguishing static vs. dynamic expressions so we can
// reject mixed expressions at compile time.
template < class E >
struct expr_traits
{
    static constexpr bool is_static = false;
    static constexpr bool is_dynamic = false;
};

}  // namespace tax
