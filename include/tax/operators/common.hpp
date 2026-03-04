#pragma once

#include <tax/tte.hpp>
#include <type_traits>

namespace tax
{

/**
 * @brief Constraint requiring same order, variable count, and scalar type.
 * @details Disallows mixing TTE objects with incompatible truncation spaces.
 */
template < typename L, typename R >
concept CompatibleTTE = ( L::order == R::order ) && ( L::nvars == R::nvars ) &&
                       std::is_same_v< typename L::scalar_type, typename R::scalar_type >;

/**
 * @brief Helper alias for the TTE expression base type of a concrete expression `E`.
 */
template < typename E >
using ExprBase = Expr< E, typename E::scalar_type, E::order, E::nvars >;

}  // namespace tax
