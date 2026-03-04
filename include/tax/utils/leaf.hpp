#pragma once

#include <type_traits>

namespace tax
{

/// @brief Tag marking materialized DA objects used as expression leaves.
struct ExprLeaf
{
};

}  // namespace tax

namespace tax::detail
{

/**
 * @brief Storage policy for expression operands.
 * @details Leaf DA objects are stored by const reference, composed nodes by value.
 */
template < typename E >
using stored_t = std::conditional_t< std::is_base_of_v< tax::ExprLeaf, std::remove_cvref_t< E > >,
                                     const std::remove_cvref_t< E >&, std::remove_cvref_t< E > >;

template < typename E >
/// @brief `true` when `E` is a materialized DA leaf type.
static constexpr bool is_leaf_v = std::is_base_of_v< tax::ExprLeaf, std::remove_cvref_t< E > >;

}  // namespace tax::detail
