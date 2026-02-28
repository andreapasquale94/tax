#pragma once

#include <type_traits>

namespace da {

struct DALeaf {};

} // namespace da

namespace da::detail {

// Leaf → const&  (zero-copy ref to named DA variable).
// ET node → by value  (prevents dangling refs to temporaries).
template <typename E>
using stored_t = std::conditional_t<
    std::is_base_of_v<da::DALeaf, std::remove_cvref_t<E>>,
    const std::remove_cvref_t<E>&,
    std::remove_cvref_t<E>>;

template <typename E>
static constexpr bool is_leaf_v = std::is_base_of_v<da::DALeaf, std::remove_cvref_t<E>>;

} // namespace da::detail
