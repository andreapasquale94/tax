#pragma once

#include <array>
#include <concepts>
#include <cstddef>

namespace da {

template <typename T>
concept Scalar = std::floating_point<T>;

template <int M>
using MultiIndex = std::array<int, M>;

} // namespace da
