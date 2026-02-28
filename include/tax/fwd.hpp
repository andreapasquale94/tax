#pragma once

#include <array>
#include <concepts>
#include <cstddef>

namespace tax {

/// @brief Scalar constraint used for DA coefficients and function values.
template <typename T>
concept Scalar = std::floating_point<T>;

/**
 * @brief Exponent vector `(a_0, ..., a_{M-1})` for multivariate monomials.
 * @tparam M Number of variables.
 */
template <int M>
using MultiIndex = std::array<int, M>;

} // namespace tax
