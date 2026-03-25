#pragma once

#include <array>
#include <tax/la/types.hpp>
#include <tax/utils/combinatorics.hpp>

namespace tax::detail
{

/// @brief Coefficient storage array for a truncated polynomial of order N in M variables.
template < typename T, int N, int M >
using CoeffArray = std::array< T, numMonomials( N, M ) >;

/// @brief Univariate basis-to-monomial (or inverse) transformation matrix of size (N+1)x(N+1).
template < typename T, int N >
using TransformMatrix = la::MatNT< T, N + 1 >;

/// @brief Augmented matrix for Gauss-Jordan elimination: (N+1) x 2*(N+1).
template < typename T, int N >
using AugmentedMatrix = la::MatNMT< T, N + 1, 2 * ( N + 1 ) >;

}  // namespace tax::detail
