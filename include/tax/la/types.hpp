#pragma once

#include <Eigen/Core>

namespace tax::la
{

/// @brief Fixed-size matrix with N rows and M columns.
template < typename T, int N, int M >
using MatNMT = Eigen::Matrix< T, N, M >;

/// @brief Fixed-size square matrix of dimension N.
template < typename T, int N >
using MatNT = Eigen::Matrix< T, N, N >;

/// @brief Fixed-size column vector of dimension N.
template < typename T, int N >
using VecNT = Eigen::Vector< T, N >;

/// @brief Fixed-size row vector of dimension N.
template < typename T, int N >
using RowVecNT = Eigen::RowVector< T, N >;

// -- Convenience double aliases -----------------------------------------------

/// @brief Fixed-size double matrix with N rows and M columns.
template < int N, int M >
using MatNM = MatNMT< double, N, M >;

/// @brief Fixed-size double square matrix of dimension N.
template < int N >
using MatN = MatNT< double, N >;

/// @brief Fixed-size double column vector of dimension N.
template < int N >
using VecN = VecNT< double, N >;

/// @brief Fixed-size double row vector of dimension N.
template < int N >
using RowVecN = RowVecNT< double, N >;

}  // namespace tax::la
