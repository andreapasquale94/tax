// Eigen type aliases. Vec/Mat are dynamic-size doubles; the fixed-size aliases
// take the scalar type explicitly.

#pragma once

#include <Eigen/Core>

namespace tax::la
{

/// Dynamic-size column vector of `double` (Eigen::VectorXd).
using Vec = Eigen::VectorXd;

/// Dynamic-size dense matrix of `double` (Eigen::MatrixXd).
using Mat = Eigen::MatrixXd;

/// Fixed-size column vector with `N` rows of scalar `T`.
template < int N, class T >
using VecNT = Eigen::Matrix< T, N, 1 >;

/// Fixed-size square matrix `N x N` of scalar `T`.
template < int N, class T >
using MatNT = Eigen::Matrix< T, N, N >;

/// Fixed-size dense matrix `N x M` of scalar `T`.
template < int N, int M, class T >
using MatNMT = Eigen::Matrix< T, N, M >;

}  // namespace tax::la
