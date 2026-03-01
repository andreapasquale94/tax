#pragma once

#include <tax/eigen/num_traits.hpp>

namespace tax
{

template < typename T, int N, int M, int Rows, int Cols >
using Mat = Eigen::Matrix< TDA< T, N, M >, Rows, Cols >;

template < int N, int M >
using Vec = Eigen::Matrix< DAn< N, M >, M, 1 >;

template < int N, int M >
using RowVec = Eigen::Matrix< DAn< N, M >, 1, M >;

}  // namespace tax
