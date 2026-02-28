#pragma once

#include <Eigen/Core>
#include <tax/da.hpp>
#include <tuple>
#include <utility>

namespace tax
{

template < typename T, int N, int M, int Rows, int Cols >
    requires( M > 1 )
using EigenDAMatrix = Eigen::Matrix< TDA< T, N, M >, Rows, Cols >;

template < int N, int M >
    requires( M > 1 )
using EigenDAVector = Eigen::Matrix< DAn< N, M >, M, 1 >;

template < int N, int M >
    requires( M > 1 )
using EigenDARowVector = Eigen::Matrix< DAn< N, M >, 1, M >;

}  // namespace tax
