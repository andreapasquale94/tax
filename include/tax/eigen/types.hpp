#pragma once

#include <tax/eigen/num_traits.hpp>

namespace tax
{

template < typename T, int N, int M, int Rows, int Cols >
using Mat = Eigen::Matrix< TDA< T, N, M >, Rows, Cols >;

template < typename Scalar, int Size >
using VecT = Eigen::Matrix< Scalar, Size, 1 >;

template < typename Scalar, int Size >
using RowVecT = Eigen::Matrix< Scalar, 1, Size >;

template < int N, int Size >
using DAVec = VecT< DA< N >, Size >;

template < int N, int Size >
using DARowVec = RowVecT< DA< N >, Size >;

template < int N, int M >
using DAnVec = VecT< DAn< N, M >, M >;

template < int N, int M >
using DAnRowVec = RowVecT< DAn< N, M >, M >;

}  // namespace tax
