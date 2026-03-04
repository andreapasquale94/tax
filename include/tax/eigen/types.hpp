#pragma once

#include <tax/eigen/num_traits.hpp>

namespace tax
{

template < typename T, int N, int M, int Rows, int Cols >
using Mat = Eigen::Matrix< TruncatedTaylorExpansionT< T, N, M >, Rows, Cols >;

template < typename Scalar, int Size >
using VecT = Eigen::Matrix< Scalar, Size, 1 >;

template < typename Scalar, int Size >
using RowVecT = Eigen::Matrix< Scalar, 1, Size >;

template < int N, int Size >
using TEVec = VecT< TE< N >, Size >;

template < int N, int Size >
using TERowVec = RowVecT< TE< N >, Size >;

template < int N, int M >
using TEnVec = VecT< TEn< N, M >, M >;

template < int N, int M >
using TEnRowVec = RowVecT< TEn< N, M >, M >;

}  // namespace tax
