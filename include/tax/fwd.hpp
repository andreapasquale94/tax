// SPDX-License-Identifier: BSD-3-Clause
//
// Forward declarations for tax public types.

#pragma once

#include <Eigen/Core>
#include <concepts>
#include <cstddef>

namespace tax
{

template < class T >
concept Scalar = std::floating_point< T >;

// Unified storage template.  Order and Vars are signed ints so that
// `Eigen::Dynamic` (= -1) can stand in as the runtime-size sentinel,
// mirroring `Eigen::Matrix<T, Rows, Cols>`.  Both must be either both
// non-negative (compile-time-fixed) or both equal to `Eigen::Dynamic`
// (runtime-fixed at construction); mixed dynamism is rejected with a
// static_assert inside the class body.
template < class T, int Order, int Vars >
class TaylorExpansionT;

template < int Order >
using TE = TaylorExpansionT< double, Order, 1 >;

template < int Order, int Vars >
using TEn = TaylorExpansionT< double, Order, Vars >;

template < class T = double >
using DynTE = TaylorExpansionT< T, Eigen::Dynamic, Eigen::Dynamic >;

}  // namespace tax
