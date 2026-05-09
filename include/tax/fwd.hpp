// SPDX-License-Identifier: BSD-3-Clause
//
// Forward declarations for tax public types.

#pragma once

#include <concepts>
#include <cstddef>

namespace tax
{

template < class T >
concept Scalar = std::floating_point< T >;

template < class T, std::size_t Order, std::size_t Vars >
class TruncatedTaylorExpansionT;

template < class T >
class DynamicTaylorExpansion;

template < std::size_t Order >
using TE = TruncatedTaylorExpansionT< double, Order, 1 >;

template < std::size_t Order, std::size_t Vars >
using TEn = TruncatedTaylorExpansionT< double, Order, Vars >;

template < class T = double >
using DynTE = DynamicTaylorExpansion< T >;

}  // namespace tax
