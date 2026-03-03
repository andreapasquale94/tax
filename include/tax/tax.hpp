#pragma once

/**
 * @file
 * @brief Umbrella include for the full tax differential algebra API.
 * @details Includes utilities, kernels, expression nodes, the materialized
 * `TDA` type, and operator overloads.
 */

#include <tax/da.hpp>
#include <tax/expr/arithmetic_ops.hpp>
#include <tax/expr/base.hpp>
#include <tax/expr/bin_expr.hpp>
#include <tax/expr/func_expr.hpp>
#include <tax/expr/math_ops.hpp>
#include <tax/expr/product_expr.hpp>
#include <tax/expr/scalar_expr.hpp>
#include <tax/expr/sum_expr.hpp>
#include <tax/expr/unary_expr.hpp>
#include <tax/kernels.hpp>
#include <tax/operators.hpp>
#include <tax/utils.hpp>

// Auto-detect Eigen if not explicitly configured via CMake
#if !defined( TAX_ENABLE_EIGEN )
#if __has_include( <Eigen/Core> )
#define TAX_ENABLE_EIGEN 1
#else
#define TAX_ENABLE_EIGEN 0
#endif
#endif

#if TAX_ENABLE_EIGEN
#include <tax/eigen/adapters.hpp>
#include <tax/eigen/map_inv.hpp>
#include <tax/eigen/tensor_function.hpp>
#include <tax/eigen/tensors.hpp>
#include <tax/eigen/types.hpp>
#endif
