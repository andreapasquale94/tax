#pragma once

/**
 * @file
 * @brief Umbrella include for the full tax differential algebra API.
 * @details Includes forward declarations, combinatorics, kernels, expression
 * nodes, the materialized `TDA` type, operator overloads, and aliases.
 */

#include <tax/fwd.hpp>
#include <tax/combinatorics.hpp>
#include <tax/kernels.hpp>
#include <tax/leaf.hpp>
#include <tax/expr/base.hpp>
#include <tax/expr/arithmetic_ops.hpp>
#include <tax/expr/math_ops.hpp>
#include <tax/expr/bin_expr.hpp>
#include <tax/expr/unary_expr.hpp>
#include <tax/expr/scalar_expr.hpp>
#include <tax/expr/func_expr.hpp>
#include <tax/expr/sum_expr.hpp>
#include <tax/expr/product_expr.hpp>
#include <tax/da.hpp>
#include <tax/operators.hpp>
#include <tax/aliases.hpp>
