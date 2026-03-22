#pragma once

/**
 * @file
 * @brief Umbrella include for the full tax differential algebra API.
 * @details Includes utilities, kernels, expression nodes, the materialized
 * `TruncatedTaylorExpansionT` type, and operator overloads.
 */

#include <tax/tte.hpp>
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
#include <tax/eigen/derivative.hpp>
#include <tax/eigen/eval.hpp>
#include <tax/eigen/invert_map.hpp>
#include <tax/eigen/types.hpp>
#include <tax/eigen/value.hpp>
#include <tax/eigen/variables.hpp>
#include <tax/ode/taylor_integrator.hpp>
