#pragma once

// Umbrella header — includes every layer of the DA library in dependency order.
//
// Lay-out of include/tax/:
//
//   fwd.hpp              — Scalar concept, MultiIndex
//   combinatorics.hpp    — binom, numMonomials, totalDegree, flatIndex
//   kernels.hpp          — array arithmetic, Cauchy product/accumulate, series kernels
//   leaf.hpp             — DALeaf tag, stored_t, is_leaf_v
//   expr/
//     base.hpp           — DAExpr<Derived,T,N,M> CRTP base
//     arithmetic_ops.hpp — OpAdd, OpSub, OpMul, OpDiv, OpScalar*, OpNeg
//     math_ops.hpp       — OpSquare, OpCube, OpSqrt, OpReciprocal
//     bin_expr.hpp       — BinExpr<L,R,Op>
//     unary_expr.hpp     — UnaryExpr<E,Op>
//     scalar_expr.hpp    — ScalarExpr<E,Op>
//     func_expr.hpp      — FuncExpr<E,Op>
//     sum_expr.hpp       — SumExpr<Es...>
//     product_expr.hpp   — ProductExpr<Es...>
//   da_type.hpp          — DA<T,N,M> leaf / materialised type
//   operators.hpp        — operator overloads, square/cube/sqrt free functions
//   aliases.hpp          — DAd, DAf, DAMd, DAMf

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
