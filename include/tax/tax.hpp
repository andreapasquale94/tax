// SPDX-License-Identifier: BSD-3-Clause
//
// Umbrella header for tax: a header-only C++23 library for truncated
// Taylor expansions in M variables of order N (a.k.a. multivariate
// Differential Algebra).  Including this single header pulls in:
//
//   - storage types (static + dynamic)
//   - the streaming expression-template machinery
//   - all user-facing operators and math functions
//
// All public symbols live in `namespace tax`.

#pragma once

#include "tax/concepts.hpp"
#include "tax/fwd.hpp"

// View-like + buffered ET nodes
#include "tax/expr/base.hpp"
#include "tax/expr/buffered_nodes.hpp"
#include "tax/expr/view_nodes.hpp"

// Storage type (also defines Expr<Derived>::eval() out-of-line)
#include "tax/storage/tte.hpp"

// Operators
#include "tax/ops/arithmetic.hpp"
#include "tax/ops/math.hpp"
