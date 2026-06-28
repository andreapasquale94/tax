// include/tax/eigen/exports.hpp
//
// Single assembly point: surface the linear-algebra helpers under `tax::` so
// the documented `tax::FN(...)` spelling resolves for dense, named, and mixed
// expansions uniformly. Included LAST by <tax/eigen.hpp>, after every overload is
// defined, so one set of using-declarations captures the complete overload set
// (no need to re-issue per header). `tax::value` template overloads for scalar
// TE and arithmetic types live in la/values.hpp and stay there; the `value`
// using-declarations here fold in the Eigen-matrix overloads.

#pragma once

#include <tax/eigen/derivatives.hpp>
#include <tax/eigen/invert.hpp>
#include <tax/eigen/mixed_named.hpp>
#include <tax/eigen/named.hpp>
#include <tax/eigen/values.hpp>

namespace tax
{
// Dense / basis-generic la helpers (previously NOT surfaced to tax::).
using la::derivative;
using la::eval;
using la::gradient;
using la::hessian;
using la::invert;
using la::jacobian;
using la::value;
using la::variables;

// Named + mixed per-axis helpers (mixed overloads live in tax::named too).
using named::eval;
using named::gradient;
using named::hessian;
using named::jacobian;
using named::value;
using named::variables;
}  // namespace tax
