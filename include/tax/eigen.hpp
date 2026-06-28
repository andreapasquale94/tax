// include/tax/eigen.hpp
//
// Linear-algebra umbrella header. Pulls in:
//
//   types       — Vec, Mat, VecNT<N,T>, MatNT<N,T>, MatNMT<N,M,T>.
//   expansion_vectors — TEVec<D,N,M>, NEVec<D,N,Axes...>, MTEVec<D,Axes...>.
//   num_traits  — Eigen::NumTraits<Expansion> + internal traits.
//   values      — variables, value, eval.
//   truncate    — free tax::truncate<N2>(scalar | Eigen vector/matrix).
//   derivatives — derivative, gradient, hessian, jacobian.
//   named       — NumTraits + per-axis gradient/hessian/jacobian/value/eval for NamedExpansion.
//   mixed_named — the same for MixedTaylorExpansion (so la.hpp alone is self-complete).
//   invert      — formal polynomial-map inversion (Picard iteration).
//   exports     — single assembly point surfacing the la + named helpers under `tax::` (last).
//
// Helpers are defined in `tax::la` / `tax::named` and surfaced under `tax::` via
// exports.hpp, so `tax::gradient(...)` / `tax::la::gradient(...)` both resolve.

#pragma once

#include <tax/eigen/derivatives.hpp>
#include <tax/eigen/expansion_vectors.hpp>
#include <tax/eigen/invert.hpp>
#include <tax/eigen/mixed_named.hpp>
#include <tax/eigen/named.hpp>
#include <tax/eigen/num_traits.hpp>
#include <tax/eigen/truncate.hpp>
#include <tax/eigen/types.hpp>
#include <tax/eigen/values.hpp>
#include <tax/eigen/exports.hpp>
