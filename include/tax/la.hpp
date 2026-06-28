// include/tax/la.hpp
//
// Linear-algebra umbrella header. Pulls in:
//
//   types       — Vec, Mat, VecNT<N,T>, MatNT<N,T>, MatNMT<N,M,T>.
//   expansion_vectors — TEVec<D,N,M>, NEVec<D,N,Axes...>, MTEVec<D,Axes...>.
//   num_traits  — Eigen::NumTraits<TaylorExpansion> + internal traits.
//   values      — variables, value, eval.
//   truncate    — free tax::truncate<N2>(scalar | Eigen vector/matrix).
//   derivatives — derivative, gradient, hessian, jacobian.
//   invert      — formal polynomial-map inversion (Picard iteration).
//
// Everything public lives in namespace `tax::la` (except tax::truncate).

#pragma once

#include <tax/la/derivatives.hpp>
#include <tax/la/expansion_vectors.hpp>
#include <tax/la/invert.hpp>
#include <tax/la/mixed_named.hpp>
#include <tax/la/named.hpp>
#include <tax/la/num_traits.hpp>
#include <tax/la/truncate.hpp>
#include <tax/la/types.hpp>
#include <tax/la/values.hpp>
