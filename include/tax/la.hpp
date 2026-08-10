// Linear-algebra umbrella header: Eigen integration for TaylorExpansion.
// Everything public lives in namespace `tax::la` (except tax::truncate).
//
// `tax/la/moments.hpp` is deliberately NOT included here: it pulls in Eigen's
// heavy `unsupported/Eigen/CXX11/Tensor` module, so it stays an opt-in header
// that consumers include explicitly when they need statistical moments.

#pragma once

#include <tax/la/derivatives.hpp>
#include <tax/la/expansion_vectors.hpp>
#include <tax/la/invert.hpp>
#include <tax/la/named.hpp>
#include <tax/la/norm.hpp>
#include <tax/la/num_traits.hpp>
#include <tax/la/truncate.hpp>
#include <tax/la/types.hpp>
#include <tax/la/values.hpp>
#include <tax/la/vector_ops.hpp>
