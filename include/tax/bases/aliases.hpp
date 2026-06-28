#pragma once

// The basis-generic carrier `tax::Expansion< T, Basis, Scheme, Storage >` now
// lives in <tax/expansion/expansion.hpp> (with `TaylorExpansion` as its
// TaylorBasis alias). This header just adds the basis-named series aliases.

#include <tax/bases/chebyshev_basis.hpp>
#include <tax/bases/hermite_basis.hpp>
#include <tax/bases/legendre_basis.hpp>
#include <tax/expansion/basis.hpp>
#include <tax/expansion/expansion.hpp>
#include <tax/expansion/scheme/isotropic.hpp>
#include <tax/expansion/taylor_basis.hpp>

namespace tax
{

/// Basis-generic expansion (univariate by default; set M for multivariate).
template < typename Basis, int N, int M = 1, typename T = double >
using Series = Expansion< T, Basis, IsotropicScheme< N, M > >;

/// Order-N, M-variate Chebyshev (first-kind) expansion.
template < int N, int M = 1, typename T = double >
using ChebyshevExpansion = Expansion< T, ChebyshevBasis, IsotropicScheme< N, M > >;

/// Order-N, M-variate Legendre expansion.
template < int N, int M = 1, typename T = double >
using LegendreExpansion = Expansion< T, LegendreBasis, IsotropicScheme< N, M > >;

/// Order-N, M-variate probabilists' Hermite expansion.
template < int N, int M = 1, typename T = double >
using HermiteExpansion = Expansion< T, HermiteBasis, IsotropicScheme< N, M > >;

}  // namespace tax
