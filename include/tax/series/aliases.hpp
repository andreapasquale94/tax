#pragma once

// The basis-generic carrier `tax::Expansion< T, Basis, Scheme, Storage >` now
// lives in <tax/core/expansion.hpp> (with `TaylorExpansion` as its
// TaylorBasis alias). This header just adds the basis-named series aliases.

#include <tax/core/scheme/isotropic.hpp>
#include <tax/core/expansion.hpp>
#include <tax/core/basis.hpp>
#include <tax/series/chebyshev_basis.hpp>
#include <tax/series/hermite_basis.hpp>
#include <tax/series/legendre_basis.hpp>
#include <tax/core/taylor_basis.hpp>

namespace tax
{

/// Univariate basis-generic series.
template < typename Basis, int N, typename T = double >
using Series = Expansion< T, Basis, IsotropicScheme< N, 1 > >;

/// Order-N, M-variate Taylor (monomial-basis) expansion.
template < int N, int M = 1, typename T = double >
using TaylorSeries = Expansion< T, TaylorBasis, IsotropicScheme< N, M > >;

/// Order-N, M-variate Chebyshev (first-kind) expansion.
template < int N, int M = 1, typename T = double >
using ChebyshevSeries = Expansion< T, ChebyshevBasis, IsotropicScheme< N, M > >;

/// Order-N, M-variate Legendre expansion.
template < int N, int M = 1, typename T = double >
using LegendreSeries = Expansion< T, LegendreBasis, IsotropicScheme< N, M > >;

/// Order-N, M-variate probabilists' Hermite expansion.
template < int N, int M = 1, typename T = double >
using HermiteSeries = Expansion< T, HermiteBasis, IsotropicScheme< N, M > >;

}  // namespace tax
