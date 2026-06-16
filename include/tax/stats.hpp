// include/tax/stats.hpp
//
// Statistics module (namespace `tax::stats`): non-Monte-Carlo computation of
// the statistical moments of a Taylor-polynomial map by contracting its
// coefficients against the raw moments of the input distribution
// (Valli, Armellin, Di Lizia & Lavagna, JGCD 2013).
//
//   MomentProvider   — concept for a source of raw input moments.
//   GaussianMoments  — zero-mean N(0, Sigma) moments via Isserlis' theorem.
//   expectation(f)   — mean of a scalar TE or Eigen matrix of TEs (also E[g]).
//   covariance(F)    — exact covariance matrix of a vector-valued TE map.
//   variance(f)      — exact variance of a scalar TE.
//   centralMoment / skewness / kurtosis — higher standardized moments.
//
// Opt-in: include <tax/stats.hpp> explicitly (it pulls in the core + la stack).

#pragma once

#include <tax/stats/concepts.hpp>
#include <tax/stats/expectation.hpp>
#include <tax/stats/gaussian.hpp>
#include <tax/tax.hpp>
