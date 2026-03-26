#pragma once

#include <tax/basis/tags.hpp>

namespace tax
{

/**
 * @brief Primary template for basis-specific operations.
 * @details Must be specialized for each polynomial basis.
 *
 * Required static methods:
 *   - multiply(out, a, b)         — truncated polynomial multiplication
 *   - reciprocal(out, a)          — multiplicative inverse
 *   - evaluate(coeffs, dx)        — evaluate polynomial at a point (univariate)
 *   - evaluate(coeffs, dx)        — evaluate polynomial at a point (multivariate)
 *   - differentiate(out, in, var) — partial derivative w.r.t. variable
 *   - integrate(out, in, var)     — partial integral w.r.t. variable
 *   - toMonomial(out, in)         — convert to monomial basis
 *   - fromMonomial(out, in)       — convert from monomial basis
 */
template < typename Basis >
struct BasisTraits;

}  // namespace tax
