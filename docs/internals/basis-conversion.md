# Basis Conversion

`tax::toHermite` / `tax::fromHermite` and `tax::toChebyshev` / `tax::fromChebyshev`
(in `tax/core/basis/hermite.hpp` and `tax/core/basis/chebyshev.hpp`) re-express a
dense, isotropic `TaylorExpansion`'s coefficients in the (probabilists')
Hermite or the (first-kind) Chebyshev polynomial basis, and back. This page
documents the math; for usage see
[Guide / Extracting Results](../guide/results.md#basis-conversion) and
[Guide / Eigen Integration](../guide/eigen.md#statistical-moments) (Hermite
conversion is what makes `tax/la/moments.hpp` correct).

## Why a different basis

A Taylor expansion stores coefficients of the monomial basis
$\{ \mathbf{x}^\alpha \}$. Both $\mathrm{He}_n$ (probabilists' Hermite) and
$T_n$ (Chebyshev, first kind) are alternative bases of the *same* truncated
polynomial space, related to the monomials by a **triangular, parity-preserving**
change of basis: a degree-$n$ basis polynomial is a combination of monomials of
degree $n, n-2, n-4, \dots$ (and vice versa) — never degree $n \pm 1$. Two
properties make each substitution useful:

- **Hermite** polynomials are the orthogonal family for the standard-normal
  weight: $\mathbb{E}[\mathrm{He}_n(X)] = 0$ for $X \sim \mathcal N(0,1)$,
  $n \ge 1$, and $\mathbb{E}[\mathrm{He}_m(X)\mathrm{He}_n(X)] = n!\,\delta_{mn}$.
  Converting a polynomial-in-Gaussian-variables to this basis turns moment
  extraction into reading off (and pairwise-multiplying) coefficients — see
  [Statistical Moments](moments.md).
- **Chebyshev** polynomials satisfy $|T_n(x)| \le 1$ on $[-1, 1]$, which makes
  the Chebyshev coefficients of a function restricted to that interval an
  immediately interpretable, well-conditioned decay sequence — useful for
  range/enclosure estimates and truncation-error diagnostics.

## Univariate connection coefficients

Write the forward (monomial → basis) and inverse (basis → monomial) relations
as

$$
x^n = \sum_{m=0}^{\lfloor n/2 \rfloor} c_{\to}(n, m)\, P_{n-2m}(x),
\qquad
P_n(x) = \sum_{m=0}^{\lfloor n/2 \rfloor} c_{\leftarrow}(n, m)\, x^{n-2m},
$$

for a basis family $P$. For the **probabilists' Hermite** polynomials
($\mathrm{He}_0=1$, $\mathrm{He}_1=x$, $\mathrm{He}_{n+1} = x\,\mathrm{He}_n - n\,\mathrm{He}_{n-1}$):

$$
c_{\to}(n, m) = \frac{n!}{m!\,(n-2m)!\,2^m},
\qquad
c_{\leftarrow}(n, m) = (-1)^m\, c_{\to}(n, m).
$$

For the **Chebyshev** polynomials of the first kind
($T_0=1$, $T_1=x$, $T_{n+1} = 2x\,T_n - T_{n-1}$), with $w(n,m) = \tfrac12$
when $n$ is even and $m = n/2$ (else $1$):

$$
c_{\to}(n, m) = w(n,m)\, \binom{n}{m}\, 2^{1-n},
\qquad
c_{\leftarrow}(n, m) = (-1)^m\, \frac{n}{2}\, \frac{(n-m-1)!}{m!\,(n-2m)!}\, 2^{n-2m}
\;\; (n \ge 1),\quad c_{\leftarrow}(0,0)=1.
$$

(`detail::basis::hermiteForwardCoeff` / `hermiteInverseCoeff` and
`chebyshevForwardCoeff` / `chebyshevInverseCoeff` implement these directly —
see DLMF 18.5.10–18.5.11 / Abramowitz & Stegun 22.3 for derivations.)

## The multivariate transform

Both families are **separable products** over independent axes:
$P_\alpha(\mathbf x) = \prod_i P_{\alpha_i}(x_i)$, exactly like the monomials
$\mathbf x^\alpha = \prod_i x_i^{\alpha_i}$. Substituting the univariate
relation axis-by-axis and collecting terms gives the multivariate transform
implemented by `detail::basis::separableBasisTransform` in
`tax/core/basis/connection.hpp`:

$$
h_\beta = \sum_{\substack{\mathbf m \ge 0 \\ |\beta| + 2|\mathbf m| \le N}}
  a_{\beta + 2\mathbf m} \prod_i c(\beta_i + 2 m_i,\, m_i),
$$

where $a$ are the input (monomial or basis) coefficients and $h$ the output
ones; passing $c_\to$ or $c_\leftarrow$ selects the direction. The same driver
serves Hermite and Chebyshev (and any future basis with this parity structure)
— only the connection-coefficient function differs.

## Implementation notes

- Scope: dense, isotropic (`IsotropicScheme<N,M>`) expansions only, mirroring
  `TaylorExpansion::truncate()`'s isotropic-only gate.
- Results are returned as `HermiteCoefficients<T,N,M>` / `ChebyshevCoefficients<T,N,M>`
  — thin wrappers around the same `Coeffs<T,N,M>` array layout, kept distinct
  from `TaylorExpansion` so basis coefficients can't be fed into monomial-basis
  arithmetic (`+`, `*`, `sin`, ...) by mistake.
- The driver enumerates a bounded, generic double simplex (all `beta` up to
  degree `N`, all `m` up to degree `N/2`, guarded per-`beta`) rather than a
  specialised axis-by-axis pass — the same "correct, generic fallback"
  complexity trade-off as `cauchyProductLoop` in the kernel layer.
- Both directions are exact matrix inverses of each other over the truncated
  space (verified by round-trip tests in `tests/core/test_basis_conversion.cpp`).

## References

- M. Abramowitz and I. A. Stegun (eds.), *Handbook of Mathematical Functions*,
  Dover, 1965, §22.3 — Chebyshev-monomial connection coefficients.
- NIST *Digital Library of Mathematical Functions*, §18.5 (Explicit
  Representations), <https://dlmf.nist.gov/18.5> — closed forms for both the
  Hermite and Chebyshev monomial expansions used here.
- G. Szegő, *Orthogonal Polynomials*, 4th ed., AMS Colloquium Publications
  Vol. 23, 1975 — the general theory of classical orthogonal polynomial
  families and their connection coefficients.
- N. Michelotti, E. R. Burnett, and F. Topputo, *Analytical Confidence
  Boundaries for Non-Gaussian Uncertainty in Perturbed Spacecraft Dynamics*,
  arXiv:2607.10095, 2026 — the motivating application: differential-algebra
  propagation with a monomial-to-Hermite basis transformation feeding
  Isserlis'-theorem moment extraction (see [Statistical Moments](moments.md)).
