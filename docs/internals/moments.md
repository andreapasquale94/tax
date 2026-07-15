# Statistical Moments

`tax::la::mean`, `tax::la::covariance`, `tax::la::skewnessTensor`,
`tax::la::kurtosisTensor`, and `tax::la::excessKurtosisTensor` (in
`tax/la/moments.hpp`) extract statistical moments of a polynomial map
$\mathbf F : \mathbb R^M \to \mathbb R^D$, represented as an Eigen vector of
dense, isotropic `TaylorExpansion`s, under one standing assumption: the
expansion's $M$ formal variables are **independent standard normal**,
$\mathbf x \sim \mathcal N(\mathbf 0, I_M)$. This page documents the math; for
usage see [Guide / Eigen Integration](../guide/eigen.md#statistical-moments).

## The Gaussian-input convention

Differential-algebra uncertainty propagation typically *whitens* the input
distribution before building expansion variables: given $\mathbf x_0$ and a
covariance $\Sigma = L L^\top$ (Cholesky factor $L$), the physical state is
$\mathbf x_0 + L\,\boldsymbol{\delta x}$, and it is $\boldsymbol{\delta x}$ —
not the physical displacement — that is built as the expansion's formal
variables. That whitening step is the caller's responsibility (e.g. scale
`tax::la::variables`'s input point through $L$); once done, `tax/la/moments.hpp`
assumes $\boldsymbol{\delta x} \sim \mathcal N(\mathbf 0, I_M)$ i.i.d. and every
moment of $\mathbf F(\boldsymbol{\delta x})$ reduces to expectations of that
one simple, decoupled distribution.

## From Gaussian moments to polynomial moments

For $\mathbf F$ given by its monomial coefficients, any joint raw moment of $k$
components is a finite sum over monomial coefficients:

$$
\mathbb E\!\left[\prod_{t=1}^{k} F_{i_t}(\mathbf x)\right]
= \sum_{\alpha^{(1)}, \dots, \alpha^{(k)}}
  \left(\prod_{t=1}^{k} a^{(i_t)}_{\alpha^{(t)}}\right)
  \mathbb E\!\left[\mathbf x^{\alpha^{(1)} + \cdots + \alpha^{(k)}}\right].
$$

Because the axes are independent, the remaining expectation factors
axis-by-axis, $\mathbb E[\mathbf x^{\mathbf p}] = \prod_j \mathbb E[x_j^{p_j}]$,
and each factor is the classical raw moment of a single standard normal
variable — a special case of **Isserlis' (Wick's) theorem**, closed-form as a
double factorial:

$$
\mathbb E[X^p] = \begin{cases} (p-1)!! & p \text{ even} \\ 0 & p \text{ odd} \end{cases},
\qquad X \sim \mathcal N(0,1).
$$

`detail::gaussianRawMoment` implements this closed form directly (no pairing
enumeration needed for a single variable); `detail::jointRawMoment1..4` walk
the $k=1,2,3,4$ sums above via nested `forEachMonomial` passes, skipping any
zero coefficient. This gives mean ($k=1$), the raw second moment feeding
covariance ($k=2$), and the raw third/fourth moments feeding the skewness and
kurtosis tensors ($k=3,4$) from one shared code path.

## Central moments via constant-term shifting

Subtracting a scalar constant $\mu$ from a polynomial changes only its
degree-0 coefficient — every other coefficient is untouched regardless of the
polynomial's own degree. So rather than expand raw-to-central moment
conversion formulas (which grow combinatorially with the moment order),
`detail::centeredCoeffs(f, mu)` builds $\widetilde f = f - \mu$ by shifting just
`f[0]`, and every central moment
$\mathbb E[(F_{i_1}-\mu_{i_1})\cdots(F_{i_k}-\mu_{i_k})]$ is then simply
`jointRawMomentK` applied to the **centered** coefficient arrays. `mean(F)`
supplies the $\mu_i$; `covariance`, `skewnessTensor`, and `kurtosisTensor` all
build on it.

## Tensor layout

`tax` has no rank-3/4 tensor type, so the higher tensors are returned as
slices of ordinary matrices:

- `skewnessTensor(F)[k]` is the $D \times D$ symmetric matrix with entry
  $(i,j) = S_{ijk} = \mathbb E[(F_i-\mu_i)(F_j-\mu_j)(F_k-\mu_k)]$.
- `kurtosisTensor(F)[k][l]` is the $D \times D$ symmetric matrix with entry
  $(i,j) = K_{ijkl} = \mathbb E[(F_i-\mu_i)(F_j-\mu_j)(F_k-\mu_k)(F_l-\mu_l)]$.

## Excess kurtosis and non-Gaussianity

For **jointly Gaussian** variables $Y_i, Y_j, Y_k, Y_l$ with covariance $C$,
Isserlis' theorem gives the fourth moment as a sum over the three pairings:

$$
\mathbb E[Y_iY_jY_kY_l] = C_{ij}C_{kl} + C_{ik}C_{jl} + C_{il}C_{jk}.
$$

`excessKurtosisTensor(F)` subtracts exactly this baseline (built from
`covariance(F)`) from `kurtosisTensor(F)`, elementwise. The diagonal entry
`excessKurtosisTensor(F)[i][i](i,i)` recovers the familiar scalar excess
kurtosis $\mathbb E[(F_i-\mu_i)^4]/\sigma_i^4 - 3$; the full tensor is the
standard way to detect and quantify departure from joint Gaussianity in
$\mathbf F$'s output distribution — the diagnostic this module was built for.

## Implementation notes

- Scope: dense, isotropic (`IsotropicScheme<N,M>`) expansions only
  (`detail::MomentsCompatibleTE`), matching the basis-conversion module's
  scope.
- Complexity is $O(\text{nCoeff}^k)$ for a $k$-th order joint moment — a
  correct, generic double/triple/quadruple simplex walk (with a zero-coefficient
  short-circuit), not a specially-optimized path. This mirrors the "generic,
  always-correct fallback" role `cauchyProductLoop` plays in the kernel layer;
  optimizing the common cases (e.g. precomputed Hermite-orthogonality weight
  tables, as in the motivating paper) is future work, not required for
  correctness.
- The implementation deliberately works from monomial coefficients directly
  (closed-form Gaussian raw moments) rather than through the Hermite basis:
  mathematically identical results, but the derivation needs nothing beyond
  elementary independence + the univariate Gaussian moment formula — no
  Hermite triple/quadruple-product linearization coefficients to derive or
  verify. See [Basis Conversion](basis-conversion.md) for where the Hermite
  route *is* used (mean/covariance reduce to trivial orthogonality lookups
  once a polynomial is expressed in the Hermite basis).

## References

- L. Isserlis, "On a Formula for the Product-Moment Coefficient of any Order
  of a Normal Frequency Distribution in any Number of Variables," *Biometrika*
  12(1–2), 1918, pp. 134–139 — the origin of the pairing formula generalized
  here for joint Gaussian moments.
- N. Michelotti, E. R. Burnett, and F. Topputo, *Analytical Confidence
  Boundaries for Non-Gaussian Uncertainty in Perturbed Spacecraft Dynamics*,
  arXiv:2607.10095, 2026 — differential-algebra uncertainty propagation using
  a monomial-to-Hermite basis transformation and Isserlis' theorem to extract
  moments analytically (the motivating reference for this module; arXiv access
  was blocked from this environment, so the implementation here is an
  independent, from-first-principles derivation rather than a port of the
  paper's algorithm — see the implementation note above).
- M. Rasotto et al., "Differential Algebra Space Toolbox for Nonlinear
  Uncertainty Propagation in Space Dynamics," ICATT 2016 — prior DA-based
  uncertainty-propagation work (also cited in
  [Recurrence Relations](recurrences.md)).
