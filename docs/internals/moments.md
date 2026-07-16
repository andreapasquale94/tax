# Statistical Moments

`tax::la::mean`, `tax::la::covariance`, `tax::la::skewnessTensor`,
`tax::la::kurtosisTensor`, and `tax::la::excessKurtosisTensor` (in
`tax/la/moments.hpp`) extract statistical moments of a polynomial map
$\mathbf F : \mathbb R^M \to \mathbb R^D$, represented as an Eigen vector of
dense, isotropic `TaylorExpansion`s, under one standing assumption: the
expansion's $M$ formal variables are **independent standard normal**,
$\mathbf x \sim \mathcal N(\mathbf 0, I_M)$. This page documents the math; for
usage see [Guide / Eigen Integration](../guide/eigen.md#statistical-moments).

`tax/la/moments.hpp` is an **opt-in** header: it is deliberately not included
by the `<tax/tax.hpp>` / `<tax/la.hpp>` umbrellas because it depends on Eigen's
heavy `unsupported/Eigen/CXX11/Tensor` module, so every consumer of the
umbrella would otherwise pay that compile-time cost. Include it explicitly
(`#include <tax/la/moments.hpp>`) when you need moments.

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

## Per-component standardized coefficients

Besides the full joint tensors, the module exposes the everyday **marginal**
non-Gaussianity coefficients as fixed-size $D \times 1$ vectors:

$$
\text{skewness}(F)_i = \frac{\mathbb E[(F_i-\mu_i)^3]}{\sigma_i^3},
\qquad
\text{kurtosis}(F)_i = \frac{\mathbb E[(F_i-\mu_i)^4]}{\sigma_i^4},
\qquad
\text{excessKurtosis}(F)_i = \text{kurtosis}(F)_i - 3,
$$

with $\sigma_i^2 = \mathrm{Var}(F_i)$. These are exactly the diagonal entries of
the corresponding central-moment tensors, normalized by the appropriate power of
$\sigma_i$ — but they are computed **directly** per component (one `jointRawMoment2`
plus a `jointRawMoment3`/`jointRawMoment4` on the centered coefficients of
$F_i$), which is $O(D)$ evaluations rather than the $O(D^3)$/$O(D^4)$ of building
the whole tensor. `kurtosis` follows the Pearson convention (value $3$ for a
Gaussian marginal); `excessKurtosis` is the Fisher form (value $0$ for a
Gaussian). A zero-variance (constant) component makes the normalization
undefined and yields a non-finite entry.

## Tensor layout

The third and fourth central-moment tensors are returned as **fixed-size**
`Eigen::TensorFixedSize` objects (Eigen's `unsupported/Eigen/CXX11/Tensor`
module), so their shape is a compile-time constant and their storage is
stack-allocated — matching the library's fixed-shape, allocation-free
convention. The map dimension $D$ is taken from `Derived::SizeAtCompileTime`;
a dynamic-size input map is rejected with a `static_assert` (the same
compile-time-$D$ requirement is now shared by `covariance`, which returns a
fixed-size $D \times D$ `Eigen::Matrix`).

- `skewnessTensor(F)` is an `Eigen::TensorFixedSize<T, Eigen::Sizes<D, D, D>>`
  with `(i, j, k)` $= S_{ijk} = \mathbb E[(F_i-\mu_i)(F_j-\mu_j)(F_k-\mu_k)]$.
- `kurtosisTensor(F)` is an
  `Eigen::TensorFixedSize<T, Eigen::Sizes<D, D, D, D>>` with
  `(i, j, k, l)` $= K_{ijkl} = \mathbb E[(F_i-\mu_i)(F_j-\mu_j)(F_k-\mu_k)(F_l-\mu_l)]$.

Both tensors are fully symmetric under any permutation of their indices. The
implementation exploits this: it computes each value once over sorted index
tuples ($i \le j \le k$, resp. $i \le j \le k \le l$) and scatters it to every
distinct permutation via `std::next_permutation`, which also collapses repeated
indices correctly. Because every index tuple is a permutation of exactly one
sorted tuple, this fills every entry — no separate zero-initialization pass is
needed.

## Excess kurtosis and non-Gaussianity

For **jointly Gaussian** variables $Y_i, Y_j, Y_k, Y_l$ with covariance $C$,
Isserlis' theorem gives the fourth moment as a sum over the three pairings:

$$
\mathbb E[Y_iY_jY_kY_l] = C_{ij}C_{kl} + C_{ik}C_{jl} + C_{il}C_{jk}.
$$

`excessKurtosisTensor(F)` (also an `Eigen::Tensor<T, 4>`) subtracts exactly
this baseline (built from `covariance(F)`) from `kurtosisTensor(F)`,
elementwise. The diagonal entry `excessKurtosisTensor(F)(i, i, i, i)` recovers
$\mathbb E[(F_i-\mu_i)^4] - 3\sigma_i^4$ (the familiar scalar excess kurtosis
scaled by $\sigma_i^4$); the full tensor is the standard way to detect and
quantify departure from joint Gaussianity in $\mathbf F$'s output distribution
— the diagnostic this module was built for.

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
