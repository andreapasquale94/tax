# Background

The mathematical objects **tax** propagates: truncated Taylor polynomials, the
graded-lexicographic ordering used to store their coefficients, how the
univariate picture generalises to many variables, and how well a truncated
expansion approximates the true function.

For the per-operation recurrence relations, see
[Recurrence Relations](../internals/recurrences.md).

---

## Truncated Taylor polynomials

A truncated Taylor expansion of a function $f$ in $M$ variables around a point $\mathbf{x}_0$ up to order $N$ is:

$$
f(\mathbf{x}_0 + \delta\mathbf{x}) = \sum_{|\alpha| \le N} f_\alpha \, \delta\mathbf{x}^\alpha + \mathcal{O}(|\delta\mathbf{x}|^{N+1})
$$

where $\alpha = (\alpha_1, \ldots, \alpha_M)$ is a **multi-index** with non-negative integer entries, $|\alpha| = \alpha_1 + \cdots + \alpha_M$ is its total degree, and

$$
\delta\mathbf{x}^\alpha = \delta x_1^{\alpha_1} \cdots \delta x_M^{\alpha_M}
$$

The Taylor coefficients $f_\alpha$ are related to partial derivatives by:

$$
f_\alpha = \frac{1}{\alpha!} \partial^\alpha f(\mathbf{x}_0), \qquad \alpha! = \alpha_1! \cdots \alpha_M!
$$

In the **univariate** case ($M = 1$), the multi-index reduces to a single integer $d$, and the expansion simplifies to:

$$
f(x_0 + \delta x) = \sum_{d=0}^{N} f_d \, \delta x^d, \qquad f_d = \frac{f^{(d)}(x_0)}{d!}
$$

---

## Graded lexicographic ordering

Coefficients are stored using **graded lexicographic (grlex) ordering**:
monomials are grouped by total degree $|\alpha|$, and within each degree group
they are sorted lexicographically by the exponent vector $\alpha$. The
container is a `std::array<T, S>` indexed by that flat index (see
[Coefficient Storage](storage.md)).

**Example for $M = 2$, $N = 2$:**

| Flat index | Multi-index $(\alpha_1, \alpha_2)$ | Monomial | Degree |
|:----------:|:------------------------------------:|:--------:|:------:|
| 0 | (0, 0) | 1 | 0 |
| 1 | (0, 1) | $\delta x_2$ | 1 |
| 2 | (1, 0) | $\delta x_1$ | 1 |
| 3 | (0, 2) | $\delta x_2^2$ | 2 |
| 4 | (1, 1) | $\delta x_1 \delta x_2$ | 2 |
| 5 | (2, 0) | $\delta x_1^2$ | 2 |

The total number of coefficients for $M$ variables at order $N$ is:

$$
S = \binom{N + M}{M}
$$

**Coefficient counts for common $(N, M)$ pairs:**

| $N \backslash M$ | 1 | 2 | 3 | 4 | 5 | 6 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 2 | 3 | 6 | 10 | 15 | 21 | 28 |
| 3 | 4 | 10 | 20 | 35 | 56 | 84 |
| 4 | 5 | 15 | 35 | 70 | 126 | 210 |
| 5 | 6 | 21 | 56 | 126 | 252 | 462 |
| 6 | 7 | 28 | 84 | 210 | 462 | 924 |
| 8 | 9 | 45 | 165 | 495 | 1287 | 3003 |
| 10 | 11 | 66 | 286 | 1001 | 3003 | 8008 |

---

## Multivariate generalisation

All univariate recurrences in the [Recurrence Relations](../internals/recurrences.md) reference generalize naturally to the multivariate case. The key substitutions are:

1. **Scalar index $d$** is replaced by **multi-index $\alpha$** with $|\alpha| = d$.
2. **Inner sums** $\sum_{k=0}^{d}$ become **sub-multi-index sums** $\sum_{\beta \le \alpha}$, iterated over all $\beta$ with $\beta_i \le \alpha_i$ for each component.
3. **Degree constraints** like $1 \le k \le d-1$ become $1 \le |\beta| \le |\alpha|-1$ (or the appropriate range).
4. The **weight factor** $d - k$ generalises to $|\alpha| - |\beta|$, and $k$ to $|\beta|$.

In the implementation, `forEachSubIndex<M>(alpha, lo, hi, callback)` enumerates all sub-multi-indices $\beta \le \alpha$ with $\text{lo} \le |\beta| \le \text{hi}$, calling `callback(flatIndex(beta), flatIndex(alpha - beta), |beta|)`. This provides a uniform interface for both univariate and multivariate kernels, with the univariate path using simple scalar loops as a fast path via `if constexpr (M == 1)`.

---

## Convergence and truncation error

The truncated polynomial $\tilde{f}_N(\mathbf{x}_0 + \delta\mathbf{x})$ of
order $N$ approximates the true function with error

$$
\bigl| f(\mathbf{x}_0 + \delta\mathbf{x}) - \tilde{f}_N(\mathbf{x}_0 + \delta\mathbf{x}) \bigr|
  \le C \, |\delta\mathbf{x}|^{N+1}
$$

within the **radius of convergence** of the underlying Taylor series, where
$C$ bounds the magnitude of the $(N+1)$-th derivatives. The library does
not compute $C$ itself, but two practical proxies are exposed on
`TaylorExpansion`:

- **Coefficient $\ell^\infty$ norm** — `coeffs_norm_inf()` returns
  $\max_{|\alpha| \le N} |f_\alpha|$. A small value at the boundary
  $|\alpha| = N$ suggests the truncation is harmless on the displacement
  scale of interest.
- **Per-degree extraction** — `f.coeff({k})` (or `f.coeff<k,...>()`) gives the
  individual $f_\alpha$; the geometric decay rate of the largest-magnitude
  coefficient at each total degree estimates the convergence radius.

This is the same idea a Jorba–Zou step-size controller applies inside a Taylor
ODE integrator to choose the step $h$ from the last two coefficient norms of
the per-step expansion.

---

## References

- M. Berz, *Modern Map Methods in Particle Beam Physics*, Advances in Imaging
  and Electron Physics, Vol. 108, Academic Press, 1999 — differential algebra
  on truncated multivariate Taylor polynomials.
- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and Techniques
  of Algorithmic Differentiation*, 2nd ed., SIAM, 2008.
- K. Makino and M. Berz, *Taylor models and other validated functional
  inclusion methods*, International Journal of Pure and Applied Mathematics
  4(4), 379–456, 2003.
- W. Rudin, *Principles of Mathematical Analysis*, 3rd ed., McGraw-Hill, 1976 —
  Taylor's theorem and the Lagrange form of the remainder.
- À. Jorba and M. Zou, *A Software Package for the Numerical Integration of ODEs
  by Means of High-Order Taylor Methods*, Experimental Mathematics 14(1),
  99–117, 2005 — using coefficient decay to estimate the convergence radius.
