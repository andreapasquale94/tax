# Recurrence Relations

This page is the **recurrence reference** for **tax**: for every supported operation it gives the degree-by-degree recurrence relation used to propagate truncated Taylor polynomials. Each entry lists the univariate ($M = 1$) form and its multivariate generalisation, matching the kernels in `include/tax/expansion/detail/`. See [Kernels & Recurrences](kernels.md) for how these are dispatched and implemented.

For the underlying theory — what truncated Taylor polynomials are, the graded-lexicographic coefficient ordering, and how univariate recurrences extend to many variables — see [Foundations & Ordering](../concepts/foundations.md). For truncation-error bounds and convergence diagnostics, see [Convergence & Truncation](../concepts/convergence.md).

---

## Arithmetic Operations

### Addition and Subtraction

Addition is coefficient-wise:

$$
(f + g)_\alpha = f_\alpha + g_\alpha
$$

Subtraction is analogous. Scalar addition modifies only the constant term: $(f + c)_\alpha = f_\alpha + c \cdot \delta_{\alpha,0}$.

### Cauchy Product (Multiplication)

**Univariate.** The product of two truncated series is the discrete convolution truncated at order $N$:

$$
(f \cdot g)_d = \sum_{k=0}^{d} f_k \, g_{d-k}, \qquad d = 0, \ldots, N
$$

**Multivariate.** The Cauchy product generalizes to a sum over sub-multi-indices:

$$
(f \cdot g)_\alpha = \sum_{\beta \le \alpha} f_\beta \, g_{\alpha - \beta}
$$

where $\beta \le \alpha$ means $\beta_i \le \alpha_i$ for all $i$.

The library exploits **symmetry** in the self-product $f \cdot f$: only unordered pairs $(\beta, \alpha - \beta)$ with $\beta \le \alpha - \beta$ (in flat index) are enumerated, roughly halving the number of multiplications.

### Scalar Multiplication and Division

Scalar multiplication scales all coefficients: $(c \cdot f)_\alpha = c \cdot f_\alpha$. Division by a scalar is multiplication by $1/c$. Division by a polynomial uses the reciprocal recurrence (see below).

---

## Algebraic Operations

### Reciprocal

Given $f$ with $f_0 \ne 0$, compute $g = 1/f$ by solving $f \cdot g = 1$ degree by degree.

**Univariate:**

$$
g_0 = \frac{1}{f_0}, \qquad g_d = -\frac{1}{f_0} \sum_{k=1}^{d} f_k \, g_{d-k}, \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{f_0} \left( \delta_{\alpha,0} - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| \le |\alpha|}} f_\beta \, g_{\alpha-\beta} \right)
$$

### Square Root

Given $f$ with $f_0 > 0$, compute $g = \sqrt{f}$ by solving $g^2 = f$.

**Univariate:**

$$
g_0 = \sqrt{f_0}, \qquad g_d = \frac{1}{2g_0} \left( f_d - \sum_{k=1}^{d-1} g_k \, g_{d-k} \right), \quad d \ge 1
$$

The inner sum exploits symmetry: for even $d$, the middle term $g_{d/2}^2$ is counted once; other pairs $(k, d-k)$ are counted twice.

**Multivariate:**

$$
g_\alpha = \frac{1}{2g_0} \left( f_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| < |\alpha|}} g_\beta \, g_{\alpha - \beta} \right)
$$

with symmetric enumeration: pairs $(\beta, \alpha - \beta)$ with flat index $\beta < \alpha - \beta$ are counted twice; diagonal pairs ($\beta = \alpha - \beta$) are counted once.

### Cubic Root

Given $f$ with $f_0 \ne 0$, compute $g = \sqrt[3]{f}$ by solving $g^3 = f$.

**Univariate:**

$$
g_0 = \sqrt[3]{f_0}, \qquad g_d = \frac{1}{3g_0^2} \left( f_d - g_0 \cdot q_d^* - \sum_{j=1}^{d-1} g_j \, q_{d-j} \right), \quad d \ge 1
$$

where $q = g^2$ is maintained incrementally: $q_d^* = \sum_{k=1}^{d-1} g_k \, g_{d-k}$ is the partial self-product (excluding the unknown $g_d$), then finalized as $q_d = 2 g_0 g_d + q_d^*$. This yields $\mathcal{O}(N^2)$ total work instead of $\mathcal{O}(N^3)$.

**Multivariate:**

$$
g_\alpha = \frac{1}{3g_0^2} \left( f_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| < |\alpha|}} g_\beta \bigl( g_0 \, g_{\alpha-\beta} + q_{\alpha-\beta} \bigr) \right)
$$

with $q = g^2$ updated degree by degree using symmetric enumeration.

---

## Trigonometric Functions

### Sine and Cosine

The sine and cosine of a series $f$ are computed simultaneously via the coupled recurrence. Let $s = \sin(f)$ and $c = \cos(f)$.

**Univariate:**

$$
s_0 = \sin(f_0), \quad c_0 = \cos(f_0)
$$

$$
s_d = \frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, c_k, \qquad d \ge 1
$$

$$
c_d = -\frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, s_k, \qquad d \ge 1
$$

**Multivariate:**

$$
s_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_{\alpha-\beta} \, c_\beta
$$

$$
c_\alpha = -\frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_{\alpha-\beta} \, s_\beta
$$

### Tangent

Tangent is computed by solving $c \cdot t = s$ degree by degree, where $s = \sin(f)$ and $c = \cos(f)$ are obtained from the coupled recurrence above.

**Univariate:**

$$
t_d = \frac{1}{c_0} \left( s_d - \sum_{k=1}^{d} c_k \, t_{d-k} \right), \qquad d \ge 0
$$

**Multivariate:**

$$
t_\alpha = \frac{1}{c_0} \left( s_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| \le |\alpha|}} c_\beta \, t_{\alpha-\beta} \right)
$$

### Arcsine

Compute $g = \arcsin(f)$ using the helper $h = \sqrt{1 - f^2}$. This reduces to solving $h \cdot g' = f'$ degree by degree.

**Univariate:**

$$
g_0 = \arcsin(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

### Arccosine

$$
\arccos(f) = \frac{\pi}{2} - \arcsin(f)
$$

Implemented by negating the arcsine result and adding $\pi/2$ to the constant term.

### Arctangent

Compute $g = \arctan(f)$ using the helper $h = 1 + f^2$. Solves $h \cdot g' = f'$ degree by degree.

**Univariate:**

$$
g_0 = \arctan(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

### Arctangent (Two-Argument)

Compute $g = \text{atan2}(y, x)$ using the helper $h = x^2 + y^2$. Solves the coupled system degree by degree.

**Univariate:**

$$
g_0 = \text{atan2}(y_0, x_0)
$$

$$
g_d = \frac{1}{d \cdot h_0} \left( d(x_0 y_d - y_0 x_d) + \sum_{k=1}^{d-1} k \bigl( x_{d-k} y_k - y_{d-k} x_k - h_{d-k} g_k \bigr) \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( (x_0 y_\alpha - y_0 x_\alpha) + \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \bigl( x_\beta y_{\alpha-\beta} - y_\beta x_{\alpha-\beta} - h_\beta g_{\alpha-\beta} \bigr) \right)
$$

---

## Hyperbolic Functions

### Hyperbolic Sine and Cosine

The coupled recurrence for $\text{sh} = \sinh(f)$ and $\text{ch} = \cosh(f)$ has the same structure as sine/cosine but with a **positive sign** coupling.

**Univariate:**

$$
\text{sh}_0 = \sinh(f_0), \quad \text{ch}_0 = \cosh(f_0)
$$

$$
\text{sh}_d = \frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, \text{ch}_k, \qquad d \ge 1
$$

$$
\text{ch}_d = \frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, \text{sh}_k, \qquad d \ge 1
$$

**Multivariate:**

$$
\text{sh}_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_{\alpha-\beta} \, \text{ch}_\beta
$$

$$
\text{ch}_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_{\alpha-\beta} \, \text{sh}_\beta
$$

Note the sign difference from the trigonometric case: both sums are positive.

### Hyperbolic Tangent

Computed by solving $\text{ch} \cdot t = \text{sh}$ degree by degree, identical in structure to the tangent recurrence.

**Univariate:**

$$
t_d = \frac{1}{\text{ch}_0} \left( \text{sh}_d - \sum_{k=1}^{d} \text{ch}_k \, t_{d-k} \right), \qquad d \ge 0
$$

**Multivariate:**

$$
t_\alpha = \frac{1}{\text{ch}_0} \left( \text{sh}_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| \le |\alpha|}} \text{ch}_\beta \, t_{\alpha-\beta} \right)
$$

### Inverse Hyperbolic Sine

Compute $g = \text{asinh}(f)$ using $h = \sqrt{1 + f^2}$. Solves $h \cdot g' = f'$.

**Univariate:**

$$
g_0 = \text{asinh}(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

### Inverse Hyperbolic Cosine

Compute $g = \text{acosh}(f)$ using $h = \sqrt{f^2 - 1}$. Requires $f_0 > 1$. Same recurrence structure as asinh.

**Univariate:**

$$
g_0 = \text{acosh}(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

### Inverse Hyperbolic Tangent

Compute $g = \text{atanh}(f)$ using $h = 1 - f^2$. Requires $|f_0| < 1$. Same recurrence structure.

**Univariate:**

$$
g_0 = \text{atanh}(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

---

## Transcendental Functions

### Exponential

Compute $g = \exp(f)$.

**Univariate:**

$$
g_0 = \exp(f_0), \qquad g_d = \frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, g_k, \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} |\beta| \, f_\beta \, g_{\alpha-\beta}
$$

This recurrence follows from differentiating $g = \exp(f)$ to get $g' = f' \cdot g$, then matching coefficients degree by degree.

### Logarithm

Compute $g = \ln(f)$ with $f_0 > 0$.

**Univariate:**

$$
g_0 = \ln(f_0), \qquad g_d = \frac{1}{f_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, f_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{f_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_\beta \, g_{\alpha-\beta} \right)
$$

This is derived from $f \cdot g' = f'$, matching coefficients.

---

## Power Functions

### Integer Power

For integer exponent $n$, $f^n$ is computed via **binary exponentiation** using the Cauchy product. Special cases: $n = 0$ returns 1, $n = 1$ returns $f$, $n = -1$ uses the reciprocal recurrence, and negative $n$ computes the reciprocal first, then raises to $|n|$.

### Real Power

Compute $g = f^c$ for real exponent $c$ with $f_0 > 0$.

**Univariate:**

$$
g_0 = f_0^c, \qquad g_d = \frac{1}{d \cdot f_0} \sum_{k=0}^{d-1} \bigl( c(d-k) - k \bigr) \, f_{d-k} \, g_k, \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{|\alpha| \cdot f_0} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} \bigl( c \cdot |\beta| - (|\alpha| - |\beta|) \bigr) \, f_\beta \, g_{\alpha-\beta}
$$

This recurrence is derived from the identity $f \cdot g' = c \cdot f' \cdot g$.

---

## Special Functions

### Error Function

Compute $g = \text{erf}(f)$ using the helper:

$$
h = \frac{2}{\sqrt{\pi}} \exp(-f^2)
$$

which is the derivative of $\text{erf}$. Then the recurrence follows the same exponential-like pattern.

**Univariate:**

$$
g_0 = \text{erf}(f_0), \qquad g_d = \frac{1}{d} \sum_{k=0}^{d-1} (d-k) \, f_{d-k} \, h_k, \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} |\beta| \, f_\beta \, h_{\alpha-\beta}
$$

---

## References

- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and Techniques
  of Algorithmic Differentiation*, 2nd ed., SIAM, 2008 — forward-mode Taylor
  coefficient propagation and the recurrences for elementary functions.
- R. D. Neidinger, *Introduction to Automatic Differentiation and MATLAB
  Object-Oriented Programming*, SIAM Review 52(3), 545–563, 2010 — explicit
  degree-by-degree recurrences for the standard transcendental functions.
- M. Berz, *Modern Map Methods in Particle Beam Physics*, Advances in Imaging
  and Electron Physics, Vol. 108, Academic Press, 1999 — differential algebra
  of truncated multivariate Taylor polynomials.
