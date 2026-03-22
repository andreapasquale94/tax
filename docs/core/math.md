# Mathematical Foundations

This page describes the mathematical basis of the **tax** library: how truncated Taylor polynomials are represented, stored, and propagated through arithmetic and transcendental operations via degree-by-degree recurrence relations.

---

## Truncated Taylor Polynomials

A truncated Taylor expansion of a function \(f\) in \(M\) variables around a point \(\mathbf{x}_0\) up to order \(N\) is:

\[
f(\mathbf{x}_0 + \delta\mathbf{x}) = \sum_{|\alpha| \le N} f_\alpha \, \delta\mathbf{x}^\alpha + \mathcal{O}(|\delta\mathbf{x}|^{N+1})
\]

where \(\alpha = (\alpha_1, \ldots, \alpha_M)\) is a **multi-index** with non-negative integer entries, \(|\alpha| = \alpha_1 + \cdots + \alpha_M\) is its total degree, and

\[
\delta\mathbf{x}^\alpha = \delta x_1^{\alpha_1} \cdots \delta x_M^{\alpha_M}
\]

The Taylor coefficients \(f_\alpha\) are related to partial derivatives by:

\[
f_\alpha = \frac{1}{\alpha!} \partial^\alpha f(\mathbf{x}_0), \qquad \alpha! = \alpha_1! \cdots \alpha_M!
\]

In the **univariate** case (\(M = 1\)), the multi-index reduces to a single integer \(d\), and the expansion simplifies to:

\[
f(x_0 + \delta x) = \sum_{d=0}^{N} f_d \, \delta x^d, \qquad f_d = \frac{f^{(d)}(x_0)}{d!}
\]

---

## Graded Lexicographic Ordering

Coefficients are stored in a contiguous `std::array` using **graded lexicographic (grlex) ordering**: monomials are grouped by total degree \(|\alpha|\), and within each degree group they are sorted lexicographically by the exponent vector \(\alpha\).

**Example for \(M = 2\), \(N = 2\):**

| Flat index | Multi-index \((\alpha_1, \alpha_2)\) | Monomial | Degree |
|:----------:|:------------------------------------:|:--------:|:------:|
| 0 | (0, 0) | 1 | 0 |
| 1 | (0, 1) | \(\delta x_2\) | 1 |
| 2 | (1, 0) | \(\delta x_1\) | 1 |
| 3 | (0, 2) | \(\delta x_2^2\) | 2 |
| 4 | (1, 1) | \(\delta x_1 \delta x_2\) | 2 |
| 5 | (2, 0) | \(\delta x_1^2\) | 2 |

The total number of coefficients for \(M\) variables at order \(N\) is:

\[
S = \binom{N + M}{M}
\]

**Coefficient counts for common \((N, M)\) pairs:**

| \(N \backslash M\) | 1 | 2 | 3 | 4 | 5 | 6 |
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

## Arithmetic Operations

### Addition and Subtraction

Addition is coefficient-wise:

\[
(f + g)_\alpha = f_\alpha + g_\alpha
\]

Subtraction is analogous. Scalar addition modifies only the constant term: \((f + c)_\alpha = f_\alpha + c \cdot \delta_{\alpha,0}\).

### Cauchy Product (Multiplication)

**Univariate.** The product of two truncated series is the discrete convolution truncated at order \(N\):

\[
(f \cdot g)_d = \sum_{k=0}^{d} f_k \, g_{d-k}, \qquad d = 0, \ldots, N
\]

**Multivariate.** The Cauchy product generalizes to a sum over sub-multi-indices:

\[
(f \cdot g)_\alpha = \sum_{\beta \le \alpha} f_\beta \, g_{\alpha - \beta}
\]

where \(\beta \le \alpha\) means \(\beta_i \le \alpha_i\) for all \(i\).

The library exploits **symmetry** in the self-product \(f \cdot f\): only unordered pairs \((\beta, \alpha - \beta)\) with \(\beta \le \alpha - \beta\) (in flat index) are enumerated, roughly halving the number of multiplications.

### Scalar Multiplication and Division

Scalar multiplication scales all coefficients: \((c \cdot f)_\alpha = c \cdot f_\alpha\). Division by a scalar is multiplication by \(1/c\). Division by a polynomial uses the reciprocal recurrence (see below).

---

## Algebraic Operations

### Reciprocal

Given \(f\) with \(f_0 \ne 0\), compute \(g = 1/f\) by solving \(f \cdot g = 1\) degree by degree.

**Univariate:**

\[
g_0 = \frac{1}{f_0}, \qquad g_d = -\frac{1}{f_0} \sum_{k=1}^{d} f_k \, g_{d-k}, \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{f_0} \left( \delta_{\alpha,0} - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| \le |\alpha|}} f_\beta \, g_{\alpha-\beta} \right)
\]

### Square Root

Given \(f\) with \(f_0 > 0\), compute \(g = \sqrt{f}\) by solving \(g^2 = f\).

**Univariate:**

\[
g_0 = \sqrt{f_0}, \qquad g_d = \frac{1}{2g_0} \left( f_d - \sum_{k=1}^{d-1} g_k \, g_{d-k} \right), \quad d \ge 1
\]

The inner sum exploits symmetry: for even \(d\), the middle term \(g_{d/2}^2\) is counted once; other pairs \((k, d-k)\) are counted twice.

**Multivariate:**

\[
g_\alpha = \frac{1}{2g_0} \left( f_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| < |\alpha|}} g_\beta \, g_{\alpha - \beta} \right)
\]

with symmetric enumeration: pairs \((\beta, \alpha - \beta)\) with flat index \(\beta < \alpha - \beta\) are counted twice; diagonal pairs (\(\beta = \alpha - \beta\)) are counted once.

### Cubic Root

Given \(f\) with \(f_0 \ne 0\), compute \(g = \sqrt[3]{f}\) by solving \(g^3 = f\).

**Univariate:**

\[
g_0 = \sqrt[3]{f_0}, \qquad g_d = \frac{1}{3g_0^2} \left( f_d - g_0 \cdot q_d^* - \sum_{j=1}^{d-1} g_j \, q_{d-j} \right), \quad d \ge 1
\]

where \(q = g^2\) is maintained incrementally: \(q_d^* = \sum_{k=1}^{d-1} g_k \, g_{d-k}\) is the partial self-product (excluding the unknown \(g_d\)), then finalized as \(q_d = 2 g_0 g_d + q_d^*\). This yields \(\mathcal{O}(N^2)\) total work instead of \(\mathcal{O}(N^3)\).

**Multivariate:**

\[
g_\alpha = \frac{1}{3g_0^2} \left( f_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| < |\alpha|}} g_\beta \bigl( g_0 \, g_{\alpha-\beta} + q_{\alpha-\beta} \bigr) \right)
\]

with \(q = g^2\) updated degree by degree using symmetric enumeration.

### Absolute Value

Absolute value propagates as:

\[
|f| = \begin{cases} f & \text{if } f_0 > 0 \\ -f & \text{if } f_0 < 0 \end{cases}
\]

This is well-defined only when \(f_0 \ne 0\).

---

## Trigonometric Functions

### Sine and Cosine

The sine and cosine of a series \(f\) are computed simultaneously via the coupled recurrence. Let \(s = \sin(f)\) and \(c = \cos(f)\).

**Univariate:**

\[
s_0 = \sin(f_0), \quad c_0 = \cos(f_0)
\]

\[
s_d = \frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, c_k, \qquad d \ge 1
\]

\[
c_d = -\frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, s_k, \qquad d \ge 1
\]

**Multivariate:**

\[
s_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_{\alpha-\beta} \, c_\beta
\]

\[
c_\alpha = -\frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_{\alpha-\beta} \, s_\beta
\]

### Tangent

Tangent is computed by solving \(c \cdot t = s\) degree by degree, where \(s = \sin(f)\) and \(c = \cos(f)\) are obtained from the coupled recurrence above.

**Univariate:**

\[
t_d = \frac{1}{c_0} \left( s_d - \sum_{k=1}^{d} c_k \, t_{d-k} \right), \qquad d \ge 0
\]

**Multivariate:**

\[
t_\alpha = \frac{1}{c_0} \left( s_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| \le |\alpha|}} c_\beta \, t_{\alpha-\beta} \right)
\]

### Arcsine

Compute \(g = \arcsin(f)\) using the helper \(h = \sqrt{1 - f^2}\). This reduces to solving \(h \cdot g' = f'\) degree by degree.

**Univariate:**

\[
g_0 = \arcsin(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
\]

### Arccosine

\[
\arccos(f) = \frac{\pi}{2} - \arcsin(f)
\]

Implemented by negating the arcsine result and adding \(\pi/2\) to the constant term.

### Arctangent

Compute \(g = \arctan(f)\) using the helper \(h = 1 + f^2\). Solves \(h \cdot g' = f'\) degree by degree.

**Univariate:**

\[
g_0 = \arctan(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
\]

### Arctangent (Two-Argument)

Compute \(g = \text{atan2}(y, x)\) using the helper \(h = x^2 + y^2\). Solves the coupled system degree by degree.

**Univariate:**

\[
g_0 = \text{atan2}(y_0, x_0)
\]

\[
g_d = \frac{1}{d \cdot h_0} \left( d(x_0 y_d - y_0 x_d) + \sum_{k=1}^{d-1} k \bigl( x_{d-k} y_k - y_{d-k} x_k - h_{d-k} g_k \bigr) \right), \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{h_0} \left( (x_0 y_\alpha - y_0 x_\alpha) + \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \bigl( x_\beta y_{\alpha-\beta} - y_\beta x_{\alpha-\beta} - h_\beta g_{\alpha-\beta} \bigr) \right)
\]

---

## Hyperbolic Functions

### Hyperbolic Sine and Cosine

The coupled recurrence for \(\text{sh} = \sinh(f)\) and \(\text{ch} = \cosh(f)\) has the same structure as sine/cosine but with a **positive sign** coupling.

**Univariate:**

\[
\text{sh}_0 = \sinh(f_0), \quad \text{ch}_0 = \cosh(f_0)
\]

\[
\text{sh}_d = \frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, \text{ch}_k, \qquad d \ge 1
\]

\[
\text{ch}_d = \frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, \text{sh}_k, \qquad d \ge 1
\]

**Multivariate:**

\[
\text{sh}_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_{\alpha-\beta} \, \text{ch}_\beta
\]

\[
\text{ch}_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_{\alpha-\beta} \, \text{sh}_\beta
\]

Note the sign difference from the trigonometric case: both sums are positive.

### Hyperbolic Tangent

Computed by solving \(\text{ch} \cdot t = \text{sh}\) degree by degree, identical in structure to the tangent recurrence.

**Univariate:**

\[
t_d = \frac{1}{\text{ch}_0} \left( \text{sh}_d - \sum_{k=1}^{d} \text{ch}_k \, t_{d-k} \right), \qquad d \ge 0
\]

**Multivariate:**

\[
t_\alpha = \frac{1}{\text{ch}_0} \left( \text{sh}_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| \le |\alpha|}} \text{ch}_\beta \, t_{\alpha-\beta} \right)
\]

### Inverse Hyperbolic Sine

Compute \(g = \text{asinh}(f)\) using \(h = \sqrt{1 + f^2}\). Solves \(h \cdot g' = f'\).

**Univariate:**

\[
g_0 = \text{asinh}(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
\]

### Inverse Hyperbolic Cosine

Compute \(g = \text{acosh}(f)\) using \(h = \sqrt{f^2 - 1}\). Requires \(f_0 > 1\). Same recurrence structure as asinh.

**Univariate:**

\[
g_0 = \text{acosh}(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
\]

### Inverse Hyperbolic Tangent

Compute \(g = \text{atanh}(f)\) using \(h = 1 - f^2\). Requires \(|f_0| < 1\). Same recurrence structure.

**Univariate:**

\[
g_0 = \text{atanh}(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
\]

---

## Transcendental Functions

### Exponential

Compute \(g = \exp(f)\).

**Univariate:**

\[
g_0 = \exp(f_0), \qquad g_d = \frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, g_k, \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} |\beta| \, f_\beta \, g_{\alpha-\beta}
\]

This recurrence follows from differentiating \(g = \exp(f)\) to get \(g' = f' \cdot g\), then matching coefficients degree by degree.

### Logarithm

Compute \(g = \ln(f)\) with \(f_0 > 0\).

**Univariate:**

\[
g_0 = \ln(f_0), \qquad g_d = \frac{1}{f_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, f_{d-k} \, g_k \right), \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{f_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_\beta \, g_{\alpha-\beta} \right)
\]

This is derived from \(f \cdot g' = f'\), matching coefficients.

### Logarithm Base 10

\[
\log_{10}(f) = \frac{\ln(f)}{\ln(10)}
\]

Computed by applying the natural logarithm recurrence and scaling all coefficients by \(1/\ln(10)\).

---

## Power Functions

### Integer Power

For integer exponent \(n\), \(f^n\) is computed via **binary exponentiation** using the Cauchy product. Special cases: \(n = 0\) returns 1, \(n = 1\) returns \(f\), \(n = -1\) uses the reciprocal recurrence, and negative \(n\) computes the reciprocal first, then raises to \(|n|\).

### Real Power

Compute \(g = f^c\) for real exponent \(c\) with \(f_0 > 0\).

**Univariate:**

\[
g_0 = f_0^c, \qquad g_d = \frac{1}{d \cdot f_0} \sum_{k=0}^{d-1} \bigl( c(d-k) - k \bigr) \, f_{d-k} \, g_k, \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{|\alpha| \cdot f_0} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} \bigl( c \cdot |\beta| - (|\alpha| - |\beta|) \bigr) \, f_\beta \, g_{\alpha-\beta}
\]

This recurrence is derived from the identity \(f \cdot g' = c \cdot f' \cdot g\).

### DA Power

When both base and exponent are DA objects, \(f^g\) is computed as:

\[
f^g = \exp(g \cdot \ln(f))
\]

using the logarithm, Cauchy product, and exponential recurrences in sequence.

---

## Special Functions

### Error Function

Compute \(g = \text{erf}(f)\) using the helper:

\[
h = \frac{2}{\sqrt{\pi}} \exp(-f^2)
\]

which is the derivative of \(\text{erf}\). Then the recurrence follows the same exponential-like pattern.

**Univariate:**

\[
g_0 = \text{erf}(f_0), \qquad g_d = \frac{1}{d} \sum_{k=0}^{d-1} (d-k) \, f_{d-k} \, h_k, \quad d \ge 1
\]

**Multivariate:**

\[
g_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} |\beta| \, f_\beta \, h_{\alpha-\beta}
\]

### Distance Functions

**Hypot (2-argument):** \(\text{hypot}(x, y) = \sqrt{x^2 + y^2}\). Computed by forming the Cauchy self-products \(x^2\) and \(y^2\), summing, and applying the square root recurrence.

**Hypot (3-argument):** \(\text{hypot}(x, y, z) = \sqrt{x^2 + y^2 + z^2}\). Same approach extended to three terms.

---

## Multivariate Generalisation

All univariate recurrences presented above generalize naturally to the multivariate case. The key substitutions are:

1. **Scalar index \(d\)** is replaced by **multi-index \(\alpha\)** with \(|\alpha| = d\).
2. **Inner sums** \(\sum_{k=0}^{d}\) become **sub-multi-index sums** \(\sum_{\beta \le \alpha}\), iterated over all \(\beta\) with \(\beta_i \le \alpha_i\) for each component.
3. **Degree constraints** like \(1 \le k \le d-1\) become \(1 \le |\beta| \le |\alpha|-1\) (or the appropriate range).
4. The **weight factor** \(d - k\) generalises to \(|\alpha| - |\beta|\), and \(k\) to \(|\beta|\).

In the implementation, the function `forEachSubIndex<M>(alpha, lo, hi, callback)` enumerates all sub-multi-indices \(\beta \le \alpha\) with \(\text{lo} \le |\beta| \le \text{hi}\), calling `callback(flatIndex(beta), flatIndex(alpha - beta), |beta|)`. This provides a uniform interface for both univariate and multivariate kernels, with the univariate path using simple scalar loops as a fast path via `if constexpr (M == 1)`.
