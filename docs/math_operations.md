# Math Operations Reference

This document describes every mathematical operation supported by TAX, together with the recurrence relations used to propagate truncated Taylor series through each operation.

## Notation

Let $f$ be a truncated multivariate Taylor polynomial in $M$ variables up to total degree $N$:

$$f(\mathbf{x}) = \sum_{|\alpha| \le N} f_\alpha \, \mathbf{x}^\alpha, \qquad \alpha \in \mathbb{N}_0^M, \quad |\alpha| = \sum_{i=1}^{M} \alpha_i$$

where $\mathbf{x}^\alpha = x_1^{\alpha_1} \cdots x_M^{\alpha_M}$ is the monomial indexed by the multi-index $\alpha$.

We write $f_\alpha$ for the coefficient of $\mathbf{x}^\alpha$ in $f$. In the univariate case ($M=1$), this simplifies to $f_d$ for the coefficient of $x^d$.

The notation $\beta \le \alpha$ means $\beta_i \le \alpha_i$ for every $i$, i.e. $\beta$ is a sub-multi-index of $\alpha$. The sum $\sum_{\beta \le \alpha}$ runs over all such sub-indices. When degree bounds are given, e.g. $1 \le |\beta| \le d-1$, the sum is further restricted to sub-indices in that degree range.

All operations are truncated: terms of total degree greater than $N$ are discarded.

---

## Arithmetic

### Addition and Subtraction

Coefficient-wise:

$$(f \pm g)_\alpha = f_\alpha \pm g_\alpha$$

TAX flattens chains of additions into a single `SumExpr` to avoid nested binary temporaries.

### Multiplication (Cauchy Product)

The product of two truncated power series is the **Cauchy product**, truncated at degree $N$.

**Univariate ($M = 1$):**

$$(f \cdot g)_d = \sum_{k=0}^{d} f_k \, g_{d-k}, \qquad d = 0, \ldots, N$$

**Multivariate:**

$$(f \cdot g)_\alpha = \sum_{\beta \le \alpha} f_\beta \, g_{\alpha - \beta}, \qquad |\alpha| \le N$$

TAX flattens chains of multiplications into a `ProductExpr` with a rolling accumulator using a single intermediate buffer.

### Division

$$f / g = f \cdot g^{-1}$$

Computed via the reciprocal series (see below).

### Negation

$$(-f)_\alpha = -f_\alpha$$

### Scalar Operations

For a scalar $s \in \mathbb{R}$:

$$(f + s)_\alpha = f_\alpha + s \, \delta_{|\alpha|,\, 0}, \qquad (s \cdot f)_\alpha = s \, f_\alpha$$

where $\delta$ is the Kronecker delta.

---

## Algebraic Operations

### Reciprocal

Given $g = 1/f$, the identity $f \cdot g = 1$ yields a degree-by-degree recurrence.

**Univariate:**

$$g_0 = \frac{1}{f_0}$$

$$g_d = -\frac{1}{f_0} \sum_{k=1}^{d} f_k \, g_{d-k}, \qquad d \ge 1$$

**Multivariate:**

$$g_\alpha = \frac{1}{f_0}\left(\delta_{|\alpha|,\,0} - \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} f_\beta \, g_{\alpha - \beta}\right)$$

Requires $f_0 \neq 0$.

### Square

$$(f^2)_\alpha = (f \cdot f)_\alpha$$

Computed directly as a Cauchy self-product.

### Cube

**Univariate:**

$$(f^3)_d = \sum_{j=0}^{d} \sum_{k=0}^{d-j} f_j \, f_k \, f_{d-j-k}$$

**Multivariate:** computed as two successive Cauchy products, $f^3 = (f \cdot f) \cdot f$.

### Square Root

Given $g = \sqrt{f}$, the identity $g^2 = f$ yields:

**Univariate:**

$$g_0 = \sqrt{f_0}$$

$$g_d = \frac{1}{2\,g_0} \left(f_d - \sum_{k=1}^{d-1} g_k \, g_{d-k}\right), \qquad d \ge 1$$

**Multivariate:**

$$g_\alpha = \frac{1}{2\,g_0} \left(f_\alpha - \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le d-1}} g_\beta \, g_{\alpha - \beta}\right), \qquad d = |\alpha|$$

Requires $f_0 > 0$.

### Absolute Value

$$|f|_\alpha = \operatorname{sgn}(f_0) \cdot f_\alpha$$

If $f_0 > 0$ the series is unchanged; if $f_0 < 0$ all coefficients are negated. Requires $f_0 \neq 0$.

### Compositional Inverse (`inv`)

Given a univariate polynomial $f(x) = \sum_{d=1}^{N} f_d\,x^d$ with $f_0 = 0$ and $f_1 \neq 0$, the compositional inverse $g = \operatorname{inv}(f)$ is the unique series $g(y) = \sum_{d=1}^{N} g_d\,y^d$ satisfying:

$$f(g(y)) = y$$

The algorithm builds $g$ degree by degree, tracking the powers $g^k$ incrementally.

**Initialisation:**

$$g_1 = \frac{1}{f_1}$$

**Power tracking:**

Let $P_d^{(k)}$ denote the coefficient of $y^d$ in $g(y)^k$. These are built by the Cauchy recurrence:

$$P_d^{(1)} = g_d, \qquad P_d^{(k)} = \sum_{j=1}^{d-1} P_{d-j}^{(k-1)}\,g_j, \qquad k \ge 2$$

Note that $P_d^{(k)}$ for $k \ge 2$ depends only on $g_1, \ldots, g_{d-1}$ (since $g_0 = 0$ eliminates the $j = d$ and $j = 0$ boundary terms).

**Degree-by-degree solve ($d \ge 2$):**

Expanding $f(g(y)) = \sum_{k=1}^{N} f_k\,g(y)^k$ and equating the coefficient of $y^d$ to zero (for $d \ge 2$):

$$\sum_{k=1}^{d} f_k\,P_d^{(k)} = 0$$

Separating the $k = 1$ term (which contributes $f_1\,g_d$):

$$f_1\,g_d + \underbrace{\sum_{k=2}^{\min(d,\,N)} f_k\,P_d^{(k)}}_{S_d} = 0$$

Since $P_d^{(k)}$ for $k \ge 2$ uses only previously computed coefficients, we solve:

$$g_d = -\frac{S_d}{f_1}$$

**Complexity:** $O(N^3)$ — at each degree $d$, updating the power table costs $O(d^2)$ work, summed over $d = 2, \ldots, N$.

Requires $f_0 = 0$ and $f_1 \neq 0$. Currently univariate only ($M = 1$).

---

## Trigonometric Functions

### Sine and Cosine (coupled recurrence)

Let $s = \sin(f)$ and $c = \cos(f)$. The chain rule gives the coupled system:

$$s' = f' \cdot c, \qquad c' = -f' \cdot s$$

Translating into coefficient recurrences via the identity $d \, h_d = \sum_{k=0}^{d-1}(d-k)\,a_{d-k}\,h_k$ for $h = \varphi(a)$:

**Univariate:**

$$s_0 = \sin(f_0), \qquad c_0 = \cos(f_0)$$

$$s_d = \frac{1}{d}\sum_{k=0}^{d-1}(d - k)\,f_{d-k}\,c_k$$

$$c_d = -\frac{1}{d}\sum_{k=0}^{d-1}(d - k)\,f_{d-k}\,s_k$$

**Multivariate ($d = |\alpha|$):**

$$s_\alpha = \frac{1}{d}\sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| \le d-1}} (d - |\beta|)\,f_{\alpha-\beta}\,c_\beta$$

$$c_\alpha = -\frac{1}{d}\sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| \le d-1}} (d - |\beta|)\,f_{\alpha-\beta}\,s_\beta$$

Both sine and cosine are always computed together since the recurrence is coupled.

### Tangent

Defined as $t = \tan(f) = \sin(f)/\cos(f)$. Rather than dividing two series, TAX solves the linear system $c \cdot t = s$ degree by degree:

$$t_0 = \frac{s_0}{c_0} = \frac{\sin(f_0)}{\cos(f_0)}$$

$$t_d = \frac{1}{c_0}\left(s_d - \sum_{k=1}^{d} c_k \, t_{d-k}\right), \qquad d \ge 1$$

where $s$ and $c$ are the sine and cosine series of $f$.

### Inverse Sine

Let $g = \arcsin(f)$. From $g' = f' / \sqrt{1 - f^2}$, define the helper $h = \sqrt{1 - f^2}$ and rewrite as $h \cdot g' = f'$.

**Univariate:**

$$g_0 = \arcsin(f_0)$$

$$g_d = \frac{1}{h_0}\left(f_d - \frac{1}{d}\sum_{k=1}^{d-1} k\,h_{d-k}\,g_k\right), \qquad d \ge 1$$

**Multivariate ($d = |\alpha|$):**

$$g_\alpha = \frac{1}{h_0}\left(f_\alpha - \frac{1}{d}\sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le d-1}} (d - |\beta|)\,h_\beta\,g_{\alpha-\beta}\right)$$

### Inverse Cosine

$$\arccos(f) = \frac{\pi}{2} - \arcsin(f)$$

Computed by negating the arcsin series and adding $\pi/2$ to the constant term.

### Inverse Tangent

Let $g = \arctan(f)$. From $g' = f'/(1 + f^2)$, define $h = 1 + f^2$ and solve $h \cdot g' = f'$.

**Univariate:**

$$g_0 = \arctan(f_0)$$

$$g_d = \frac{1}{h_0}\left(f_d - \frac{1}{d}\sum_{k=1}^{d-1} k\,h_{d-k}\,g_k\right), \qquad d \ge 1$$

**Multivariate ($d = |\alpha|$):**

$$g_\alpha = \frac{1}{h_0}\left(f_\alpha - \frac{1}{d}\sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le d-1}} (d - |\beta|)\,h_\beta\,g_{\alpha-\beta}\right)$$

### Two-argument Inverse Tangent

Given $g = \operatorname{atan2}(y, x)$, the total derivative is:

$$g' = \frac{x\,y' - y\,x'}{x^2 + y^2}$$

Define $h = x^2 + y^2$ (via Cauchy products). Rearranging $h \cdot g' = x\,y' - y\,x'$:

**Univariate:**

$$g_0 = \operatorname{atan2}(y_0,\, x_0)$$

$$g_d = \frac{1}{d\,h_0}\left(d\,(x_0\,y_d - y_0\,x_d) + \sum_{k=1}^{d-1} k\,(x_{d-k}\,y_k - y_{d-k}\,x_k - h_{d-k}\,g_k)\right)$$

**Multivariate ($d = |\alpha|$):**

$$g_\alpha = \frac{1}{h_0}\left((x_0\,y_\alpha - y_0\,x_\alpha) + \frac{1}{d}\sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le d-1}} (d - |\beta|)\,(x_\beta\,y_{\alpha-\beta} - y_\beta\,x_{\alpha-\beta} - h_\beta\,g_{\alpha-\beta})\right)$$

---

## Hyperbolic Functions

### Sinh and Cosh (coupled recurrence)

Let $\mathit{sh} = \sinh(f)$ and $\mathit{ch} = \cosh(f)$. The coupled system is:

$$\mathit{sh}' = f' \cdot \mathit{ch}, \qquad \mathit{ch}' = f' \cdot \mathit{sh}$$

Note the **positive** sign on $\mathit{ch}'$ (contrast with $\cos' = -f' \cdot \sin$).

**Univariate:**

$$\mathit{sh}_0 = \sinh(f_0), \qquad \mathit{ch}_0 = \cosh(f_0)$$

$$\mathit{sh}_d = \frac{1}{d}\sum_{k=0}^{d-1}(d-k)\,f_{d-k}\,\mathit{ch}_k$$

$$\mathit{ch}_d = \frac{1}{d}\sum_{k=0}^{d-1}(d-k)\,f_{d-k}\,\mathit{sh}_k$$

### Tanh

Solved from $\cosh(f) \cdot t = \sinh(f)$ degree by degree, analogous to tangent:

$$t_0 = \frac{\sinh(f_0)}{\cosh(f_0)}$$

$$t_d = \frac{1}{\mathit{ch}_0}\left(\mathit{sh}_d - \sum_{k=1}^{d} \mathit{ch}_k \, t_{d-k}\right), \qquad d \ge 1$$

---

## Transcendental Functions

### Exponential

Let $g = \exp(f)$. From $g' = f' \cdot g$:

**Univariate:**

$$g_0 = \exp(f_0)$$

$$g_d = \frac{1}{d}\sum_{k=0}^{d-1}(d-k)\,f_{d-k}\,g_k$$

**Multivariate ($d = |\alpha|$):**

$$g_\alpha = \frac{1}{d}\sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le d}} |\beta|\,f_\beta\,g_{\alpha-\beta}$$

### Natural Logarithm

Let $g = \ln(f)$. From $f \cdot g' = f'$:

**Univariate:**

$$g_0 = \ln(f_0)$$

$$g_d = \frac{1}{f_0}\left(f_d - \frac{1}{d}\sum_{k=1}^{d-1} k\,f_{d-k}\,g_k\right), \qquad d \ge 1$$

**Multivariate ($d = |\alpha|$):**

$$g_\alpha = \frac{1}{f_0}\left(f_\alpha - \frac{1}{d}\sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le d-1}} (d - |\beta|)\,f_\beta\,g_{\alpha-\beta}\right)$$

Requires $f_0 > 0$.

### Base-10 Logarithm

$$\log_{10}(f) = \frac{\ln(f)}{\ln(10)}$$

Computed by applying the natural logarithm series and scaling all coefficients by $1/\ln(10)$.

---

## Power Functions

### Integer Power (`ipow`)

For integer exponent $n \in \mathbb{Z}$:

| Case      | Method                                      |
|-----------|---------------------------------------------|
| $n = 0$   | $g = 1$                                     |
| $n = 1$   | $g = f$                                     |
| $n = -1$  | $g = f^{-1}$ (reciprocal series)            |
| $n < -1$  | $g = (f^{-1})^{\lvert n\rvert}$             |
| $n \ge 2$ | Binary exponentiation with Cauchy products  |

**Binary exponentiation** computes $f^n$ using $O(\log n)$ Cauchy products, all truncated at degree $N$:

$$r \leftarrow 1, \quad b \leftarrow f$$

$$\text{while } n > 0: \quad \begin{cases} r \leftarrow r \cdot b & \text{if } n \text{ is odd} \\ b \leftarrow b \cdot b, \quad n \leftarrow \lfloor n/2 \rfloor \end{cases}$$

### Real Power (`dpow`)

For $c \in \mathbb{R}$, let $g = f^c$. The identity $f \cdot g' = c\,f'\cdot g$ gives:

**Univariate:**

$$g_0 = f_0^{\,c}$$

$$g_d = \frac{1}{d\,f_0}\sum_{k=0}^{d-1}\bigl(c\,(d-k) - k\bigr)\,f_{d-k}\,g_k$$

**Multivariate ($d = |\alpha|$):**

$$g_\alpha = \frac{1}{d\,f_0}\sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le d}}\bigl(c\,|\beta| - (d - |\beta|)\bigr)\,f_\beta\,g_{\alpha-\beta}$$

Requires $f_0 > 0$ in general (or $f_0 \neq 0$ when $c$ is a non-negative integer).

### DA Power (`tpow`)

When both base and exponent are DA objects, $f^g$ is computed by composition:

$$f^g = \exp\!\bigl(g \cdot \ln(f)\bigr)$$

This chains the logarithm, Cauchy product, and exponential series.

---

## Special Functions

### Error Function

Let $g = \operatorname{erf}(f)$. The derivative of the error function is:

$$\operatorname{erf}'(x) = \frac{2}{\sqrt{\pi}}\,e^{-x^2}$$

The algorithm first computes the helper series:

$$h = \frac{2}{\sqrt{\pi}}\,\exp\!\left(-f^2\right)$$

and then solves $g' = f' \cdot h$ using the same recurrence structure as the exponential:

**Univariate:**

$$g_0 = \operatorname{erf}(f_0)$$

$$g_d = \frac{1}{d}\sum_{k=0}^{d-1}(d-k)\,f_{d-k}\,h_k$$

**Multivariate ($d = |\alpha|$):**

$$g_\alpha = \frac{1}{d}\sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le d}} |\beta|\,f_\beta\,h_{\alpha-\beta}$$

---

## Distance Functions

### Hypot (2-argument)

$$\operatorname{hypot}(x, y) = \sqrt{x^2 + y^2}$$

Computed by forming the Cauchy self-products $x^2$ and $y^2$, summing coefficient-wise, and applying the square root series.

### Hypot (3-argument)

$$\operatorname{hypot}(x, y, z) = \sqrt{x^2 + y^2 + z^2}$$

Same approach with three squared terms.

---

## Multivariate Generalisation

All univariate recurrences generalise to $M$ variables by replacing:

- The scalar index $d$ with a multi-index $\alpha = (\alpha_1, \ldots, \alpha_M) \in \mathbb{N}_0^M$ of total degree $d = |\alpha|$.
- Inner sums $\sum_{k=a}^{b}$ with sums over sub-multi-indices $\sum_{\beta \le \alpha}$ restricted to appropriate degree ranges.

Coefficients are stored in **graded lexicographic (grlex) order**: first grouped by total degree $|\alpha|$, then ordered lexicographically within each degree group.

The total number of coefficients is:

$$\binom{N + M}{M} = \frac{(N + M)!}{N!\,M!}$$

| $N$ | $M$ | Coefficients  |
|-----|-----|---------------|
| 2   | 1   | 3             |
| 2   | 2   | 6             |
| 3   | 2   | 10            |
| 3   | 3   | 20            |
| 5   | 2   | 21            |
