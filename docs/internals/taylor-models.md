# Taylor Models

How `tax::model` implements Makino's remainder-enhanced DA (PhD thesis
MSUCL-1093, 1998, ch. 4), and where it deliberately deviates. Headers:
`tax/model/interval.hpp`, `taylor_model.hpp`, `arithmetic.hpp`, `math.hpp`;
facade `tax/model.hpp`.

A Taylor model of order $N$ is the quadruple
$T_f = (P_f, I_f, \mathbf{x}_0, [\mathbf{a},\mathbf{b}])$ with the invariant

$$
f(\mathbf{x}) \in P_f(\mathbf{x}-\mathbf{x}_0) + I_f
\qquad \forall\, \mathbf{x} \in [\mathbf{a},\mathbf{b}] .
$$

Every operation below is a constructive proof that its result satisfies this
invariant, given that its inputs do. $B(\cdot)$ denotes a rigorous range
bound of a polynomial over the displacement domain
$D_i = [a_i - x_{0i},\, b_i - x_{0i}]$.

---

## Data model

`TaylorModel<T, N, M>` stores:

| Field | Type | Content |
|---|---|---|
| `poly_` | `TaylorExpansion<T, IsotropicScheme<N,M>, Dense>` | $P$, in displacement coordinates |
| `rem_`  | `Interval<T>` | $I$ |
| `x0_`   | `std::array<T, M>` | $\mathbf{x}_0$ |
| `dom_`  | `std::array<Interval<T>, M>` | $[\mathbf{a},\mathbf{b}]$, absolute coordinates |

The polynomial part reuses the dense core unchanged — graded-lex layout,
`constexpr` kernels, no allocation. Binary operations demand bitwise-equal
`x0_`/`dom_` (thesis: operations are only defined for a common expansion
parameter $\alpha$) and throw `std::invalid_argument` otherwise. COSY's `RD`
additionally stores per-order bound intervals $I^0 \dots I^n$; we recompute
them on demand instead — see [Chapter 5](#what-chapter-5-adds) below.

## Outward-rounded intervals

Rigor under IEEE-754 requires every computed interval endpoint to be rounded
*away* from the exact result. Toggling the FPU rounding mode is neither
`constexpr` nor thread-friendly, so `Interval<T>` instead evaluates in the
default round-to-nearest mode and widens after the fact:

- `nextUp` / `nextDown` are implemented bit-wise via `std::bit_cast`
  (increment/decrement of the payload with sign handling), fully `constexpr`.
- **Arithmetic** (`+ - * /`, `pointPow`): result endpoints are within
  ½ ulp of exact, so **1 ulp** outward per side guarantees enclosure.
- **libm endpoints** (`exp log sin cos sinh cosh` of the endpoint values):
  padded **2 ulps**, assuming a faithful (≤ 1 ulp) libm. `sqrt` is correctly
  rounded by IEEE, so 1 ulp suffices.
- **Exact values are never padded**: constructor inputs, and the 0 lower
  bound of `sqr`/even `pow` (thesis eq. (5.4) — $x^2 \ge 0$ holds exactly).

Multiplication/division take the min/max over the four endpoint products or
quotients before padding. `sqr(x)` and even `pow(x, n)` use
mignitude/magnitude so that e.g. $[-1,2]^2 = [0,4]$ rather than the
dependency-poisoned $[-2,4]$ of `x * x`; endpoint powers are computed by
outward-rounded binary exponentiation on point intervals (`pointPow`), which
stays `constexpr` and avoids trusting `std::pow`.

### sin/cos enclosures

`sin(I)`/`cos(I)` must decide whether an extremum lies inside `I`. The test
"does $c + 2\pi k$ hit $I$ for some integer $k$" is performed *in interval
arithmetic*: compute the enclosure $t \supseteq (I - c)/2\pi$ (with $\pi$
itself an interval, `Interval::padded(pi_v)`) and report true iff
$\lfloor t_{hi} \rfloor \ge \lceil t_{lo} \rceil$. Because $t$ is a superset,
the test can only err toward "contains an extremum", which merely widens the
result to $\pm 1$ — never an invalid enclosure. Endpoint values are padded
libm calls; both endpoints are monotonically clamped into $[-1,1]$. Widths
of a full period or more short-circuit to $[-1,1]$.

## Range bounding

`detail::polyRangeBound(P, pows)` is the naive order-sum bounder (thesis
§5.4, "algorithm 0/1"): the interval sum over all monomials of
$c_\alpha \prod_i D_i^{\alpha_i}$, where the power table
`DomainPowers<T, M, P>` precomputes $D_i^j$ via `pow` (so even powers keep
their sharp lower bound — this is what makes bounds of even monomials like
$dx^2$ come out as $[0, d^2]$, not $[-d^2, d^2]$).

Properties exploited:

- degree-0 and degree-1 parts are bounded **exactly** (interval evaluation of
  a linear function over a box is exact);
- graded-lex keeps each degree in one contiguous flat-index block
  $[\binom{d-1+M}{M}, \binom{d+M}{M})$, so the per-order bounds $I^k$
  (`orderBound(k)`) and the top-order block (needed by antiderivation) are
  single-range loops;
- multi-indices come from a `constexpr` `MultiIndexTable<N, M>` — no runtime
  `unflatIndex` calls in the loops.

This bounder is deliberately **pluggable-simple**; the sharper bounders of
thesis §5.4.3 are future work (see below). All bound consumers overestimate
gracefully: a wider $B(P)$ only widens remainders, never invalidates them.

## Multiplication

For $f \in P_a + I_a$, $g \in P_b + I_b$ (pointwise: $f = P_a + e_a$ with
$e_a \in I_a$):

$$
fg = \underbrace{(P_a P_b)_{\le N}}_{\text{polynomial part}}
   + \underbrace{(P_a P_b)_{> N}}_{\text{excess}}
   + P_a e_b + P_b e_a + e_a e_b .
$$

The truncated product $(P_a P_b)_{\le N}$ is computed by the existing dense
Cauchy kernels (unroll/stencil dispatch — the hot path is shared with `TE`).
The remainder becomes

$$
I_{fg} \;=\; B\big((P_a P_b)_{>N}\big) \;+\; B(P_a)\, I_b \;+\; B(P_b)\, I_a
\;+\; I_a I_b .
$$

`detail::excessProductBound` bounds the excess **without forming the
degree-$2N$ product**: it enumerates coefficient pairs
$(\alpha, \beta)$ with $|\alpha| + |\beta| > N$ and accumulates the interval
of $c_\alpha c_\beta \prod_i D_i^{\alpha_i+\beta_i}$ (powers up to $2N$ from
the same table). The graded-lex degree blocks make the inner loop start
directly at the first partner degree $> N - |\alpha|$ — pairs that cannot
exceed order $N$ are never visited. Cost is $O(n_{\text{coeff}}^2 \cdot M)$
in the worst case, with zero-coefficient skips.

`square(f)` is a dedicated operation because $(P + e)^2 = P^2 + 2Pe + e^2$
allows $I^2$ to use the sharp `sqr` rule (5.4):
$I_{f^2} = B((P^2)_{>N}) + 2B(P)I + \operatorname{sqr}(I)$.

### Scalar and interval operands

For an exact scalar $s$: polynomial and remainder scale directly. For an
*unknown constant* $s \in J$ (an `Interval` operand), the midpoint
$m = \operatorname{mid}(J)$ goes into the polynomial and the residual is
bounded:

$$
s f = m P + (s - m) P + s e
\;\subseteq\; m P + (J - m)\,B(P) + J\,I .
$$

Addition of $J$ is the special case $mP$ → $P + m$,
remainder $+\; (J - m)$.

## Intrinsic functions

All intrinsics share one two-step recipe (thesis §4.3.2, implemented as in
§5.3.3). Split $f = c + \bar f$ where $c$ is the constant coefficient, so
$\bar f$ has Taylor model $(P - c,\, I)$ and

$$
A = B(P - c) + I \;\supseteq\; \bar f([\mathbf a, \mathbf b]), \qquad
\Theta = \operatorname{hull}(0, A) \;\supseteq\; \theta \bar f, \qquad
W = c + \Theta ,
$$

with $\theta \in (0,1)$ the Lagrange parameter. Then:

1. **Series.** Evaluate
   $\sum_{k=0}^{N} a_k \bar f^{\,k}$, $a_k = g^{(k)}(c)/k!$, by **Horner's
   scheme in Taylor-model arithmetic**. Each multiplication is a full TM
   multiplication, so the truncation excess and remainder cross terms of the
   series polynomial itself (the $I^R_{N,poly}$ of eq. (5.5)) accumulate
   automatically.
2. **Lagrange tail.** Add the interval enclosure of
   $\dfrac{g^{(N+1)}(c + \theta\bar f)}{(N+1)!}\, \bar f^{\,N+1}
   \;\subseteq\; \dfrac{g^{(N+1)}(W)}{(N+1)!}\, A^{N+1}$,
   evaluated entirely in outward-rounded interval arithmetic
   (the $1/k!$ factors via an interval `invFactorial`).

Domain conditions are certified on $W$, which encloses every point
$c + \theta\bar f(\mathbf x)$ the Lagrange form can sample; failures throw
`std::domain_error`.

Per-function series coefficients (recursions, evaluated in `T`) and Lagrange
factors (evaluated in `Interval<T>`):

| $g$ | coefficients $a_k$ | Lagrange enclosure added to $I$ |
|---|---|---|
| `exp` | $a_k = a_{k-1}/k$, $a_0 = e^c$ | $e^c \frac{A^{N+1}}{(N+1)!} \exp(\Theta)$ — eq. (4.10) |
| `log` | $a_k = \frac{(-1)^{k+1}}{k c^k}$ | $\frac{(-1)^{N}}{N+1} \left(\frac{A}{c}\right)^{N+1} V^{-(N+1)}$ |
| `reciprocal` | $a_k = \frac{(-1)^k}{c^{k+1}}$ | $(-1)^{N+1} \left(\frac{A}{c}\right)^{N+1} \frac{1}{c}\, V^{-(N+2)}$ — eq. (4.11) |
| `sqrt` | $a_k = a_{k-1} \cdot \frac{-(2k-3)}{2kc}$, $a_1 = \frac{\sqrt c}{2c}$ | coefficient recursion continued one step *in intervals* × $A^{N+1} V^{-(N+\frac12)}$ |
| `isqrt` | $a_k = a_{k-1} \cdot \frac{-(2k-1)}{2kc}$ | same, × $A^{N+1} V^{-(N+\frac32)}$ |
| `sin`, `cos` | derivative 4-cycle $\{\pm\sin c, \pm\cos c\}/k!$ | $\frac{A^{N+1}}{(N+1)!}\, g^{(N+1)}(W)$ via interval `sin`/`cos` |
| `sinh`, `cosh` | 2-cycle $\{\sinh c, \cosh c\}/k!$ | $\frac{A^{N+1}}{(N+1)!}\, g^{(N+1)}(W)$ via interval `sinh`/`cosh` |
| `asin` | derivative recursion at $c$ (below) | same recursion over $W$ in intervals |
| `atan` | $a_k = \frac{\cos^k\!\varphi\, \sin k(\varphi + \frac\pi2)}{k}$, $\varphi = \arctan c$ | $\frac{A^{N+1}}{N+1} [-1, 1]$ (since $|{\arctan}^{(N+1)}| \le N!$) |

with $V = 1 + \Theta/c \supseteq (c + \theta\bar f)/c$, guaranteed positive
by the respective domain checks. For `sqrt`/`isqrt` the Lagrange magnitude
$\frac{(2N\mp1)!!}{(N+1)!\,2^{N+1}}$ is obtained by simply continuing the
coefficient recursion one extra step in interval arithmetic — no factorials
are ever formed. `asin` uses the thesis recursion

$$
\arcsin^{(k+2)}(a) = \frac{(2k+1)\, a \arcsin^{(k+1)}(a) + k^2 \arcsin^{(k)}(a)}{1-a^2},
$$

once at the point $c$ (scalar, for the coefficients) and once over the
interval $W$ (for the tail). `tan = sin · reciprocal(cos)`,
`tanh = sinh · reciprocal(cosh)`, `acos = π/2 − asin`, `pow(f, n)` by binary
exponentiation over `square`/`*`, and `f / g = f · reciprocal(g)`.

!!! note "Deviation from the thesis: inverse trigonometrics"
    The thesis computes `asin`/`atan` through addition formulas
    ($\arcsin f = \arcsin c + \arcsin(f\sqrt{1-c^2} - c\sqrt{1-f^2})$, …),
    which re-center the series at 0 and are sharper for large $c$. But those
    identities only hold under branch conditions the thesis leaves implicit
    (e.g. $\arcsin a - \arcsin b$ equals the formula only when it lands in
    $[-\pi/2, \pi/2]$); violating them silently would produce a *wrong
    enclosure*. We use the direct Taylor expansion at $c$ with the same
    derivative recursions — unconditionally correct on the checked domain
    ($W \subset (-1,1)$ for `asin`/`acos`; none for `atan`), at the price of
    somewhat wider remainders near the domain edges.

## Antiderivation

Thesis eq. (4.12): for $\partial_i^{-1} f = \int_{x_{0i}}^{x_i} f$, the
polynomial part integrates $P_{N-1}$ (the core `TE::integ` already drops
exactly the monomials that would exceed order $N$, i.e. the order-$N$
block), and the freed top order joins the remainder:

$$
\partial_i^{-1}(P, I) = \Big( \textstyle\int_0^{dx_i} P_{N-1},\;
\big(B(P_N - P_{N-1}) + I\big) \cdot \operatorname{hull}(0, D_i) \Big).
$$

$B(P_N - P_{N-1})$ is the single top-degree block bound (`orderBound(N)`).
We multiply by $\operatorname{hull}(0, D_i)$ rather than the thesis's scalar
width $b_i - a_i$: the integration path runs from $0$ to $dx_i \in D_i$, so
$\int_0^{dx_i} g$ with $g(t) \in J$ lies in $J \cdot \operatorname{hull}(0,
D_i)$ — rigorous and at least as tight. Definite integrals follow eq. (4.13)
by evaluating the antiderivative at displacement endpoints (multidimensional
boxes by inclusion–exclusion over corners; see
`tests/model/test_thesis_examples.cpp`, §5.5.2).

## Rigor contract

Two different arithmetic layers coexist, with an explicit boundary:

- **Conservative:** everything computed as an `Interval` — range bounds,
  excess bounds, remainder propagation, Lagrange tails, domain checks,
  antiderivation scaling. Outward rounding makes these true enclosures under
  IEEE-754 with a faithful libm.
- **Not swept into the remainder:** rounding of the polynomial
  *coefficients* themselves (kernel Cauchy products, series-coefficient
  recursions in `T`, `acos`'s $\pi/2$, …) — roughly 1 ulp per coefficient
  operation. The invariant is exact in exact coefficient arithmetic.

Consequence: remainder widths reported by `tax` are honest for any practical
domain width, but the module does not yet constitute a formal
proof-of-enclosure down to the last ulp. Closing that gap is precisely the
coefficient-error tallying of thesis ch. 5 (COSY sweeps each floating-point
coefficient error into the remainder as it happens). Any new code feeding a
remainder **must** stay on the `Interval` side of this line.

## Complexity

For $n = \binom{N+M}{M}$ coefficients: TM multiplication is one dense Cauchy
product plus an $O(n^2 M)$ excess-bound sweep and two $O(nM)$ range bounds.
Each intrinsic is $N$ TM multiplications (Horner) plus $O(N)$ interval
operations for coefficients and tail. Power tables cost $O(M N \log N)$
interval multiplications per operation. Nothing allocates; all tables are
`constexpr` or stack-resident.

---

## What chapter 5 adds

Chapter 4 is the *mathematics* of Taylor models; chapter 5 is their
*production implementation* as COSY INFINITY's `RD` type. Relative to what
`tax::model` implements today, ch. 5 adds four things:

1. **Cached per-order bound intervals.** `RD` stores $I^0, \dots, I^n$
   alongside the remainder (Table 5.4), so $B(\bar P) = \sum_k I^k$ is a free
   lookup inside every multiplication and intrinsic, and squaring can use
   the order-by-order formula of §5.3.2. We recompute bounds on demand
   ($O(nM)$ per call) — same results, more arithmetic per operation.
   *Verdict: a pure performance optimization; worth doing if profiling shows
   bound computation dominating (likely for high-order/many-variable models
   inside hot loops such as verified ODE stepping). It's a mechanical change:
   carry a `std::array<Interval<T>, N+1>` invalidated by polynomial
   mutation.*

2. **Sharper polynomial bounders** (§5.4): the tightening-algorithm ladder
   (naive sums → default alg. 1 → `POLVAL`/Horner → scanning), the **exact
   quadratic bounder** — solving $\nabla(\text{order} \le 2\ \text{part}) =
   0$ exactly, recursing over boundary faces — and the **iterative
   third-order refinement** (5.10). The thesis's own data (Table 5.6) shows
   why: the *remainder* is nearly bounder-independent, but the *total* bound
   improves ~7× (from $\pm 36$ to $\pm 4.8$) going from naive to
   tightened bounds. Our order-3 total bound for $1/x + x$ is 0.1514 vs the
   thesis's 0.14988 for the same reason.
   *Verdict: the highest-value upgrade. `bound()` sharpness is the product's
   visible quality metric, and it feeds every downstream consumer (domain
   checks, ADS-style splitting decisions, quadrature widths). The
   quadratic bounder is exact for the dominant low orders, moderate effort,
   and drops in behind `polyRangeBound` — which is why that function is
   documented as pluggable. The scanning algorithms (4/5) are heuristic
   cross-checks and not worth carrying.*

3. **Floating-point coefficient-error sweeping** — the `RD` implementation
   accounts for the rounding of every polynomial coefficient operation in
   the remainder, which is what upgrades "rigorous in exact coefficient
   arithmetic" to "rigorous, period".
   *Verdict: required if `tax::model` is ever used for computer-assisted
   proofs; unnecessary for engineering-grade verified numerics where the
   ~1-ulp-per-op coefficient noise is orders of magnitude below the
   truncation remainder. Implementing it means either interval coefficients
   or an error-magnitude side-channel through every kernel — significant,
   invasive, and best done after the bounder work.*

4. **Bookkeeping conveniences** (storage layout, `RDAVAR`-style
   constructors, accessor routines — §5.1): superseded by C++ types; nothing
   to port.

Suggested order if/when extending: (2) exact quadratic bounder behind a
pluggable bounder policy → (1) per-order bound caching → (3) coefficient
sweeping. The worked examples of both chapters are already pinned as tests
(`tests/model/test_thesis_examples.cpp`), so each upgrade can be measured
directly against Tables 4.2/4.3/5.6–5.8.
