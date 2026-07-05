# Taylor Models API (`tax::model`)

Reference for the interval and Taylor-model types. All names live in
`namespace tax::model`; `Interval`, `TaylorModel` and the alias `TM` are
re-exported under `tax`. Free functions (intrinsics, `hull`, `sqr`, …) are
found via ADL, or can be qualified as `tax::model::…`.

---

## `Interval<T>`

```cpp
namespace tax::model {

template <std::floating_point T>
class Interval;

}  // namespace tax::model
```

A closed interval $[lo, hi]$ with **outward-rounded** arithmetic: endpoints
*given* to a constructor are exact; endpoints *computed* by any operation are
widened by 1 ulp per side (2 ulps for libm-evaluated endpoints), so every
result is a guaranteed enclosure of the exact-arithmetic result. NaN
endpoints are unsupported. Arithmetic and integer powers are `constexpr`.

### Construction

| Signature | Description |
|---|---|
| `Interval()` | `[0, 0]` |
| `Interval(T v)` | point interval `[v, v]` (implicit) |
| `Interval(T lo, T hi)` | throws `std::invalid_argument` if `lo > hi` |
| `Interval::zero()` | `[0, 0]` |
| `Interval::padded(v, ulps = 1)` | `v` widened outward by `ulps` per side |
| `Interval::outward(lo, hi)` | endpoints widened by one ulp per side |

### Accessors

| Member | Description |
|---|---|
| `lower()`, `upper()` | endpoints |
| `mid()` | approximate midpoint |
| `width()` | upper bound of `hi - lo` |
| `mag()` | $\max\,\lvert x\rvert$ over the interval |
| `mig()` | $\min\,\lvert x\rvert$ (0 if the interval contains 0) |
| `contains(T)` / `contains(Interval)` | containment tests |
| `operator==` | exact endpoint equality |

### Arithmetic

`+`, `-` (binary and unary), `*`, `/`, compound assignments, each for
`Interval ⊕ Interval` and mixed `Interval ⊕ T` — all per thesis Table 4.1
with outward rounding. `operator/` throws `std::domain_error` when 0 lies in
the divisor.

### Free functions

| Function | Description |
|---|---|
| `hull(a, b)` | smallest interval containing both |
| `intersect(a, b)` | intersection; throws `std::domain_error` if disjoint |
| `sqr(x)` | interval square, sharp rule (5.4): never dips below 0 |
| `pow(x, int n)` | integer power; even powers keep the sharp non-negative lower bound; `n < 0` reciprocates (`constexpr`) |
| `exp(x)`, `log(x)`, `sqrt(x)` | monotone enclosures; `log`/`sqrt` throw `std::domain_error` on domain violation |
| `sinh(x)`, `cosh(x)` | monotone / even enclosures (`cosh` ≥ 1) |
| `sin(x)`, `cos(x)` | enclosures with detection of enclosed extrema; fall back to $[-1, 1]$ beyond one period |
| `operator<<` | streams `[lo, hi]` |

---

## `TaylorModel<T, N, M>`

```cpp
namespace tax::model {

template <std::floating_point T, int N, int M = 1>
    requires (N >= 0 && M >= 1)
class TaylorModel;

template <int N, int M = 1>
using TM = TaylorModel<double, N, M>;   // re-exported as tax::TM

}  // namespace tax::model
```

The quadruple $(P, I, \mathbf{x}_0, [\mathbf{a},\mathbf{b}])$ guaranteeing
$f(\mathbf{x}) \in P(\mathbf{x}-\mathbf{x}_0) + I$ on the domain. The
polynomial part is a dense `TaylorExpansion<T, IsotropicScheme<N,M>>` in the
displacement $d\mathbf{x} = \mathbf{x} - \mathbf{x}_0$.

### Associated types & constants

| Member | Description |
|---|---|
| `Poly` | `TaylorExpansion<T, IsotropicScheme<N,M>, storage::Dense>` |
| `Input` | `std::array<T, M>` — displacement vector |
| `Point` | `std::array<T, M>` — expansion point |
| `Domain` | `std::array<Interval<T>, M>` — domain box |
| `interval_type`, `scalar_type`, `scheme` | `Interval<T>`, `T`, the index scheme |
| `order_v`, `vars_v`, `nCoefficients` | $N$, $M$, $\binom{N+M}{M}$ |

### Construction

| Signature | Description |
|---|---|
| `TaylorModel()` | zero function on the degenerate domain $\{0\}^M$ |
| `TaylorModel(Poly, Interval, Point, Domain)` | full constructor; throws `std::invalid_argument` if $\mathbf{x}_0 \notin [\mathbf{a},\mathbf{b}]$ |
| `constant(v, x0, dom)` | exact constant, remainder `[0, 0]` |
| `variable(x0, dom)` | univariate identity (`M == 1`, `N >= 1`) |
| `variable<I>(x0, dom)` | coordinate $x_I$ (compile-time index) |
| `variable(x0, dom, i)` | runtime index; throws `std::out_of_range` |
| `variables(x0, dom)` | `std::array<TaylorModel, M>` of all coordinates |

### Accessors & bounds

| Member | Description |
|---|---|
| `polynomial()` | polynomial part (const and mutable) |
| `remainder()` | remainder interval (const and mutable) |
| `expansionPoint()`, `domain()` | $\mathbf{x}_0$, $[\mathbf{a},\mathbf{b}]$ |
| `displacementDomain()` | $D_i = [a_i - x_{0i},\, b_i - x_{0i}]$; always contains 0 |
| `value()` | constant coefficient (COSY `CONS`) |
| `polynomialBound()` | rigorous $B(P)$ over the domain |
| `orderBound(k)` | per-order bound $I^k$ of the degree-$k$ homogeneous part |
| `bound()` | total enclosure $B(P) + I$ (COSY `IN`) |
| `eval(dx)` | `Interval` enclosure of $f(\mathbf{x}_0 + d\mathbf{x})$; throws `std::domain_error` outside the domain |
| `compatibleWith(g)` | same expansion point and domain? |
| `integ<I>()` / `integ(i)` | antiderivative Taylor model w.r.t. $x_I$, eq. (4.12) |

### Operators

All binary Taylor-model operators require compatible operands
(`std::invalid_argument` otherwise).

| Form | Notes |
|---|---|
| `±f`, `f ± g`, `f ± s`, `s ± f`, `f ± J` | `s` scalar, `J` an `Interval` (unknown constant in `J`) |
| `f * g` | truncated Cauchy product; degree-$>N$ excess + $B(P_f)I_g + B(P_g)I_f + I_f I_g$ into the remainder |
| `f * s`, `s * f`, `f / s`, `f * J`, `J * f` | scalar / interval scaling |
| `f / g`, `s / f` | via `reciprocal`; throws `std::domain_error` if the divisor's enclosure contains 0 |
| `+=`, `-=`, `*=`, `/=` | for every right-hand side accepted by the binary form |

### Intrinsics

Each follows the §4.3.2 recipe (constant-part split → Horner series in
Taylor-model arithmetic → Lagrange remainder over the argument enclosure) and
throws `std::domain_error` when its domain condition cannot be certified:

| Function | Domain condition on the enclosure $W$ |
|---|---|
| `exp`, `sin`, `cos`, `sinh`, `cosh`, `atan` | none |
| `log`, `sqrt`, `isqrt` | $W \subset (0, \infty)$ |
| `reciprocal` | $0 \notin W$ |
| `asin`, `acos` | $W \subset (-1, 1)$ |
| `tan`, `tanh` | via `sin/cos`, `sinh/cosh` — the division checks its own condition |
| `square` | none (dedicated sharp remainder, (5.4)) |
| `pow(f, int n)` | `n < 0` requires $0 \notin W$ |

### Exceptions summary

| Exception | Thrown by |
|---|---|
| `std::invalid_argument` | incompatible operands; expansion point outside domain; `Interval(lo, hi)` with `lo > hi` |
| `std::domain_error` | intrinsic domain violations; interval division by 0-containing interval; `eval` outside the domain; disjoint `intersect` |
| `std::out_of_range` | runtime-index `variable(x0, dom, i)` / `integ(i)` with invalid `i` |

The remainder formulas and the rounding contract are documented in
[Internals / Taylor Models](../internals/taylor-models.md).
