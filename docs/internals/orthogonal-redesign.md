# Orthogonal-polynomial redesign: the `Basis` policy

## Motivation

`tax` was born as a *Taylor* expansion engine: it propagates a truncated
multivariate **monomial** expansion through arbitrary expressions. But the
machinery — store a coefficient vector, push it through a recurrence per
operation — is not special to the monomial basis. Any polynomial family
`{P_0, P_1, P_2, …}` works the same way; only three things change:

1. the **product** rule (how `P_i · P_j` re-expands in the family),
2. **evaluation** at a point,
3. the coefficient-space **derivative / integral**.

This redesign factors those three operations out of the carrier type into a
**`Basis` policy**, so the same storage and the same linear-space surface serve
Taylor, Chebyshev, and — by adding a policy — Legendre, Hermite, Laguerre, …

## The abstraction

```
f(x) = Σ_{k=0}^{N} c_k · P_k(x)
```

A *basis policy* is a stateless type providing a small static surface
(templated on coefficient type `T` and order `N`):

| member                  | meaning                                              |
|-------------------------|------------------------------------------------------|
| `name()`                | short identifier, for IO                             |
| `term(k)`               | pretty label of `P_k` (`"x^2"`, `"T_2"`, …)          |
| `product(out, a, b)`    | truncated bilinear product in the family             |
| `eval(c, x)`            | value of the expansion at `x`                        |
| `derivative(out, c)`    | coefficients of `f'` in the same family              |
| `integral(out, c)`      | coefficients of `∫f` (constant of integration 0)     |

The carrier is

```cpp
template < typename Basis, int N, typename T = double >
class tax::Series;     // aliases: TaylorSeries<N>, ChebyshevSeries<N>
```

It owns the `std::array<T, N+1>` and everything **basis-independent**:
`+`, `-`, scalar `*`/`/`, the compound assignments, and the two factories
`constant(v)` and `variable()`. Those two factories are basis-independent on
purpose: every classical family has `P_0 ≡ 1` and `P_1 ≡ x`, so a constant is
`v·P_0` (set `c_0`) and the identity map is `1·P_1` (set `c_1`) in *any*
conforming basis.

Everything that differs is one `Basis::…` call away, dispatched at compile time
with zero overhead.

## What ships now

### `TaylorBasis` — the monomial family `P_k = x^k`

Wired straight onto the **existing kernel layer**: the product calls the
library's `cauchyProduct` and the transcendental surface (`exp`, `sin`, `sqrt`,
…) delegates to the existing univariate `series*` recurrence kernels — a
`Series<TaylorBasis,N,T>` carries the exact same coefficient layout as
`IsotropicScheme<N,1>`. This proves the policy *wraps* the engine rather than
duplicating it: the new Taylor series gets the whole elementary-function
catalogue and the unrolled hot path for free.

### `ChebyshevBasis` — Chebyshev polynomials of the first kind `P_k = T_k`

Self-contained, in the plain (un-normalised) convention `f = Σ c_k T_k`:

- **product** via `T_i T_j = ½(T_{i+j} + T_{|i-j|})` (truncated);
- **evaluation** by the Clenshaw recurrence;
- **derivative / integral** by the closed-form coefficient recurrences,
  verified against analytic forms and as mutual inverses.

Plus two Chebyshev-specific capabilities with no Taylor analogue:

- **`chebyshevInterpolate<N>(f)`** — build the order-`N` near-best *uniform*
  approximation of an arbitrary callable on `[-1,1]` from its samples at the
  Chebyshev–Gauss–Lobatto nodes (discrete cosine sum). This is the canonical
  Chebyshev use case: global approximation, not a single-point jet.
- **exact basis conversion** `toChebyshev` / `toTaylor` — a degree-`N`
  polynomial has an exact image in either basis, so a function built with the
  Taylor recurrences can be moved losslessly into the Chebyshev basis (and
  back).

## Capability matrix

| operation                       | Taylor | Chebyshev |
|---------------------------------|:------:|:---------:|
| `+ - *` scalar, `+= -= *=`      |   ✓    |     ✓     |
| `Series * Series` (product)     |   ✓    |     ✓     |
| `eval`, `deriv`, `integ`        |   ✓    |     ✓     |
| `pow(f, n)` integer             |   ✓    |     ✓     |
| series `/`, `reciprocal`        |   ✓    |     ✓     |
| `square`, `cube` (exact)        |   ✓    |     ✓     |
| `exp, log, sin, sqrt, …`        |   ✓    |     ✓¹    |
| `pow(f, p)` real exponent       |   —    |     ✓¹    |
| function interpolation          |   —    |     ✓     |
| `toChebyshev` / `toTaylor`      |   ✓    |     ✓     |

¹ **How the two bases compose differently.** The Taylor surface composes `g(f)`
through the classical triangular ODE recurrences (reused from the existing
kernels), so it is exact-to-truncation and `constexpr`. The Chebyshev surface
has no such recurrence, so it composes the way a spectral library does — sample
`g(f(x))` at the Chebyshev–Gauss–Lobatto nodes and re-interpolate
(`detail::chebCompose`). That yields a *near-best uniform* approximation of
`g(f)` over `[-1,1]` (spectrally accurate for analytic `g`, `f`) at the cost of
being runtime rather than `constexpr`. Pure-algebraic Chebyshev ops (`*`,
`square`, `cube`) stay exact and `constexpr`.

## How the existing types fit

The legacy `tax::TaylorExpansion<T, Scheme, Storage>` is the *multivariate,
multi-storage* Taylor instance of this same idea — its `Scheme` already owns the
product (`Scheme::cauchyProduct`). The clean end-state is to let a `Scheme`
(layout) and a `Basis` (algebra) compose: `Expansion<T, Basis, Scheme,
Storage>`, with `TaylorExpansion` becoming `Expansion<T, TaylorBasis, …>`. This
slice deliberately lands the abstraction **additively** on the univariate dense
case so the full existing suite (44 tests) keeps passing untouched, then leaves
the wider unification as mechanical follow-up.

## Roadmap

- **More families**: Legendre, Hermite, Laguerre — each is one policy
  (recurrence-defined product + Gauss-quadrature evaluation/projection).
- **Transcendentals in any basis**: solve `y' = g'·y` etc. as a (banded) linear
  system in coefficient space — the basis already supplies the product and
  derivative operators the solve needs.
- **Domain mapping**: carry an affine `[a,b] → [-1,1]` map so Chebyshev models
  live on arbitrary intervals.
- **Compose with `Scheme`/`Storage`**: multivariate (tensor-product) and sparse
  orthogonal expansions; fold `TaylorExpansion` onto `Expansion<…, TaylorBasis,
  …>`.
- **Eigen / named axes**: extend `tax::la` and the named-axis surface to the
  basis-generic `Series`.
```
