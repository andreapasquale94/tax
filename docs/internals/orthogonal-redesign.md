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
| `name()` / `term(k)`           | short identifier and pretty `P_k` label, for IO              |
| `product<T,Scheme>(out,a,b)`   | truncated bilinear product in the family (tensored)         |
| `eval<T,Scheme>(c, x)`         | value of the expansion at the point vector `x`              |
| `derivative<T,Scheme>(out,c,axis)` | coefficients of `∂f/∂x_axis` in the same family        |
| `integral<T,Scheme>(out,c,axis)`   | coefficients of `∫f dx_axis` (constant 0)             |

The methods are templated on the **scheme** (the monomial layout), so a single
policy serves univariate and multivariate, isotropic and mixed-order. The
concept (`tax::Basis`) checks all four against a representative scheme, so a
malformed policy fails at the boundary, not deep inside instantiation.

The carrier is

```cpp
template < typename T, typename Basis, typename Scheme, typename Storage = storage::Dense >
class tax::Expansion;
// aliases: Series<Basis,N,T>; TaylorSeries<N,M,T>; ChebyshevSeries<N,M,T>
```

It owns the `std::array<T, Scheme::nCoeff>` and everything **basis-independent**:
`+`, `-`, scalar `*`/`/`, the compound assignments, and the factories
`constant(v)` and `variable<I>()`. Those factories are basis-independent on
purpose: every classical family has `P_0 ≡ 1` and `P_1 ≡ x`, so a constant is
`v·P_0` (set `c_0`) and the I-th coordinate is `1·P_1(x_I)` (set the degree-1
slot of axis `I`) in *any* conforming basis.

Everything that differs is one `Basis::…` call away, dispatched at compile time
with zero overhead. The three policies (Basis × Scheme × Storage) compose
freely; `TaylorExpansion<T,Scheme,Storage>` is the same mathematics as
`Expansion<T,TaylorBasis,Scheme,Storage>` (shared kernels) and a regression test
checks they agree coefficient-for-coefficient, univariate **and** multivariate.

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

The **multivariate** case is the tensor product `T_α = Π_i T_{α_i}(x_i)`: the
product folds each axis independently (`2^M` sign combinations), evaluation
tensors the per-axis Clenshaw tables, and `∂/∂x_axis` / `∫dx_axis` run the 1-D
recurrence fiber-by-fiber along that axis — all `constexpr`.

**Domain mapping** is carried in the basis *type*: `ChebyshevBasisOn<Lo,Hi>`
(floating-point NTTP, default `[-1,1]`) maps `x ∈ [Lo,Hi]` to the canonical
`u ∈ [-1,1]`. Evaluation maps the point, differentiation/integration carry the
`du/dx = 2/(Hi−Lo)` chain-rule factor, and interpolation samples on `[Lo,Hi]`.
Two models on different intervals are different types and cannot be silently
mixed.

Plus two Chebyshev-specific capabilities with no Taylor analogue:

- **`chebyshevInterpolate<N, Basis>(f)`** — build the order-`N` near-best
  *uniform* approximation of an arbitrary callable on the basis's interval from
  its Chebyshev–Gauss–Lobatto samples (discrete cosine sum). The canonical
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
| **multivariate** (`+ - * eval deriv integ`) | ✓ | ✓ |
| **domain mapping** `[a,b]`      |   —²   |     ✓     |

² The Taylor carrier is centred at 0 (`eval` is absolute `x`); per-expansion
recentre/scale is a future convenience, not a basis property.

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

The merge is **done**: there is one class, `Expansion<T, Basis, Scheme,
Storage>`, and

```cpp
template < typename T, typename Scheme, typename Storage = storage::Dense >
using TaylorExpansion = Expansion< T, TaylorBasis, Scheme, Storage >;
```

is a transparent alias (a `static_assert` in `test_series_unify` pins the type
identity). The whole legacy ecosystem — named axes, mixed-order, sparse storage,
`Batch` coefficients, the Eigen `NumTraits` / gradient / Jacobian surface, and
every existing operator — now rides on the unified class unchanged, because the
Taylor instance reproduces the previous behaviour exactly:

- the basis-specific operations (`eval`, `deriv`, `integ`) delegate to
  `TaylorBasis`, which is the same monomial arithmetic the class used inline;
- the Taylor-only *value* semantics that don't generalise (`k!`-scaled
  derivative values, `gradient`, `hessian`, displacement `eval`) remain members
  — they are simply not part of the basis-generic contract;
- the legacy free operators stay authoritative for `TaylorBasis`; the
  basis-generic operator surface (`series/operators.hpp`) is gated on
  `!is_same_v<B, TaylorBasis>`, so Chebyshev (and future families) get their own
  non-conflicting set.

Sparse storage remains `TaylorBasis`-only (a `static_assert` enforces it);
generalising the sparse recurrences to other families is future work.

## Roadmap

- **More families**: Legendre, Hermite, Laguerre — each is one policy
  (recurrence-defined product + Gauss-quadrature evaluation/projection).
- **Transcendentals in any basis, exactly**: solve `y' = g'·y` etc. as a banded
  linear system in coefficient space (replacing Chebyshev's sample-and-fit with
  a `constexpr` exact path); multivariate Chebyshev composition.
- **Chebyshev on the ecosystem**: named axes, sparse, and `tax::la` for
  non-Taylor bases; per-axis Chebyshev domains.
- **Spectral utilities**: Clenshaw–Curtis quadrature, rootfinding, adaptive
  degree.
```
