# Mixed-Order (Anisotropic) Expansions

Sometimes a problem has variables with very different smoothness scales. A
short-time integrator may need a Taylor expansion of order 20 in the time step
`t` but only order 4 in a spatial coordinate `x`. A single isotropic
`TE<24, 2>` would allocate all the monomials up to total degree 24 — including
`x⁵, x⁶, … x²⁴` — none of which carry useful information. Mixed-order
expansions avoid this: they keep a monomial only when *each axis's* partial
degree is within that axis's individual order cap, forming a **box** in
monomial space rather than a simplex.

For `x@4` and `t@20` the box contains `x⁴·t²⁰` but never `x⁵`. The
coefficient count is `5 × 21 = 105` instead of `C(26,2) = 325`, and no
high-degree `x` monomials are computed at any step.

---

## The unified type

`TaylorExpansion` is parameterized by an *index scheme*. The classic,
single-order form uses `IsotropicScheme<N, M>` (total-degree simplex); the
mixed form uses `MixedScheme<Group<Dim,Order>…>` (per-group box).

The convenience alias for the anonymous box type is:

```cpp
tax::BoxTE<tax::Group<Dim, Order>…>
```

`BoxTE` is **not a separate type** — it is a `TaylorExpansion` with a
`MixedScheme` index scheme. It carries the full dense math surface (`+`, `-`,
`*`, `/`, `sin`, `exp`, `sqrt`, `pow`, …) and integrates with `tax::la`
exactly as `TE` does.

```cpp
#include <tax/tax.hpp>

// Two variable groups: var 0 @ order 4, var 1 @ order 3.
using ME = tax::BoxTE<tax::Group<1, 4>, tax::Group<1, 3>>;

typename ME::Input p{0.3, -0.2};
ME x = ME::variable<0>(p);
ME y = ME::variable<1>(p);

auto f = sin(x * y) + exp(x);          // full math surface
auto g = tax::la::gradient( f );       // Eigen::Vector2d of first-order derivatives
auto H = tax::la::hessian( f );        // Eigen::Matrix2d Hessian
```

The box size is `numMonomials(4, 1) × numMonomials(3, 1) = 5 × 4 = 20`
coefficients, compared to `numMonomials(7, 2) = 36` for a joint-simplex
`TE<7, 2>`.

---

## Named axes (the ergonomic layer)

For multi-axis problems the anonymous `BoxTE` spelling is inconvenient.
The named layer attaches compile-time string labels to each group:

```cpp
#include <tax/tax.hpp>

// Scalar axis "x" at per-axis order 4:
auto x = tax::mixed::variable<"x", 4>(1.0);

// 3-D axis "p" at per-axis order 20:
std::array<double, 3> p0{0.1, 0.2, 0.3};
auto p = tax::mixed::variables<"p", 20, 3>(p0);   // std::array of 3 expansions

// Compose: union axis set {p@20, x@4} — no x⁵ or higher stored.
auto f = sin(x) + x * p[0];
```

The result type is `tax::MixedExpansion<double, OrderedAxis<"p",3,20>, OrderedAxis<"x",1,4>>`,
re-exported as `tax::MixedExpansion<T, Axes…>`.

### Canonical type and max-order promotion

The axis list in every `MixedExpansion` is sorted by name and unique, so
`x * p[0]` and `p[0] * x` produce the *same* type. When two operands share an
axis name, the result uses the **maximum per-axis order** of the two:

```cpp
auto x2 = tax::mixed::variable<"x", 2>(0.3);
auto x5 = tax::mixed::variable<"x", 5>(0.3);
auto f = x2 * x5;   // shared axis "x" → promoted to order 5
```

This means composition never silently truncates a shared axis below what either
operand required.

---

## Operations

### Standard accessors

`MixedExpansion` supports the same accessors as `TaylorExpansion`:

```cpp
double val  = f.value();                   // constant term f(x0)
double c    = f.coeff(alpha);              // raw Taylor coefficient (runtime multi-index)
double d    = f.derivative(alpha);         // k!-scaled derivative (runtime multi-index)
double y    = f.eval(dx);                  // evaluate polynomial at displacement dx
```

### Named differentiation and integration

`deriv<"name">()` and `integ<"name">()` differentiate or integrate with
respect to the first coordinate of a named axis (use the `Local` template
parameter for axes with `Dim > 1`). The axis set and per-axis orders are
preserved:

```cpp
auto df_dx = f.deriv<"x">();              // ∂f/∂x, same axis set and orders
auto fi_t  = f.integ<"t">();              // ∫f dt, same axis set and orders
```

### Slice

`slice<Names…>()` projects onto a named subset of axes. Monomials with nonzero
degree in any dropped axis are discarded (the dropped axes are frozen at their
expansion point):

```cpp
auto sx = f.slice<"x">();                 // only the "x"-axis terms survive
auto sp = f.slice<"p", "x">();            // keep both; equivalent to identity if those are all axes
```

### Truncate

`truncate<"name", N2>()` lowers one axis's per-axis order from its current
value to `N2`, dropping any monomials whose degree in that axis exceeds `N2`:

```cpp
// f is {t@20, x@4}; lower t to order 2 → {t@2, x@4}, 3×5 = 15 coefficients.
auto ft = f.truncate<"t", 2>();
```

This is useful for progressive refinement or for producing a cheaper surrogate
from a high-order expansion.

### Named `la` helpers

`<tax/tax.hpp>` provides `Eigen::NumTraits` for `MixedExpansion`, so
mixed expansions work as Eigen scalars inside `Eigen::Matrix<…>`. The
name-addressed differential helpers work with both `tax::named::` and the
public `tax::` spellings:

```cpp
auto gx = tax::gradient<"x">(f);         // gradient w.r.t. all coords of axis "x"
auto Hx = tax::hessian<"x">(f);          // Hessian restricted to axis "x"
auto Jx = tax::jacobian<"x">(F);         // Jacobian of an Eigen vector F w.r.t. "x"
```

Each result is an Eigen matrix/vector of plain `double` values (the dimensions
match the axis's `Dim`). `tax::la::VecNT<D, MixedExpansion<…>>` works as
expected for storing vectors of mixed expansions.

---

## Box vs simplex; the joint cap

The default layout for `MixedScheme` and `MixedExpansion` is the **full
box**: a monomial is kept iff every group's block degree is within that group's
per-axis order. There is no joint total-degree constraint across groups.

A joint total-degree cap — dropping monomials whose *summed* degree across
groups exceeds a bound, to shave the high mixed terms the product box still
carries — is a **planned follow-up**, not yet part of the API. `MixedScheme`
is currently `MixedScheme<Group<Dim,Order>…>` with no joint-cap parameter; the
full box is the only mode today, and it is the right default for the vast
majority of use cases.

For the classical joint-simplex named layer (same order on all axes) see
[Named Expansions](named.md).

---

## Aliases

| Name | Meaning |
|------|---------|
| `tax::Group<Dim, Order>` | One variable group: `Dim` variables capped at `Order` |
| `tax::BoxTE<Groups…>` | Anonymous box expansion: `TaylorExpansion<double, MixedScheme<Groups…>>` |
| `tax::OrderedAxis<Name, Dim, Order>` | Named axis with its own per-axis order |
| `tax::MixedExpansion<T, Axes…>` | Named per-axis-order expansion (double alias: `T = double` is most common) |
| `tax::mixed::variable<"name", Order>(x0)` | Factory: 1-D named axis variable |
| `tax::mixed::variables<"name", Order, D>(arr)` | Factory: D-D named axis variables (`std::array<…, D>`) |

---

## Notes & limits

- **Dense storage only.** `BoxTE` / `MixedExpansion` use dense
  coefficient storage. Sparse storage is defined only for the isotropic scheme
  (`STE<N, M>`).
- **Multi-dimensional axes truncate by total degree within the axis.** A
  `Group<3, 4>` group keeps all monomials in its three variables whose total
  degree (within those three variables) is ≤ 4. The per-axis cap operates
  *across* groups, not *within* a single multi-dimensional group.
- **Batch coefficients.** `Batch<T, K>` (SIMD lanes) and `MixedScheme` are
  structurally compatible, but `BoxTE<…>` with a `Batch` scalar is not a
  shipped convenience alias. If you need ensemble propagation on a mixed-order
  box, construct the `TaylorExpansion<Batch<double,K>, MixedScheme<…>>` type
  directly.
- **Full math surface.** Because `BoxTE` is a `TaylorExpansion`, every
  operator and math function that works for `TE` also works for `BoxTE`: the
  kernel dispatch is driven by the scheme's `cauchyProduct` /
  `forEachRecurrenceRow`, not by any special-casing in the operator layer.

See also [Batch (SIMD) Coefficients](batch.md) for the lane-parallel form and
[Named Expansions](named.md) for the joint-simplex named layer.
