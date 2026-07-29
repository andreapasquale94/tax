# Named Expansions

`tax::NamedTaylorExpansion<T, N, Axes...>` wraps a dense
`TaylorExpansion<T, N, M>` and attaches a compile-time list of **named axes** —
each a contiguous block of the underlying $M$ variables identified by a
compile-time string. You then build, combine, and project expansions by
**variable name** instead of by raw flat index; the dependency set of a result
is tracked in its type and derived automatically from its operands.

The whole named API is reachable directly under `tax` — `tax::NE`,
`tax::variable`, `tax::variables`, `tax::Axis`, `tax::NamedTaylorExpansion`, and
the Eigen helpers `tax::gradient` / `tax::hessian` / `tax::jacobian` — all from
`<tax/tax.hpp>`. (It lives in `namespace tax::named` and is re-exported; prefer
the `tax::` spelling.) Exact signatures are in the
[Named API reference](../reference/named.md); this page is the how-to.

```cpp
#include <tax/tax.hpp>

auto x = tax::variable<"x", 4>(1.0);
auto p = tax::variable<"p", 4>(0.5);

// x and p carry *different* axis sets; composing them runs in the union.
auto f = sin(x) + x * p;     // type: NamedTaylorExpansion over axes {p, x}
```

---

## Axes and the canonical type

An axis is `tax::Axis<Name, Dim>` — a name (a `tax::FixedString` NTTP) and a
block of `Dim ≥ 1` consecutive variables:

```cpp
using PosX = tax::Axis<"x", 3>;   // a 3-D axis called "x"
using Time = tax::Axis<"t", 1>;   // a scalar axis called "t"
```

Every axis list is kept in **canonical order** (sorted by name, unique), so the
type does not depend on the order you wrote the operands: `x * p` and `p * x`
produce the *same* type. The alias `NE` spells the common `double`-valued case:

```cpp
template <int N, typename... Axes>
using NE = tax::NamedTaylorExpansion<double, N, Axes...>;   // e.g. NE<4, Axis<"x",3>>
```

---

## Creating named variables

```cpp
// Scalar (1-D) axis — returns a single expansion:
auto t = tax::variable<"t", 6>(0.0);                 // NE<6, Axis<"t",1>>

// Multi-dimensional axis — returns std::array of the D coordinate variables:
std::array<double, 3> x0{1.0, 2.0, 3.0};
auto x = tax::variables<"x", 6>(x0);                 // std::array<NE<6,Axis<"x",3>>, 3>

// Eigen expansion point (tax::variables Eigen overload):
Eigen::Vector3d v0{1.0, 2.0, 3.0};
auto xv = tax::variables<"x", 6>(v0);                // Eigen vector of named variables
```

---

## Composition across axis sets

Arithmetic between expansions over **different** axis sets runs in the union of
the two sets: both operands are first **embedded** into the union, then the
dense kernels do the work. The result type carries the union of axes:

```cpp
auto x = tax::variable<"x", 4>(1.0);   // axes {x}
auto y = tax::variable<"y", 4>(2.0);   // axes {y}

auto g = x * x + x * y + y * y;        // axes {x, y}, computed exactly
```

A value that depends on **fewer** axes promotes implicitly into a wider axis set
(value-preserving: the absent axes get zero derivatives), so a narrow expansion
can be passed where a wider one is expected. All the usual math functions
(`sin`, `exp`, `sqrt`, `pow`, `atan2`, …) work on named expansions and preserve
the axis set.

---

## Slicing and named derivatives

`slice<Names...>()` projects an expansion back onto a subset of its axes by
keeping the monomials that do not depend on the dropped axes (i.e. restricting
the dropped axes to their expansion point):

```cpp
auto h = f.slice<"x">();        // drop every axis except "x"
```

`deriv<"name">()` and `integ<"name">()` differentiate / integrate symbolically
with respect to a named axis (the axis set is preserved). Composing them with
`slice` gives the "sub-derivative" projection:

```cpp
auto dfx = f.deriv<"p">().slice<"x">();   // ∂f/∂p, then restricted to axis x
```

---

## Eigen helpers

`<tax/tax.hpp>` provides a `NumTraits` specialisation so named expansions live
inside Eigen vectors/matrices, plus name-addressed differential operators:

```cpp
auto gx = tax::gradient<"x">(f);   // gradient w.r.t. the coords of axis "x"
auto Hx = tax::hessian<"x">(f);    // Hessian  w.r.t. axis "x"
auto Jx = tax::jacobian<"x">(F);   // Jacobian of an Eigen vector F w.r.t. "x"
```

---

## Anisotropic axes: per-axis orders

Sometimes variables have very different smoothness scales. A short-time
integrator may need order 20 in the time step `t` but only order 4 in a spatial
coordinate `x`. A single isotropic `TE<24, 2>` would allocate every monomial up
to *total* degree 24 — including `x⁵ … x²⁴`, none of which carry useful
information. **Mixed-order** expansions keep a monomial only when *each axis's*
partial degree is within that axis's own cap, forming a **box** in monomial
space rather than a simplex. For `x@4` and `t@20` the box contains `x⁴·t²⁰` but
never `x⁵` — `5 × 21 = 105` coefficients instead of `C(26,2) = 325`.

Mixed-order axes use `tax::OrderedAxis<Name, Dim, Order>` and the
`tax::mixed::` factories; the type is `tax::MixedTaylorExpansion<T, Axes...>`
(alias `MTE`). It carries the full dense math surface and the `tax::la` helpers
exactly like the single-order named type — everything above applies unchanged.

```cpp
#include <tax/tax.hpp>

// Scalar axis "x" at per-axis order 4:
auto x = tax::mixed::variable<"x", 4>(1.0);

// 3-D axis "p" at per-axis order 20:
std::array<double, 3> p0{0.1, 0.2, 0.3};
auto p = tax::mixed::variables<"p", 20, 3>(p0);   // std::array of 3 expansions

// Compose: union axis set {p@20, x@4} — no x⁵ or higher is ever stored.
auto f = sin(x) + x * p[0];
```

### Max-order promotion

Axis lists are canonical (sorted, unique) here too, so `x * p[0]` and
`p[0] * x` produce the same type. When two operands share an axis name, the
result uses the **maximum** per-axis order of the two — composition never
silently truncates a shared axis below what either operand required:

```cpp
auto x2 = tax::mixed::variable<"x", 2>(0.3);
auto x5 = tax::mixed::variable<"x", 5>(0.3);
auto f  = x2 * x5;   // shared axis "x" → promoted to order 5
```

### Lowering one axis: `truncate<"name", N2>()`

```cpp
// f is {t@20, x@4}; lower t to order 2 → {t@2, x@4}, 3×5 = 15 coefficients.
auto ft = f.truncate<"t", 2>();
```

Useful for progressive refinement or for producing a cheaper surrogate from a
high-order expansion.

### The anonymous box type

When names add no value, `tax::MixedTE<tax::Group<Dim, Order>…>` is a
`TaylorExpansion` over a `MixedScheme` directly — same math surface, no axis
labels:

```cpp
using ME = tax::MixedTE<tax::Group<1, 4>, tax::Group<1, 3>>;   // var0@4, var1@3
typename ME::Input p{0.3, -0.2};
ME x = ME::variable<0>(p);
auto f = sin(x * ME::variable<1>(p)) + exp(x);   // 5 × 4 = 20 coefficients
```

### Notes & limits

- **Box, not joint simplex.** A monomial is kept iff every group's block degree
  is within that group's cap; there is no joint total-degree constraint across
  groups. A joint cap is a planned follow-up, not part of the API today.
- **Multi-dimensional axes truncate by total degree *within* the axis.** A
  `Group<3, 4>` keeps monomials whose total degree across its three variables is
  ≤ 4; the per-axis cap operates *across* groups, not *within* one group.

| Alias | Meaning |
|------|---------|
| `tax::Group<Dim, Order>` | One variable group: `Dim` variables capped at `Order` |
| `tax::MixedTE<Groups…>` | Anonymous box expansion (`TaylorExpansion<double, MixedScheme<Groups…>>`) |
| `tax::OrderedAxis<Name, Dim, Order>` | Named axis with its own per-axis order |
| `tax::MixedTaylorExpansion<T, Axes…>` / `MTE<Axes…>` | Named per-axis-order expansion |
| `tax::mixed::variable<"name", Order>(x0)` | Factory: 1-D named axis variable |
| `tax::mixed::variables<"name", Order, D>(arr)` | Factory: D-D named axis variables |

---

See the [Named API reference](../reference/named.md) for the full list of types,
factories, member operations, and Eigen helpers (single-order and mixed).
