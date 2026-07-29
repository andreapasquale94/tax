# CLAUDE.md — AI Assistant Guide for `tax`

## Project Overview

**tax** is a header-only C++23 library for **Truncated Algebraic eXpansions (TAX)** — truncated multivariate Taylor polynomials that propagate complete Taylor series through arbitrary expressions. In a single evaluation pass, it yields function values and all partial derivatives up to order N. It provides allocation-free dense storage, *named* expansions (type-level variable axes) including *mixed-order* axes and Eigen integration (`tax::la`).

> **Note:** adaptive ODE integration (`tax::ode`) and Automatic Domain Splitting (`tax::ads`) are no longer part of this repository — they were split out, unchanged, into a separate companion plugin built on top of `tax` (see the README). Do not look for `include/tax/ode` or `include/tax/ads` here.

- **Version:** 0.1.0
- **License:** BSD 3-Clause
- **C++ Standard:** C++23 (required)
- **Build system:** CMake

---

## Repository Structure

```
tax/
├── include/tax/              # Header-only library (the entire library lives here)
│   ├── tax.hpp               # Umbrella header — users include only this
│   ├── la.hpp                # Facade: linear-algebra / Eigen helpers (tax::la)
│   ├── core/                 # The TaylorExpansion type and its foundations
│   │   ├── concepts.hpp      #   Scalar, TaylorPolynomial, DensePolynomial concepts
│   │   ├── multi_index.hpp   #   MultiIndex<M>, flatIndex/unflatIndex, numMonomials
│   │   ├── enumeration.hpp   #   forEachMonomial / forEachSubIndex
│   │   ├── scheme.hpp        #   index-scheme facade; scheme/{concept,isotropic,mixed}.hpp
│   │   │                     #   IsotropicScheme<N,M> (single order) + MixedScheme (per-axis)
│   │   ├── taylor_expansion.hpp  # TaylorExpansion<T, Scheme>: the dense expansion type
│   │   ├── named.hpp         #   NamedTaylorExpansion<T,N,Axes...>: single-order named axes
│   │   ├── mixed_named.hpp   #   MixedTaylorExpansion<T,Axes...>: per-axis-order named axes
│   │   ├── promote.hpp       #   promote_t<Ts...>: common (union-of-axes) expansion type
│   │   └── storage/          #   DenseContainer (std::array) coefficient container
│   ├── kernels/              # Series recurrence kernels (tax::detail::kernels)
│   │   ├── cauchy.hpp        #   cauchyProduct dispatch (+ in-header config macros)
│   │   ├── cauchy_unroll.hpp #   fully unrolled univariate (M == 1) product
│   │   ├── cauchy_stencil.hpp#   precomputed stencil table product (M >= 2)
│   │   ├── recurrence_stencil.hpp # shared decomposition table for M>=2 recurrences
│   │   ├── mixed_stencils.hpp#   Cauchy / recurrence stencils for MixedScheme
│   │   ├── algebra.hpp       #   shared recurrence drivers (seriesDerivQuotient /
│   │   │                     #   seriesDerivProduct), square/cube, reciprocal, sqrt, cbrt, pow
│   │   ├── trigonometric.hpp #   sin, cos, tan, asin, acos, atan
│   │   ├── transcendental.hpp#   exp, log, sinh/cosh/tanh (+ fused sinhCosh) + inverses, erf
│   │   └── fused.hpp         #   pair-fused kernels: expSinCos (exp·trig), sqrtInvSqrt
│   ├── operators/            # Free-function operator surface over the kernels
│   │   ├── arithmetic.hpp        #   +, -, *, /, compound assignment
│   │   ├── math_unary.hpp        #   sin, exp, sqrt, square, …
│   │   ├── math_binary.hpp       #   pow, pow<N>/pow<N,M>, halfPow<K>/invSqrtPow<K>, atan2, …
│   │   ├── math_fused.hpp        #   sinCos, sinhCosh, sqrtInvSqrt, expSin/expCos/expSinCos
│   │   │                         #   (dense + named + mixed, pair-returning forms)
│   │   ├── mixed_math.hpp        #   pow/atan2 for MixedTaylorExpansion + tax:: re-exports
│   │   └── named_{arithmetic,math_unary,math_binary}.hpp  # same surface for named/mixed
│   ├── la/                   # Eigen integration (namespace tax::la; some re-exported as tax::)
│   │   ├── types.hpp         #   Vec, Mat, VecNT<N,T>, MatNT, MatNMT
│   │   ├── expansion_vectors.hpp #   TEVec<D,N,M>, NEVec<D,N,Axes...>, MTEVec<D,Axes...>
│   │   ├── num_traits.hpp    #   Eigen::NumTraits<TaylorExpansion>
│   │   ├── values.hpp        #   variables(x0), value(), eval()
│   │   ├── truncate.hpp      #   free tax::truncate<N2>(scalar | Eigen vector/matrix)
│   │   ├── derivatives.hpp   #   derivative, gradient, hessian, jacobian
│   │   ├── norm.hpp          #   norm<P, Q> of a vector of expansions (Q defaults to 1)
│   │   ├── vector_ops.hpp    #   dot, cross, angle, unitvec, unitcross, projvec, projplane
│   │   ├── named.hpp         #   NumTraits + gradient/hessian/jacobian by axis name
│   │   ├── mixed_named.hpp   #   the same for mixed-order named expansions
│   │   └── invert.hpp        #   formal polynomial-map inversion (Picard)
│   └── io/series.hpp         # human-readable streaming: operator<<, series(), to_string()
├── tests/                    # Google Test suite
│   ├── core/                 #   ctor/accessors, multi-index, enumeration, deriv/integ, named
│   ├── kernels/              #   dense/unroll/stencil Cauchy verification
│   ├── operators/            #   one file per math-function family
│   ├── mixed/                #   MixedScheme + mixed-order named expansions
│   ├── eigen/                #   tax::la helpers (gradient, jacobian, invert, named, …)
│   ├── io/                   #   series / streaming
│   ├── regression/           #   DACE comparison suite (opt-in, TAX_BUILD_REGRESSIONS)
│   └── testUtils.hpp         #   shared helpers/macros
├── docs/                     # MkDocs docs: guide/, reference/, concepts/, internals/
├── cmake/                    # CMake package config template
├── .github/workflows/        # CI: tests.yml, sanitizers.yml, regressions.yml, docs.yml
├── .clang-format             # Code style configuration
├── pyproject.toml            # scikit-build-core wheel config (Python bindings planned)
├── CMakeLists.txt            # Root CMake configuration
└── README.md
```

---

## Building

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Test
ctest --test-dir build --output-on-failure
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `TAX_BUILD_UNITTESTS`   | `ON`  | Build Google Test unit-test suite |
| `TAX_BUILD_REGRESSIONS` | `OFF` | Build DACE-based regression tests (fetches DACE v2.1.0) |

Kernel dispatch (`TAX_USE_UNROLL` / `TAX_USE_STENCIL`) is configured **in-header**
in `<tax/kernels/cauchy.hpp>` and defaults to ON for every consumer. It is
deliberately *not* injected from the build system: differing values across
translation units would change inline function definitions (ODR). A project may
pre-define either macro to `0`, but the value must be identical project-wide.

### Dependencies

- **Required:** Eigen3 (`find_package(Eigen3 REQUIRED)`), Threads
- **Optional:** DACE v2.1.0 — fetched via FetchContent when `TAX_BUILD_REGRESSIONS=ON`
- **Test framework:** Google Test v1.17 — fetched automatically if not found

---

## Core Concepts

### The Main Type

```cpp
tax::TaylorExpansion<T, Scheme>
// T       = coefficient type (double or float)
// Scheme  = index scheme: IsotropicScheme<N,M> (one order N over M vars)
//           or MixedScheme<...> (per-axis orders); fixes the monomial layout
```

Most code uses the aliases rather than naming a `Scheme` directly (all `double`-valued unless noted):
```cpp
tax::TE<N, M = 1>         // plain (integer-indexed) expansion
tax::TEn<N, M>            // explicit multivariate spelling
tax::NE<N, Axes...>       // named (single order)          — see Named Expansions
tax::MTE<Axes...>         // mixed-order named             — see Named Expansions
```

Coefficients are a `std::array<T, numMonomials(N, M)>` in graded-lex order:
full `constexpr` surface, no heap. Named and mixed-order expansions are thin
wrappers over the same array.

### Creating Variables

```cpp
// Univariate
auto x = tax::TE<5>::variable(1.0);            // x = 1.0 + 1*dx

// Multivariate (static index)
typename tax::TE<3, 2>::Input p{1.0, 2.0};
auto x = tax::TE<3, 2>::variable<0>(p);
auto y = tax::TE<3, 2>::variable<1>(p);

// Multivariate (runtime index)
auto z = tax::TE<3, 2>::variable(1.0, /*var_idx=*/0);

// Eigen vector of coordinate variables (tax::la)
Eigen::Vector2d x0{1.0, 2.0};
auto vars = tax::la::variables<tax::TE<3, 2>>(x0);
```

### Using the Library

```cpp
#include <tax/tax.hpp>

auto x = tax::TE<5>::variable(1.0);
auto f = sin(x) * exp(x);

double val = f.value();           // constant term
double c2  = f.coeff<2>();        // raw Taylor coefficient
double d2  = f.derivative<2>();   // k!-scaled derivative
double y   = f.eval(dx);          // evaluate Taylor polynomial at x0+dx
auto   df  = f.deriv<0>();        // symbolic partial derivative
auto   F   = f.integ<0>();        // symbolic integral
```

`coeff` / `derivative` / `deriv` / `integ` all exist in compile-time
(`<...>`), `MultiIndex<M>`, and runtime-`int` forms.

The pure-polynomial surface is `constexpr` and runs in constant evaluation:
arithmetic, `square`/`cube`/`reciprocal`, integer `pow`, division, and the
`deriv`/`integ`/`eval`/`truncate` accessors (see
`tests/core/test_constexpr.cpp`). The transcendental functions seed their
recurrence with a libm call (`std::exp`, `std::sin`, ...), so they are
runtime-only.

### Coefficient Storage

- Graded-lexicographic ordering: all degree-0 first, then degree-1, etc.
- Size: `nCoefficients = numMonomials(N, M) = C(N+M, M)`
- `coeff` retrieves the raw Taylor coefficient; `derivative` applies `k!` scaling
- The degree-d monomials occupy the contiguous flat-index block
  `[numMonomials(d-1, M), numMonomials(d, M))` — several hot paths
  (truncation criteria, kernels) rely on this

---

## Kernels

All math operations are degree-by-degree recurrence relations in
`include/tax/kernels/` (`tax::detail::kernels`), operating directly on the
coefficient arrays. Two shared drivers in `algebra.hpp` implement the common
recurrence shapes once (univariate + multivariate walks):
`seriesDerivQuotient` solves `h·out' = ±src'` (log, asin/acos/atan/atan2,
asinh/acosh/atanh) and `seriesDerivProduct` solves `out' = src'·h` (exp, erf).
Most kernels reduce to "compute h, seed the constant term, call the driver".

The pure-polynomial kernels (`seriesSquare`/`seriesCube`/`seriesReciprocal`/
`seriesDivide`/`seriesPowInt` and the two drivers) are `constexpr`. The
transcendental kernels evaluate one libm seed on the constant term
(`out[0] = std::exp(a[0])`, etc.) and are therefore runtime-only; when adding
one, follow the existing pattern (`using std::exp; out[0] = exp(a[0]);`).

Pair-fused kernels live in `fused.hpp` (`seriesExpSinCos`, `seriesSqrtInvSqrt`)
and are exposed via `operators/math_fused.hpp` (`expSin`, `expCos`,
`expSinCos`, `sinCos`, `sinhCosh`, `sqrtInvSqrt`); the fused exp·trig pass is
~2x faster than composing `exp(v) * cos(u)`. These were ported from the
expression-template prototype branch — the ET layer itself benchmarked at
parity and was deliberately not ported. The Cauchy product has three dense variants behind one
dispatch (`cauchyProduct`):

- `cauchyProductLoop` — generic, `constexpr`-safe (used in constant evaluation)
- `cauchyProductUnroll` — pack-expansion unrolled, M == 1
- `cauchyProductStencil` — precomputed (out, a, b) index table, M >= 2;
  exactly `numMonomials(N, 2M)` entries, built once at first use
  (runtime static — not usable in constant evaluation)

All other M >= 2 recurrences (exp, log, sin/cos, tan, pow, reciprocal,
sqrt, ...) are driven by one shared decomposition table:
`forEachRecurrenceRow<N, M>(fn)` in `recurrence_stencil.hpp` hands each
kernel the precomputed (flatIndex(beta), flatIndex(gamma), |beta|) rows
per output monomial; only the recurrence weight stays in the kernel.
Constant evaluation (and `TAX_USE_STENCIL=0`) enumerates the same rows
on the fly — bit-identical results.

When adding a new math function: implement the recurrence in the right
kernel file, expose it via `operators/math_unary.hpp` or `math_binary.hpp`,
and add tests under `tests/operators/` (plus `tests/kernels/` if it has its
own kernel).

---

## Eigen Integration (`tax::la`)

```cpp
#include <tax/tax.hpp>   // la.hpp is included by the umbrella

Eigen::Vector2d x0{1.0, 2.0};
auto v = tax::la::variables<tax::TE<3, 2>>(x0);   // Eigen vector of TE variables

auto vals = tax::la::value(f);          // constant terms
auto grad = tax::la::gradient(f);       // gradient
auto J    = tax::la::jacobian(F);       // Jacobian
auto H    = tax::la::hessian(f);        // Hessian
auto Finv = tax::la::invert(F);         // formal map inversion (Picard)
```

`num_traits.hpp` specialises `Eigen::NumTraits` so TE types work as Eigen
scalars (`tax::la::VecNT<D, TE>` is a convenient Eigen vector of TE).

---

## Named Expansions (`tax::named`)

`tax::NamedTaylorExpansion<T, N, Axes...>` wraps a dense
`TaylorExpansion<T, N, M>` and attaches a compile-time list of **named axes**
(`Axis<Name, Dim>`, where `Name` is a `FixedString` NTTP and `Dim` is a block of
consecutive variables). It is implemented in `tax::named` and the whole API —
including the Eigen helpers `gradient`/`hessian`/`jacobian` — is re-exported
under `tax`; use the `tax::` spelling.

```cpp
#include <tax/tax.hpp>

auto x = tax::variable<"x", 4>(1.0);          // 1-D axis "x"      → NE<4, Axis<"x",1>>
auto p = tax::variables<"p", 4>(arr3);        // 3-D axis "p"      → std::array of NE
auto f = sin(x) + x * p[0];                   // composes in the union of axes {p, x}
auto g = f.deriv<"x">().slice<"p">();         // named ∂/∂x, then projected onto axis p
auto J = tax::jacobian<"x">(F);               // Jacobian of Eigen vector F w.r.t. "x"
```

- **Canonical type:** axis lists are sorted-by-name and unique, so `x * p` and
  `p * x` produce the *same* type (`NE<N, Axes...>` is the `double` alias).
- **embed / compose / slice:** operands over different axis sets are embedded
  into the union before the dense kernels run; the result type tracks the union.
  A narrower expansion promotes implicitly into a wider axis set.
- **Named differential ops:** `deriv<"name">()`, `integ<"name">()`,
  `slice<Names...>()`; LA helpers `gradient<"name">`, `hessian<"name">`,
  `jacobian<"name">` in `la/named.hpp`.

Key files: `core/named.hpp` (the type + `variable`/`variables` factories,
embed/slice/compose) and `la/named.hpp` (Eigen `NumTraits` + name-addressed
gradient/hessian/jacobian).

### Mixed-order named expansions (`tax::MTE`)

`tax::MixedTaylorExpansion<T, Axes...>` (alias `MTE<Axes...>`) is the same idea
but each axis carries its **own** truncation order via `OrderedAxis<Name, Dim,
Order>` and a `MixedScheme` layout. Factories live in `tax::mixed`
(`tax::mixed::variable<"x", Order>(...)`, `tax::mixed::variables<...>`). Axis
lists are sorted/unique (canonical type) and operands embed into the union just
like `NE`. Key files: `core/mixed_named.hpp`, `kernels/mixed_stencils.hpp`,
`la/mixed_named.hpp`.

### Cross-cutting utilities

- `tax::promote_t<Ts...>` (`core/promote.hpp`) — the common type operands
  promote into (union of axes; scalars promote into the expansion). Handy for
  declaring a homogeneous container that must hold a mix of axis sets.
- `tax::truncate<N2>(x)` (`la/truncate.hpp`) — free order-reducing truncation
  for a scalar expansion **or** an Eigen vector/matrix (element-wise).
- `tax::la::TEVec<D,N,M>` / `NEVec<D,N,Axes...>` / `MTEVec<D,Axes...>`
  (`la/expansion_vectors.hpp`) — `VecNT<D, …>` shorthands for Eigen vectors of
  expansions.
- Printing (`io/series.hpp`): `std::cout << f` (polynomial series),
  `tax::series(f, opts)` (tabular / per-element for Eigen vectors),
  `tax::to_string(f)`.

---

## Code Conventions

### Naming

| Category | Convention | Examples |
|----------|-----------|---------|
| Types/Classes | `PascalCase` | `TaylorExpansion`, `MultiIndex`, `NamedTaylorExpansion`, `Axis` |
| Template params | `UPPERCASE` or short | `T`, `N`, `M`, `D`, `Derived` |
| Free functions & methods | `camelCase` | `variable()`, `flatIndex()`, `seriesReciprocal()`, `deriv()`, `popFront()` |
| Local variables | `snake_case` | `n_coeff`, `dx`, `half_width` |
| Namespaces | `lowercase` | `tax`, `tax::detail`, `tax::named`, `tax::la` |
| Type aliases | Short uppercase | `TE<N, M>`, `TEn<N, M>`, `NE<N, Axes...>` |

### C++ Patterns

- **`constexpr` everywhere in core:** size calculations, index mappings, and
  coefficient operations must stay `constexpr`; kernels that use runtime
  statics (stencil) must keep a `constexpr`-safe fallback behind `if !consteval`
- **`noexcept` on all operations** (exception: methods that `throw`, e.g.
  runtime-index `deriv(int)`)
- **No heap allocation anywhere in the library:** coefficients are always a
  `std::array`
- **Concepts over SFINAE:** `tax::Scalar`, `TaylorPolynomial`,
  `DensePolynomial`
- **`if constexpr`** for univariate (M == 1) vs multivariate branches
- **`[[nodiscard]]`** on accessors, computation results, expensive operations
- **Internal details in `detail` namespaces:** `tax::detail::kernels`,
  `tax::named::detail`

### Formatting

Enforced by `.clang-format` (Google style, customized):
- Indent: **4 spaces** (no tabs)
- Column limit: **100 characters**
- Brace wrapping: new line after class/struct/function/namespace/control statements
- Spaces inside parentheses and angle brackets: `TaylorExpansion< T, Scheme >`

```bash
clang-format -i $(git ls-files 'include/**/*.hpp')
```

---

## Testing

Tests are organized by module, one `.cpp` per concern, each registered via
`tax_add_test(name SOURCES path.cpp)` in `tests/CMakeLists.txt`.

```cpp
#include <gtest/gtest.h>
#include <tax/tax.hpp>

TEST(Trig, SinUnivariateOrder3)
{
    auto x = tax::TE<3>::variable(0.0);
    auto f = sin(x);
    EXPECT_NEAR(f[1], 1.0, 1e-12);
    EXPECT_NEAR(f[3], -1.0 / 6.0, 1e-12);
}
```

```bash
ctest --test-dir build --output-on-failure
./build/tests/test_trig          # single executable
```

DACE-comparison regression tests live in `tests/regression/` and only build
with `-DTAX_BUILD_REGRESSIONS=ON`.

---

## CI/CD

- **`tests.yml`** — push/PR: build + unit tests across the support matrix
- **`regressions.yml`** — DACE comparison suite
- **`sanitizers.yml`** — ASAN/UBSAN/TSAN jobs
- **`docs.yml`** — MkDocs documentation build/deploy

### Before Submitting a PR

1. All ctest targets pass locally
2. Code is formatted with `clang-format`
3. No new dynamic allocations introduced in the dense core library
4. New math operations have kernel tests AND operator tests

---

## Common Pitfalls

- **Do not heap-allocate:** `TaylorExpansion` must remain allocation-free
- **Do not break `constexpr`:** all index arithmetic stays compile-time; if a
  fast path needs runtime statics, guard it with `if !consteval` and keep the
  loop kernel as the constant-evaluation fallback
- **Graded-lex ordering is sacred:** the coefficient order (`flatIndex`) is
  relied on everywhere — including contiguous-degree-block tricks — never change it
- **Kernel config macros are in-header:** never re-introduce
  `TAX_USE_UNROLL`/`TAX_USE_STENCIL` as build-system definitions (ODR hazard)
- **M = 0 is invalid:** always assert or `static_assert` M >= 1
- **Include the umbrella header:** `<tax/tax.hpp>` (core + named + mixed + la) —
  not individual sub-headers
- **`pyproject.toml` is forward-looking:** there are no Python binding sources
  in the tree yet
