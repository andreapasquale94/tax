# CLAUDE.md — AI Assistant Guide for `tax`

## Project Overview

**tax** is a header-only C++23 library for **truncated multivariate Taylor
expansions** (Differential Algebra in the Berz / Makino sense).  Each value
is a polynomial in M variables truncated at total degree N; arithmetic and
math operations propagate complete Taylor jets in one pass.

- **Version:** 0.2.0
- **License:** BSD 3-Clause
- **C++ Standard:** C++23 (required)
- **Build system:** CMake (>= 3.20)
- **Linear-algebra backend:** Eigen 3.4+

---

## Architectural Pillars

The whole library hangs off two design decisions.

### 1. Dual sizing: one template, one math core

A single template handles both compile-time-fixed and runtime-fixed
sizes, modelled directly on `Eigen::Matrix<T, Rows, Cols>` with
`Eigen::Dynamic` (= -1) as the runtime-size sentinel.

```
TaylorExpansionT<T, int Order, int Vars>

  Order, Vars >= 0           →  static path:   Eigen::Matrix<T, monomialCount(N, M), 1>
  Order = Vars = Dynamic     →  dynamic path:  Eigen::VectorX<T>

TE<N>            = TaylorExpansionT<double, N, 1>
TEn<N, M>        = TaylorExpansionT<double, N, M>
DynTE<T = double> = TaylorExpansionT<T, Eigen::Dynamic, Eigen::Dynamic>
```

Mixed dynamism (one static, one dynamic) is rejected with a
`static_assert`.  Both configurations satisfy a single
`tax::TaxExpression` concept and feed identical **coefficient
kernels** (slice-aware Cauchy convolution, exp/log, sin/cos, sqrt,
...).  The static path is the C++ hot path with compile-time constant
sizes; the dynamic path is what Python bindings expose, with no
`std::variant` over an (N, M) grid and no JIT.

Each `TaxExpression` (storage or ET node) publishes four trait
constants matching Eigen's `RowsAtCompileTime` convention:

- `OrderAtCompileTime` (int) — template arg, possibly `Eigen::Dynamic`.
- `VarsAtCompileTime` (int) — same.
- `IsStatic` (bool) — `true` iff both above are non-`Dynamic`.
- `IsDynamic` (bool) — `!IsStatic`.

Mixed static/dynamic expressions are rejected at compile time via the
`SameKindExpression` concept.

### 2. Slice-streamed expression templates

ET nodes evaluate **degree-by-degree**, never whole-polynomial-at-once.
Two node categories:

- **View-like nodes** (`AddExpr`, `SubExpr`, `NegExpr`, `ScalarMulExpr`,
  `ScalarAddExpr`): allocate nothing.  Their `degreeSlice(d)` returns a
  custom `ParentSliceView` that holds a stable reference to the parent
  ET node and computes elements on demand by recursing into the parent's
  `coeffAtSlice(d, i)`.
- **Buffered nodes** (`MulExpr`, `DivExpr`, `SquareExpr`, `SqrtExpr`,
  `ExpExpr`, `LogExpr`, `SinCosExpr`, `SinhCoshExpr`): own a `coeffs_`
  buffer plus any auxiliary buffer the recurrence needs (e.g. cos
  alongside sin).  Their `advanceTo(d)` fills slices monotonically.

Driver loop in `tax::detail::streamingAssign` (called by
`.eval()` on either storage type):

```cpp
for (std::size_t d = 0; d <= dst.order(); ++d) {
    expr.advanceTo(d);
    auto out_d = dst.degreeSlice(d);
    auto in_d  = expr.degreeSlice(d);
    for (Eigen::Index i = 0; i < in_d.size(); ++i) {
        out_d.coeffRef(i) = in_d.coeff(i);
    }
}
```

The root assignment writes directly into the destination, so the top of
the tree allocates nothing either.

### Operand storage trait

`expr::etstore_t<E>` selects how each ET node stores an operand: by
**const&** for storage types (whose lifetime is owned by user code) and
**by value** for nested ET nodes (so intermediate temporaries from
chained operators don't dangle past their constructor's scope).

---

## Repository Layout

```
tax/
├── include/tax/
│   ├── tax.hpp              # umbrella header (users include only this)
│   ├── fwd.hpp              # forward declarations
│   ├── concepts.hpp         # TaylorExpansion / StreamingExpression
│   ├── util/                # multi-index utilities
│   ├── kernels/             # slice-aware coefficient kernels
│   ├── storage/             # static + dynamic TTE storage types
│   ├── expr/                # ET base + view-like + buffered nodes
│   └── ops/                 # arithmetic, math free funcs
├── tests/                   # GoogleTest suite
├── python/                  # nanobind Python bindings (tax-python)
│   ├── CMakeLists.txt
│   ├── src/tax_module.cpp
│   ├── tax/__init__.py
│   └── tests/test_dyn_te.py
├── CMakeLists.txt
├── CLAUDE.md
└── LICENSE
```

---

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `TAX_BUILD_TEST` | `ON` | Build the GoogleTest suite |
| `TAX_BUILD_PYTHON` | `OFF` | Build the nanobind Python module (`tax`) |

### Dependencies

- **Required:** Eigen 3.4+ (system package or `CMAKE_PREFIX_PATH`).
- **Test framework:** Google Test (fetched via FetchContent).
- **Python bindings (optional):** `pip install nanobind`, then configure
  with `-DTAX_BUILD_PYTHON=ON`.

---

## Public API

```cpp
#include <tax/tax.hpp>

auto x        = tax::TE<5>::variable(1.0);
auto [u, v]   = tax::TEn<3, 2>::variables(std::array<double, 2>{1.0, 2.0});
auto x_const  = tax::TE<5>::constant(3.0);
auto z        = tax::TE<5>::zero();
auto o        = tax::TE<5>::one();

auto f = u * tax::sin(v) + u * v;       // ET expression, no allocation yet
tax::TEn<3, 2> result;
result = (f).eval();                            // drives the streaming sweep

result.value();
result.coeff({1, 0});
result.derivative({1, 1});
result.eval(std::array{0.1, 0.05});
result.coeffsNormInf();
result.coeffsNorm<2>();
```

Free functions: `+`, `-`, `*`, `/`, unary `-`, scalar combinations,
`tax::sin`, `tax::cos`, `tax::tan`, `tax::sinh`, `tax::cosh`,
`tax::tanh`, `tax::exp`, `tax::log`, `tax::sqrt`, `tax::square`,
`tax::cube`.

The dynamic path is `DynTE<double>` (alias for
`TaylorExpansionT<double, Eigen::Dynamic, Eigen::Dynamic>`).  Its
factories take `(value, order, nvars[, var_idx])` since sizes are
runtime-fixed at construction.

---

## Code Conventions

| Category | Convention | Examples |
|----------|-----------|----------|
| Types | `PascalCase` | `TaylorExpansionT`, `MulExpr`, `ParentSliceView` |
| Free functions / methods | `camelCase` | `monomialCount`, `degreeSlice`, `advanceTo`, `coeffAtSlice`, `streamingAssign` |
| Local variables | `snake_case` | `out_d`, `inv_b0`, `alpha_buf` |
| Template parameters | uppercase short names | `T`, `N`, `M`, `E`, `L`, `R` |
| Aliases | short upper-case | `TE<N>`, `TEn<N,M>`, `DynTE<T>` |
| Compile-time trait constants    | Eigen-style `XxxAtCompileTime` / `IsXxx` | `OrderAtCompileTime`, `VarsAtCompileTime`, `IsStatic`, `IsDynamic` |
| Detail namespaces | `tax::detail`, `tax::expr::detail`, `tax::kernels::detail` | |

C++ patterns:

- `[[nodiscard]]` on accessors and pure operations.
- `noexcept` everywhere it's actually guaranteed.
- `mutable` on buffered ETs' coefficient/state members so a buffered
  node can sit behind a `const&` in a parent ET tree.
- `if constexpr` for static-vs-dynamic branching.
- `std::span<const std::size_t>` is the universal multi-index
  parameter type.

Formatting follows `.clang-format` (Google style, 4-space indent,
100-column limit, opening braces on new lines).

---

## Testing

Tests are organised by concern, each producing a separate executable:

| File | Coverage |
|------|---------|
| `test_multi_index.cpp` | combinatorics, flatIndex / unflatIndex, iteration |
| `test_storage.cpp` | static TTE factories, accessors, eval, norms |
| `test_arithmetic.cpp` | +, -, *, / and scalar variants |
| `test_math.cpp` | sin, cos, sinh, cosh, exp, log, sqrt, square, cube, tan |
| `test_dynamic.cpp` | DynTE factories + a few ops |

Run all: `ctest --test-dir build --output-on-failure`.
Run one: `./build/tests/test_math`.

---

## Adding a New Mathematical Function

1. **Kernel** — write the slice-aware degree-by-degree recurrence in the
   appropriate header under `include/tax/kernels/`, taking operand
   slice-providers and a destination slice-provider.
2. **Buffered ET node** — add a class in `include/tax/expr/buffered_nodes.hpp`
   that owns its `coeffs_` buffer (and any auxiliaries) and dispatches
   to the kernel during `advanceTo`.
3. **Operator** — expose a free function in `include/tax/ops/math.hpp`
   constrained on `TaxExpression`.
4. **Tests** — add coverage in `tests/test_math.cpp` (or a new file
   if substantial); register it in `tests/CMakeLists.txt`.

---

## Non-goals (explicit)

- Not a chain-rule-on-tangents AD library; this is polynomial algebra.
- Buffered ET nodes structurally **must** allocate (Cauchy convolution
  requires operand history).  Don't try to remove `coeffs_` from them.
- No mixed static/dynamic expressions.  The `SameKindExpression`
  concept rejects them at compile time.
- ODE / ADS / DACE benchmarks are out of scope for the current state of
  the library and will be re-introduced on top of this foundation in
  subsequent commits.

## Python bindings

`python/` contains the nanobind module (`tax`) exposing
`TaylorExpansionT<double, Eigen::Dynamic, Eigen::Dynamic>` (i.e.
`tax::DynTE<double>`) as the `tax.DynTE` return type plus the math
free functions.  Build with `-DTAX_BUILD_PYTHON=ON`.
Construction goes through **module-level utility functions**
(`tax.zero`, `tax.one`, `tax.constant`, `tax.variable`,
`tax.variables`); `tax.DynTE` itself is not directly constructible from
Python.  Arithmetic operators internally evaluate the C++ ET
expressions into a fresh `DynTE` per call (Python cannot meaningfully
own lazy ET temporaries across statements).  The static-extent C++ path
is intentionally not exposed; no `std::variant` over an (Order, Vars)
grid, no JIT.
