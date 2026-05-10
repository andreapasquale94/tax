# Changelog

## 0.2.0 — 2026-05-09

Ground-up rewrite. The prior 0.1.x layout (ADS, ODE, separate
expression flattening, kernel hierarchy) is discarded; this release
is the new foundation everything else will build on.

### Added

- **Unified `TaylorExpansionT<T, Order, Vars>` storage template.**
  Models `Eigen::Matrix<T, Rows, Cols>`: signed-int template
  parameters with `Eigen::Dynamic` (= -1) as the runtime-size
  sentinel. The previous `TruncatedTaylorExpansionT` (static) and
  `DynamicTaylorExpansion` (dynamic) classes are gone, collapsed into
  this single template. Aliases `TE<N>`, `TEn<N, M>`, `DynTE<T>` cover
  the same surface they did before. Mixed dynamism (one static,
  one dynamic) is rejected with a `static_assert`. Empty-base
  optimisation keeps the static path byte-for-byte identical to a
  raw `Eigen::Matrix` (asserted in `tests/test_storage.cpp`).
- **Eigen-style trait constants.** `kStatic` / `kOrder` / `kVars` are
  renamed to `IsStatic` / `OrderAtCompileTime` / `VarsAtCompileTime`,
  mirroring Eigen's `RowsAtCompileTime` convention. A new
  `IsDynamic = !IsStatic` constant is exposed alongside.
- **Slice-streamed expression templates.** View-like nodes (`AddExpr`,
  `SubExpr`, `NegExpr`, `ScalarMulExpr`, `ScalarAddExpr`) allocate
  nothing; buffered nodes (`MulExpr`, `DivExpr`, `SquareExpr`,
  `SqrtExpr`, `ExpExpr`, `LogExpr`, `SinCosExpr`, `SinhCoshExpr`)
  fill `coeffs_` monotonically through `advanceTo`.
- **Two storage types**, sharing one math core:
  `TruncatedTaylorExpansionT<T, N, M>` (static, `Eigen::Matrix`) and
  `DynamicTaylorExpansion<T>` (runtime-sized, `Eigen::VectorX`).
- **`.eval()` driver** that streams an ET into a destination without
  allocating at the root.
- **Compile-time index accessors**: `coeff<1, 0>()`,
  `derivative<1, 1>()` resolve flat-index and factorial at compile
  time.
- **`std::array` overloads** of `coeff`, `derivative`, `eval` so
  callers can write `result.coeff({1, 0})` directly.
- **`SameKindExpression` concept** rejecting mixed static/dynamic
  expressions at compile time.
- **nanobind Python bindings** (`-DTAX_BUILD_PYTHON=ON`) exposing
  `DynTE<double>` and the math free functions; arithmetic operators
  evaluate eagerly into a fresh `DynTE`. Construction goes through
  module-level utility functions (`tax.zero`, `tax.one`,
  `tax.constant`, `tax.variable`, `tax.variables`); the `DynTE` class
  is a return type only and is not directly constructible from
  Python.
- **Python wheels.** A `pyproject.toml` driven by `scikit-build-core`
  produces a manylinux_2_28 wheel for CPython 3.10 through 3.13.  The
  `wheels` GitHub Actions workflow runs `cibuildwheel` on every push
  and uploads the built wheels (plus an sdist) as downloadable
  artifacts.  No PyPI publishing — wheels are consumed straight from
  the Actions UI.
- **Documentation site** (this MkDocs build).

### Math coverage

- Arithmetic: `+`, `-`, `*`, `/`, unary `-`, scalar variants.
- Trig + hyperbolic: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`.
- Paired evaluators: `sincos(x)` and `sinhcosh(x)` return owner
  objects exposing `.sin()`/`.cos()` (resp. `.sinh()`/`.cosh()`)
  views that share one set of internal buffers — the second `.eval()`
  is a buffer copy.
- Inverse trig + hyperbolic: `asin`, `acos`, `atan`, `asinh`,
  `acosh`, `atanh`, `atan2(y, x)`.
- Exp / log: `exp`, `log`, `log10`.
- Roots, powers, hypot: `sqrt`, `cbrt`, `square`, `cube`,
  `pow<N>(x)` (compile-time integer), `pow(x, p)` (runtime real),
  `hypot(x, y)`, `hypot(x, y, z)`.
- Special: `erf`.

### Tests

- 47 GoogleTests across 5 executables (multi-index, storage,
  arithmetic, math, dynamic).
- 13 pytest cases for the Python bindings.
- All pass under both Release and ASan + UBSan.

### Out of scope (deferred)

- ODE / ADS / Eigen vector-of-TTE adapters / DACE comparison /
  benchmarks. They will be re-introduced on top of this foundation in
  later releases.
- Power with real or DA exponent, `atan` / `asin` / `acos` /
  `atanh` / `asinh` / `acosh`, `erf`, `atan2`, `hypot`, `abs`,
  `log10`, `cbrt`. Most are wired through composition once the
  underlying recurrence kernel lands.
