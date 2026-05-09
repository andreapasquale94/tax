# Changelog

## 0.2.0 — 2026-05-09

Ground-up rewrite. The prior 0.1.x layout (ADS, ODE, separate
expression flattening, kernel hierarchy) is discarded; this release
is the new foundation everything else will build on.

### Added

- **Slice-streamed expression templates.** View-like nodes (`AddExpr`,
  `SubExpr`, `NegExpr`, `ScalarMulExpr`, `ScalarAddExpr`) allocate
  nothing; buffered nodes (`MulExpr`, `DivExpr`, `SquareExpr`,
  `SqrtExpr`, `ExpExpr`, `LogExpr`, `SinCosExpr`, `SinhCoshExpr`)
  fill `coeffs_` monotonically through `advanceTo`.
- **Two storage types**, sharing one math core:
  `TruncatedTaylorExpansionT<T, N, M>` (static, `Eigen::Matrix`) and
  `DynamicTaylorExpansion<T>` (runtime-sized, `Eigen::VectorX`).
- **`<<=` driver** that streams an ET into a destination without
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
- **Documentation site** (this MkDocs build).

### Math coverage

- Arithmetic: `+`, `-`, `*`, `/`, unary `-`, scalar variants.
- Math: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `exp`, `log`,
  `sqrt`, `square`, `cube`.

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
