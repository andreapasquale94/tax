# Python JIT Layer — M5: Regression + Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock the correctness of the M0–M4 JIT layer with broad regression batteries (jit≡eager across many functions; eager vs closed-form/analytic oracles) and characterize its performance (a deterministic fusion-structural guard + a benchmark script comparing eager vs jit vs hand-written C++), plus ship both two-body maps as runnable, regression-locked examples.

**Architecture:** Pure test/benchmark/example additions on top of the finished M0–M4 layer — **no library changes**. Regression uses two oracle kinds: (a) **jit≡eager** (the fused result must equal the op-by-op eager result, to 1e-12) over a broad expression set, and (b) **closed-form/analytic** identities (Taylor series of `exp`/`log`/`sin`, identities like `sin²+cos²=1`, `exp(log(1+x))=1+x`) the eager layer must reproduce. Performance is captured deterministically (jit fuses an N-op function into exactly ONE compiled kernel vs eager's N) — avoiding flaky wall-time assertions — with a separate human-run benchmark script for the actual timings.

**Tech Stack:** Python ≥3.10 (numpy + pytest + stdlib `time`), the finished M0–M4 layer (`tax`), the system C++23 compiler + Eigen + `tax` headers (for the toolchain-dependent regression/benchmark).

## Global Constraints

- **Builds on M0–M4** (branch `feature/python-jit-expansions`, all merged; 91 tests passing). **No library code changes** — M5 adds only tests, a benchmark script, and an examples module.
- **Test runner:** `cd /Users/andrea/Documents/Codes/tax/python && .venv/bin/python -m pytest ...`. Toolchain-dependent tests/benchmarks import `needs_toolchain` from `tests._helpers`.
- **Two oracle kinds, both independent of the code under test:** jit≡eager (fusion preserves semantics) and closed-form/analytic (absolute correctness vs math). Never assert a value against a recomputation through the same code path.
- **No flaky timing assertions in CI.** The perf *guard* is the deterministic fusion-structure test (compile-count: jit=1, eager>1). Wall-time numbers live in a standalone script run by hand, documented qualitatively.
- **Examples are regression-locked:** the `examples/` module is imported and asserted by a test, so it can never silently rot.
- **All prior constraints hold** (static storage, pure-Python base, graded-lex, the documented jit over-promotion, etc.).
- **Out of scope (optional follow-on):** `batch=K` lock-step multi-point evaluation (it changes the emitted scalar type to `Batch<double,K>` — a real codegen change deserving its own focused plan); DACE-suite cross-check of the Python path (the C++ core is already DACE-regressed; the Python path is C++-cross-checked structurally in M2/M3); GPU/SIMD.

---

## File Structure

```
python/
├── tests/
│   ├── test_regression_jit_eager.py   # CREATE: broad jit≡eager battery
│   ├── test_regression_closed_form.py # CREATE: eager vs closed-form/analytic oracles
│   ├── test_perf_fusion.py            # CREATE: deterministic fusion-structure guard
│   └── test_examples.py               # CREATE: imports + asserts the examples module
├── bench/
│   └── bench_two_body.py              # CREATE: standalone timing script (eager vs jit vs C++)
└── examples/
    └── two_body.py                    # CREATE: runnable, documented two-body maps (unnamed + named)
```

---

### Task 1: jit≡eager regression battery

**Files:**
- Create: `python/tests/test_regression_jit_eager.py`

**Interfaces:**
- Consumes: the public surface (`tax.variable`/`variables` with/without `name=`, the math functions, operators, `**`, `concatenate`, `dot`/`norm`, `@tax.jit`).
- Produces: a parametrized `@needs_toolchain` battery asserting `jit(f)(args).numpy() == f(args).numpy()` (eager) to 1e-12 over a diverse set of scalar, multivariate, vector, and named functions.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_regression_jit_eager.py
import numpy as np
import pytest
import tax
from tests._helpers import needs_toolchain

# Each case: (builder, make_inputs). builder(*inputs) composes a computation that
# works on either eager handles or jit tracers; we run it eagerly and under @tax.jit
# and assert the fused result equals the op-by-op result.
SCALAR_CASES = {
    "sin_exp":      lambda x: tax.sin(x) * tax.exp(x),
    "deep_mix":     lambda x: tax.sin(tax.exp(x)) * tax.log(2.0 + x) - tax.atan(x),
    "pow_int":      lambda x: (x - 2.0) ** 3 + (x + 1.0) ** 2,
    "pow_real":     lambda x: (x * x + 1.0) ** 1.5,
    "ratio":        lambda x: tax.tanh(x) / (1.0 + tax.cosh(x)),
    "transcend":    lambda x: tax.erf(x) + tax.atanh(x / 4.0) - tax.cbrt(2.0 + x),
}

@needs_toolchain
@pytest.mark.parametrize("name", list(SCALAR_CASES))
def test_jit_equals_eager_scalar(name):
    f = SCALAR_CASES[name]
    x = tax.variable(0.3, order=6)
    eager = f(x)
    jitted = tax.jit(f)(x)
    np.testing.assert_allclose(jitted.numpy(), eager.numpy(), atol=1e-12, rtol=0)

@needs_toolchain
def test_jit_equals_eager_vector_named():
    def f(x, p):
        r = tax.norm(x)
        return tax.concatenate([x[0] * p, x[1] / (1.0 + r), tax.sin(x[0] + x[1])])
    x = tax.variables([1.0, 2.0], order=4, name="x")
    p = tax.variable(0.5, order=4, name="p")
    eager = f(x, p)
    jitted = tax.jit(f)(x, p)
    assert eager.scheme == jitted.scheme
    np.testing.assert_allclose(jitted.numpy(), eager.numpy(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(jitted.jacobian("x"), eager.jacobian("x"), atol=1e-12, rtol=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_regression_jit_eager.py -v`
Expected: FAIL — file does not exist yet (collection error). After creating it, the tests RUN (toolchain present) and PASS; if any case mismatches, that signals a real fusion bug — investigate, don't loosen the tolerance.

- [ ] **Step 3: (No implementation — this is a regression battery over the finished layer)**

The test file IS the deliverable. If a case fails, the bug is in M1–M4 library code (escalate); do not modify the test to pass.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_regression_jit_eager.py -v`
Expected: PASS (7 cases). Full suite: `cd python && .venv/bin/python -m pytest -q` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_regression_jit_eager.py
git commit -m "test(py): jit-equals-eager regression battery (scalar/vector/named)"
```

---

### Task 2: Closed-form / analytic regression battery

**Files:**
- Create: `python/tests/test_regression_closed_form.py`

**Interfaces:**
- Consumes: the eager public surface.
- Produces: a `@needs_toolchain` battery asserting eager results against known Taylor series and analytic identities (independent of the implementation) — the absolute-correctness lock.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_regression_closed_form.py
import math
import numpy as np
import tax
from tests._helpers import needs_toolchain

@needs_toolchain
def test_exp_series():
    x = tax.variable(0.0, order=6)
    # exp(x) Maclaurin coeffs are 1/k!
    expected = np.array([1.0 / math.factorial(k) for k in range(7)])
    np.testing.assert_allclose(tax.exp(x).numpy(), expected, atol=1e-12)

@needs_toolchain
def test_log1p_series():
    x = tax.variable(0.0, order=5)
    # log(1+x) = x - x^2/2 + x^3/3 - x^4/4 + x^5/5
    expected = np.array([0.0, 1.0, -0.5, 1.0 / 3, -0.25, 0.2])
    np.testing.assert_allclose(tax.log(1.0 + x).numpy(), expected, atol=1e-12)

@needs_toolchain
def test_pythagorean_identity():
    x = tax.variable(0.7, order=6)
    s2c2 = tax.sin(x) * tax.sin(x) + tax.cos(x) * tax.cos(x)
    expected = np.zeros(7); expected[0] = 1.0          # identically 1
    np.testing.assert_allclose(s2c2.numpy(), expected, atol=1e-12)

@needs_toolchain
def test_exp_log_inverse():
    x = tax.variable(0.0, order=5)
    # exp(log(1+x)) == 1 + x exactly (as a truncated series)
    expected = np.zeros(6); expected[0] = 1.0; expected[1] = 1.0
    np.testing.assert_allclose(tax.exp(tax.log(1.0 + x)).numpy(), expected, atol=1e-12)

@needs_toolchain
def test_tanh_equals_sinh_over_cosh():
    x = tax.variable(0.4, order=6)
    np.testing.assert_allclose(tax.tanh(x).numpy(),
                               (tax.sinh(x) / tax.cosh(x)).numpy(), atol=1e-12)

@needs_toolchain
def test_multivariate_product_rule():
    # f = x0 * x1 at (a,b): value a*b; gradient [b, a]; mixed 2nd partial 1
    X = tax.variables([3.0, 5.0], order=2)
    f = X[0] * X[1]
    assert f.value() == 15.0
    np.testing.assert_allclose(f.gradient(), [5.0, 3.0], atol=1e-12)
    assert f.derivative(1, 1) == 1.0          # d²/dx0dx1 (x0 x1) = 1

@needs_toolchain
def test_named_chain_rule_sin_of_product():
    # g = sin(x*p) ; ∂g/∂x = p cos(x p), ∂g/∂p = x cos(x p) at (x,p)
    x = tax.variable(0.5, order=3, name="x")
    p = tax.variable(2.0, order=3, name="p")
    g = tax.sin(x * p)
    c = math.cos(0.5 * 2.0)
    np.testing.assert_allclose(g.gradient("x"), [2.0 * c], atol=1e-12)   # p cos(xp)
    np.testing.assert_allclose(g.gradient("p"), [0.5 * c], atol=1e-12)   # x cos(xp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_regression_closed_form.py -v`
Expected: FAIL (collection error) before the file exists; after creating it, RUN + PASS. A mismatch signals a real numerical bug — investigate.

- [ ] **Step 3: (No implementation — regression battery)**

The test file is the deliverable.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_regression_closed_form.py -v`
Expected: PASS (7 tests). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_regression_closed_form.py
git commit -m "test(py): closed-form/analytic regression battery (series + identities)"
```

---

### Task 3: Fusion structural guard + benchmark script

**Files:**
- Create: `python/tests/test_perf_fusion.py`
- Create: `python/bench/bench_two_body.py`

**Interfaces:**
- Produces (test): a deterministic `@needs_toolchain` guard that a multi-op function under `@tax.jit` compiles **exactly one** kernel, while the same computation eagerly compiles **more than one** — proving fusion collapses N ops into 1 kernel (and 1 FFI crossing) without timing flakiness.
- Produces (script): a standalone `bench_two_body.py` (run by hand) that times eager vs jit vs a hand-written C++ baseline for the two-body RHS and prints a table.
- Consumes: `tax._frontend.eager._KERNEL_CACHE`, `tax._codegen.build.compile_kernel`, the public surface, `tax._codegen.{build,load}` (for the C++ baseline).

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_perf_fusion.py
import tax
from tax._frontend import eager
from tax._codegen import build
from tests._helpers import needs_toolchain

@needs_toolchain
def test_jit_fuses_multi_op_into_single_kernel(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))   # fresh on-disk cache
    count = {"n": 0}
    real = build.compile_kernel
    def counting(*a, **k):
        count["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(build, "compile_kernel", counting)

    x = tax.variable(0.0, order=4)

    # Eager: sin, exp, mul -> three distinct (op, scheme) kernels compiled.
    eager._KERNEL_CACHE.clear()
    start = count["n"]
    _ = tax.sin(x) * tax.exp(x)
    eager_compiles = count["n"] - start
    assert eager_compiles >= 2          # op-by-op compiles multiple kernels

    # JIT: the whole function fuses into ONE kernel.
    eager._KERNEL_CACHE.clear()
    @tax.jit
    def f(x):
        return tax.sin(x) * tax.exp(x)
    start = count["n"]
    _ = f(x)
    jit_compiles = count["n"] - start
    assert jit_compiles == 1            # fusion -> a single compiled kernel
    assert jit_compiles < eager_compiles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_perf_fusion.py -v`
Expected: FAIL (collection error) before the file exists; after creating it, RUN + PASS. If `jit_compiles != 1`, fusion is broken (the whole function should be one graph → one kernel) — investigate.

- [ ] **Step 3: Write the benchmark script**

```python
# python/bench/bench_two_body.py
"""Manual benchmark: eager vs @tax.jit vs hand-written C++ for the planar two-body RHS.

Run from python/:  .venv/bin/python bench/bench_two_body.py
(Requires a C++23 compiler + Eigen, like the test suite. Not a pytest test —
the deterministic fusion guard lives in tests/test_perf_fusion.py.)
"""
import os, time, pathlib
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("TAX_INCLUDE", str(REPO / "include"))
os.environ.setdefault("TAX_CACHE_DIR", str(REPO / "python" / ".tax_cache"))

import tax
from tax._codegen import build, load

MU = 398600.4418
N_CALLS = 2000

def two_body(x, mu):
    r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
    return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])

# Hand-written C++ baseline: the same RHS as one fused extern "C" kernel.
_BASELINE = r'''
#include <tax/tax.hpp>
#include <algorithm>
using namespace tax;
extern "C" int tax_kernel(const double* const* ins, double* const* outs) noexcept {
    using E = TaylorExpansion<double, IsotropicScheme<4, 4>>;
    E::Data d; std::copy_n(ins[0], E::nCoefficients, d.data());
    E rx{d}; std::copy_n(ins[1], E::nCoefficients, d.data());
    E ry{d}; std::copy_n(ins[2], E::nCoefficients, d.data());
    E vx{d}; std::copy_n(ins[3], E::nCoefficients, d.data());
    E vy{d};
    E r3 = pow(rx * rx + ry * ry, 1.5);
    E a0 = vx, a1 = vy, a2 = (-MU_VAL) * rx / r3, a3 = (-MU_VAL) * ry / r3;
    std::copy_n(a0.coefficients().data(), E::nCoefficients, outs[0]);
    std::copy_n(a1.coefficients().data(), E::nCoefficients, outs[1]);
    std::copy_n(a2.coefficients().data(), E::nCoefficients, outs[2]);
    std::copy_n(a3.coefficients().data(), E::nCoefficients, outs[3]);
    return 0;
}
'''.replace("MU_VAL", repr(MU))

def _time(label, fn, n=N_CALLS):
    fn()                                  # warm up (compile / cache)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1e6   # microseconds per call
    print(f"  {label:<28} {dt:8.2f} us/call")
    return dt

def main():
    x = tax.variables([1.0, 0.0, 0.0, 1.0], order=4)
    jitted = tax.jit(lambda t, x, mu: two_body(x, mu))

    # hand C++ baseline
    so = build.compile_kernel(_BASELINE, "bench_two_body_baseline", cxx=build.find_compiler(),
                              includes=build.include_dirs(), opt_flags=["-O3"])
    fn = load.load_kernel(so)
    rows = [x[i].coeffs for i in range(4)]
    n = x.scheme.n_coeff
    def call_cpp():
        load.call_kernel(fn, rows, [n, n, n, n])

    print(f"Planar two-body RHS, order 4, {N_CALLS} warm calls each:")
    _time("eager (per-op FFI)", lambda: two_body(x, MU))
    _time("jit (fused, 1 FFI)", lambda: jitted(0.0, x, MU))
    _time("hand-written C++ baseline", call_cpp)
    print("Expectation: jit ≈ C++ baseline, both well below eager (which pays one FFI per op).")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test (and smoke-run the script)**

Run: `cd python && .venv/bin/python -m pytest tests/test_perf_fusion.py -v`
Expected: PASS. Then smoke-run the script once: `cd python && .venv/bin/python bench/bench_two_body.py` — it prints a 3-row table (eager / jit / C++ µs-per-call); confirm it runs without error and jit is in the same ballpark as the C++ baseline and faster than eager. Full suite: `cd python && .venv/bin/python -m pytest -q` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_perf_fusion.py python/bench/bench_two_body.py
git commit -m "test+bench(py): deterministic fusion guard + eager/jit/C++ two-body benchmark"
```

---

### Task 4: Worked, regression-locked examples

**Files:**
- Create: `python/examples/two_body.py`
- Create: `python/tests/test_examples.py`

**Interfaces:**
- Produces (`examples/two_body.py`): `unnamed_rhs() -> tax.Array` and `named_rhs() -> tax.Array` — the two north-star maps as documented, runnable functions; plus a `main()` that prints their value and Jacobian. Runnable as `python examples/two_body.py`.
- Produces (test): `test_examples.py` imports the example functions and asserts their values/Jacobians, so the documented examples can never silently rot.
- Consumes: the public surface.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_examples.py
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
import two_body                                   # examples/two_body.py
from tests._helpers import needs_toolchain

MU = 398600.4418

@needs_toolchain
def test_unnamed_example_rhs():
    dx = two_body.unnamed_rhs()
    np.testing.assert_allclose(dx.value(), [0.0, 1.0, -MU, 0.0], rtol=1e-9, atol=1e-6)

@needs_toolchain
def test_named_example_rhs():
    dx = two_body.named_rhs()
    np.testing.assert_allclose(dx.value(), [0.0, 1.0, -MU, 0.0], rtol=1e-9, atol=1e-6)
    # ∂(rhs)/∂mu = [0, 0, -rx/r^3, -ry/r^3] = [0,0,-1,0] at the unit-radius state
    np.testing.assert_allclose(dx.jacobian("mu"), [[0.0], [0.0], [-1.0], [0.0]],
                               rtol=1e-9, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_examples.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'two_body'` (the examples module does not exist yet).

- [ ] **Step 3: Write the examples module**

```python
# python/examples/two_body.py
"""Planar restricted two-body RHS as a tax.jit-compiled map — the north-star example.

The state is [rx, ry, vx, vy]; the RHS is [vx, vy, -mu rx / r^3, -mu ry / r^3]
with r = sqrt(rx^2 + ry^2). Two flavors:
  * unnamed_rhs(): integer-indexed coordinates, gravitational parameter mu as a
    plain float baked into the kernel.
  * named_rhs(): named axes "x" (the 4-D state) and "mu" (the parameter), so the
    result carries a named state-transition block jacobian("x") and a parameter
    sensitivity jacobian("mu").

Run:  python examples/two_body.py
"""
import os, pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("TAX_INCLUDE", str(REPO / "include"))
os.environ.setdefault("TAX_CACHE_DIR", str(REPO / "python" / ".tax_cache"))

import tax

MU = 398600.4418
STATE = [1.0, 0.0, 0.0, 1.0]          # rx, ry, vx, vy (unit-radius circular-ish)

def _rhs(x, mu):
    r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
    return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])

def unnamed_rhs():
    """Order-4 expansion in the 4 state coordinates; mu is a baked constant."""
    x = tax.variables(STATE, order=4)

    @tax.jit
    def rhs(t, x, mu):
        return _rhs(x, mu)

    return rhs(0.0, x, MU)

def named_rhs():
    """Named axes 'x' (state) and 'mu' (parameter), pinned with a jit signature."""
    x = tax.variables(STATE, order=4, name="x")
    mu = tax.variable(MU, order=4, name="mu")

    @tax.jit([tax.f64, tax.ArrayType(order=4, size=4, name="x"),
              tax.ExpansionType(order=4, name="mu")])
    def rhs(t, x, mu):
        return _rhs(x, mu)

    return rhs(0.0, x, mu)

def main():
    dx = unnamed_rhs()
    print("unnamed RHS value:", dx.value())
    print("state-transition jacobian:\n", dx.jacobian())
    dn = named_rhs()
    print("named RHS value:", dn.value())
    print("d(rhs)/d(x) block:\n", dn.jacobian("x"))
    print("d(rhs)/d(mu) sensitivity:\n", dn.jacobian("mu"))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_examples.py -v`
Expected: PASS. Smoke-run the example: `cd python && .venv/bin/python examples/two_body.py` — prints the values + Jacobians without error. Full suite: `cd python && .venv/bin/python -m pytest -q` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add python/examples/two_body.py python/tests/test_examples.py
git commit -m "docs+test(py): worked two-body examples (unnamed + named), regression-locked"
```

---

## Self-Review

**Spec coverage (M5 = regression + perf, per the M4 roadmap):**
- jit≡eager regression (fusion preserves semantics) across a broad expression set → Task 1. ✓
- Absolute correctness vs closed-form/analytic oracles (series + identities, multivariate + named) → Task 2. ✓
- Performance characterization — deterministic fusion-structure guard (jit=1 kernel, eager>1) → Task 3 (test). ✓
- eager vs jit vs hand-written-C++ benchmark (the fusion win + first-touch vs warm) → Task 3 (script). ✓
- Both two-body maps as documented, runnable, regression-locked examples → Task 4. ✓
- No library changes (M5 is tests/bench/examples only). ✓
- Out of scope (documented): `batch=K` (separate focused plan — it parameterizes the emitted scalar type to `Batch<double,K>`); DACE-suite cross-check of the Python path; GPU/SIMD.

**Placeholder scan:** No "TBD"/"similar to Task N"/"add error handling". Every step has complete code and concrete oracles (all verified: `exp`=1/k!; `log(1+x)`=[0,1,-1/2,1/3,-1/4,1/5]; `sin²+cos²`=1; `exp(log(1+x))`=1+x; `tanh=sinh/cosh`; product/chain-rule gradients; the two-body value/Jacobian).

**Type consistency:** Uses only the established public surface (`tax.variable`/`variables`/`jit`/`f64`/`ExpansionType`/`ArrayType`/`concatenate`/`norm`/math fns; `Expansion.value`/`numpy`/`gradient`/`derivative`; `Array.value`/`numpy`/`jacobian`) and internal hooks `eager._KERNEL_CACHE`, `build.compile_kernel`/`find_compiler`/`include_dirs`, `load.load_kernel`/`call_kernel` — all matching their M1–M4 definitions. The benchmark's hand-C++ baseline mirrors the emitted-kernel ABI (`ins`/`outs` as `double**`).

---

## Roadmap (after M5)

- **M6 — Packaging:** vendored `tax`/Eigen headers in a `py3-none-any` wheel; `cffi` (API mode) FFI to cut per-call overhead; a shipped precompiled header for warm builds; compiler-discovery docs; `pyproject.toml` finalization; the worked examples surfaced in the docs site.
- **Optional — `batch=K`:** lock-step multi-point evaluation. A focused plan: parameterize the emitted scalar type (`double` → `Batch<double,K>`), widen the input/output buffers to K lanes, and add `tax.jit(batch=K)` + an eager batched factory. Worthwhile if Monte-Carlo / many-point evaluation becomes a use case.
```
