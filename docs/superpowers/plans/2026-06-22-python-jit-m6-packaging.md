# Python JIT Layer — M6: Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the M0–M5 JIT layer an installable, documented, self-contained package — precompiled-header warm builds (~6× faster first-touch compile), `tax` headers vendored into the wheel (so it works without `TAX_INCLUDE`), and a usage README.

**Architecture:** Mostly packaging/docs over the finished layer. The one library change is PCH support in `build.py`: a `<tax/tax.hpp>` precompiled header is built once per (compiler, flags) and force-included on kernel compiles — a **pure flags-level addition** (verified: a kernel TU that keeps its `#include <tax/tax.hpp>` compiles fine with `-include-pch` because the header is `#pragma once`, so `emit` is unchanged, and the resulting `.so` is equivalent with or without the PCH, so the kernel cache key is unchanged). Vendoring is a build hook that copies `include/tax` into `tax/_vendor/include/tax` at wheel-build time; `find_tax_include()` already falls back to that vendored path.

**Tech Stack:** Python ≥3.10 (numpy + pytest + hatchling), the finished M0–M5 layer, the system C++23 compiler + Eigen + `tax` headers.

## Global Constraints

- **Builds on M0–M5** (branch `feature/python-jit-expansions`, all merged; 108 tests passing). Reuse existing modules.
- **Test runner:** `cd /Users/andrea/Documents/Codes/tax/python && .venv/bin/python -m pytest ...`. Toolchain-dependent tests import `needs_toolchain` from `tests._helpers`.
- **Pure-Python wheel.** The base package stays pure-Python (`py3-none-any`) — no compiled extension on the import/call path; the only native code is the JIT'd kernel. (cffi *API mode* would add a compiled shim and break this — deferred; see Roadmap.)
- **PCH is opt-out, with graceful fallback.** Enabled by default; `TAX_USE_PCH=0` disables it. If the PCH build fails for any reason, kernel compilation silently proceeds *without* it (never an error). The PCH must be built with the **same `opt_flags`** as the kernel compile (clang bakes `__OPTIMIZE__` into the PCH — a `-O3`/no-`-O3` mismatch is a hard error). The kernel cache key is **unchanged** by PCH (`flags_for_key` does not include the `-include-pch` marker), so PCH and non-PCH share the same cached `.so`.
- **Vendor the `tax` headers** (small, ours) into `tax/_vendor/include/tax` at build; rely on **discovered Eigen** (`find_eigen_include`: `TAX_EIGEN_INCLUDE` → `pkg-config eigen3` → common dirs). Eigen is a documented runtime prerequisite, not vendored (its full header tree is heavy; full self-containment is an optional follow-on).
- **`tax/_vendor/` is git-ignored** (synced at build, never committed — avoids a drifting duplicate of `include/tax`).
- **Runtime prerequisite:** a C++23 compiler discoverable via `TAX_CXX`/`CXX`/`PATH`, plus Eigen discoverable. Documented in the README.
- **All prior constraints hold** (static storage, graded-lex, the documented jit over-promotion, etc.).
- **Out of scope (Roadmap):** cffi API-mode FFI (pure-Python tension; the M5 benchmark showed the jit↔C++ gap is Python-level, not just the ctypes call); vendoring Eigen for a fully offline wheel; `batch=K`.

---

## File Structure

```
python/
├── pyproject.toml                  # MODIFY: hatchling build hook + metadata (license/readme/classifiers)
├── hatch_build.py                  # CREATE: hatchling custom build hook -> calls the vendor sync
├── README.md                       # CREATE: install prereqs, quickstart, eager-vs-jit, perf, limitations
├── tax/
│   ├── _vendor/
│   │   └── __init__.py             # CREATE: sync_from_repo() -> copies include/tax into _vendor/include/tax
│   └── _codegen/
│       └── build.py                # MODIFY: pch_path() + -include-pch in compile_kernel (graceful, opt-out)
└── tests/
    ├── test_pch.py                 # CREATE: PCH numerics-unchanged + built-once + disable
    ├── test_vendor.py              # CREATE: sync populates _vendor; self-contained compile (TAX_INCLUDE unset)
    └── test_readme.py             # CREATE: README exists + references the key public API
```

---

### Task 1: PCH warm builds

**Files:**
- Modify: `python/tax/_codegen/build.py`
- Test: `python/tests/test_pch.py`

**Interfaces:**
- Produces (in `build.py`): `pch_path(cxx, includes, opt_flags) -> pathlib.Path | None` — builds (once, on-disk cached, atomic) a `<tax/tax.hpp>` precompiled header with the given `opt_flags`; returns its path, or `None` if disabled (`TAX_USE_PCH=0`) or the build fails. `compile_kernel` force-includes it via `-include-pch <path>` when non-None.
- Consumes: `cache_dir`, `cache_key`, `compiler_id`, `flags_for_key`, `STD_FLAG` (existing).

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_pch.py
import numpy as np
import tax
from tax._codegen import build
from tests._helpers import needs_toolchain

@needs_toolchain
def test_pch_kernel_numerics_unchanged_and_built_once(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TAX_USE_PCH", raising=False)        # default = enabled
    from tax._frontend import eager
    eager._KERNEL_CACHE.clear()
    x = tax.variable(0.0, order=5)
    f = tax.sin(x)                                           # compiles a kernel (PCH used)
    np.testing.assert_allclose(f.numpy(),
                               [0, 1, 0, -1.0 / 6, 0, 1.0 / 120], atol=1e-12)
    g = tax.exp(x)                                           # second kernel reuses the same PCH
    np.testing.assert_allclose(g.numpy(),
                               [1, 1, 0.5, 1.0 / 6, 1.0 / 24, 1.0 / 120], atol=1e-12)
    pchs = list(tmp_path.glob("*.pch"))
    assert len(pchs) == 1                                    # PCH built exactly once, reused

@needs_toolchain
def test_pch_disabled_still_compiles(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TAX_USE_PCH", "0")                  # disabled
    from tax._frontend import eager
    eager._KERNEL_CACHE.clear()
    x = tax.variable(0.0, order=4)
    f = tax.exp(x)
    np.testing.assert_allclose(f.numpy(),
                               [1, 1, 0.5, 1.0 / 6, 1.0 / 24], atol=1e-12)
    assert list(tmp_path.glob("*.pch")) == []               # no PCH built when disabled

def test_pch_path_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("TAX_USE_PCH", "0")
    assert build.pch_path("c++", ["/x"], ["-O3"]) is None    # no compiler invoked when disabled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_pch.py -v`
Expected: FAIL — `AttributeError: module 'tax._codegen.build' has no attribute 'pch_path'`.

- [ ] **Step 3: Add PCH support to `build.py`**

```python
# add to python/tax/_codegen/build.py (near compile_kernel)
def _pch_enabled() -> bool:
    return os.environ.get("TAX_USE_PCH", "1") != "0"

def pch_path(cxx: str, includes: list[str], opt_flags: list[str]):
    """Path to a cached <tax/tax.hpp> PCH built with `opt_flags`, or None.

    Built once per (compiler, flags) and reused; the PCH must use the SAME
    opt_flags as the kernel compile (clang bakes __OPTIMIZE__ into it). Any
    failure returns None so kernel compilation proceeds without a PCH.
    """
    if not _pch_enabled():
        return None
    key = cache_key("__pch__:tax/tax.hpp", cid=compiler_id(cxx),
                    flags=flags_for_key(opt_flags))
    out = cache_dir() / f"{key}.pch"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=cache_dir()) as td:
            hdr = pathlib.Path(td) / "tax_all.hpp"
            hdr.write_text("#include <tax/tax.hpp>\n")
            tmp = pathlib.Path(td) / "tax.pch"
            cmd = [cxx, STD_FLAG, *opt_flags, "-x", "c++-header",
                   *[f"-I{i}" for i in includes], str(hdr), "-o", str(tmp)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                return None
            os.replace(tmp, out)        # atomic publish
        return out
    except OSError:
        return None
```

```python
# modify compile_kernel(): after the temp dir is set up, before building `cmd`,
# obtain the PCH and force-include it when available.
        tmp_so = pathlib.Path(td) / "kernel.so"
        pch = pch_path(cxx, includes, opt_flags)
        pch_flags = ["-include-pch", str(pch)] if pch is not None else []
        cmd = [cxx, STD_FLAG, *opt_flags, *pch_flags, "-shared", "-fPIC",
               *[f"-I{i}" for i in includes], str(cpp), "-o", str(tmp_so)]
```

(The kernel cache key is unchanged — `flags_for_key`/`cache_key` are not touched — so a `.so` compiled with or without the PCH is cache-equivalent. The PCH `.pch` lives in the same cache dir, keyed separately.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_pch.py -v`
Expected: PASS (numerics identical with PCH on; PCH built once; `TAX_USE_PCH=0` builds none). Full suite: `cd python && .venv/bin/python -m pytest -q` — no regressions. (First-touch compile is ~6× faster with the PCH — `<tax/tax.hpp>` cold ≈ 600 ms vs ≈ 100 ms warm on the reference machine — but this is a reported observation, not an asserted timing.)

- [ ] **Step 5: Commit**

```bash
git add python/tax/_codegen/build.py python/tests/test_pch.py
git commit -m "feat(py): precompiled-header warm builds (~6x first-touch compile), opt-out + graceful fallback"
```

---

### Task 2: Vendor `tax` headers into the wheel

**Files:**
- Create: `python/tax/_vendor/__init__.py`
- Create: `python/hatch_build.py`
- Modify: `python/pyproject.toml`
- Modify: `python/.gitignore`
- Test: `python/tests/test_vendor.py`

**Interfaces:**
- Produces: `tax._vendor.sync_from_repo() -> pathlib.Path` — copies the repo's `include/tax` tree into `tax/_vendor/include/tax` and returns the `tax/_vendor/include` path; raises `FileNotFoundError` if the repo `include/tax` isn't found (i.e. not building from the repo). The hatchling build hook calls it so a built wheel bundles the headers; `find_tax_include()` (existing) resolves `tax/_vendor/include` when `TAX_INCLUDE` is unset.
- Consumes: nothing (stdlib `shutil`/`pathlib`).

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_vendor.py
import pathlib
import pytest
import tax._vendor as vendor

def test_sync_from_repo_populates_vendor(tmp_path, monkeypatch):
    inc = vendor.sync_from_repo()                       # copy include/tax -> _vendor/include/tax
    inc = pathlib.Path(inc)
    assert (inc / "tax" / "tax.hpp").is_file()          # umbrella header vendored
    assert (inc / "tax" / "core" / "taylor_expansion.hpp").is_file()

def test_find_tax_include_uses_vendor_when_env_unset(monkeypatch):
    vendor.sync_from_repo()
    monkeypatch.delenv("TAX_INCLUDE", raising=False)
    from tax._codegen import build
    build.find_tax_include.cache_clear()               # lru_cache from M1
    resolved = pathlib.Path(build.find_tax_include())
    assert (resolved / "tax" / "tax.hpp").is_file()    # resolves the vendored copy
    build.find_tax_include.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_vendor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tax._vendor'`.

- [ ] **Step 3: Implement the sync helper, the build hook, and wire pyproject**

```python
# python/tax/_vendor/__init__.py
"""Vendored third-party/sibling headers for self-contained JIT compilation.

At wheel-build time, sync_from_repo() copies the repo's include/tax tree here so
the installed package can compile kernels without TAX_INCLUDE pointing at a source
checkout. This directory's contents are git-ignored (synced at build, not committed).
"""
from __future__ import annotations
import pathlib
import shutil

def _repo_include_tax() -> pathlib.Path:
    # _vendor/__init__.py -> _vendor -> tax -> python -> <repo root>
    return pathlib.Path(__file__).resolve().parents[3] / "include" / "tax"

def sync_from_repo() -> pathlib.Path:
    """Copy <repo>/include/tax into tax/_vendor/include/tax; return tax/_vendor/include."""
    src = _repo_include_tax()
    if not src.is_dir():
        raise FileNotFoundError(f"repo headers not found at {src} (not building from the repo?)")
    dst_root = pathlib.Path(__file__).resolve().parent / "include"
    dst = dst_root / "tax"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst_root
```

```python
# python/hatch_build.py
"""Hatchling build hook: vendor the tax headers into the wheel before file collection."""
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(self.root) ))   # ensure `tax` importable
        from tax._vendor import sync_from_repo
        sync_from_repo()
```

```toml
# python/pyproject.toml — add the build hook and richen metadata.
# Under [build-system] keep hatchling. Add:

[tool.hatch.build.targets.wheel.hooks.custom]
path = "hatch_build.py"

# And extend [project] with:
#   readme = "README.md"
#   license = { text = "BSD-3-Clause" }
#   keywords = ["taylor", "automatic-differentiation", "jit"]
#   classifiers = [
#     "Programming Language :: Python :: 3",
#     "License :: OSI Approved :: BSD License",
#     "Topic :: Scientific/Engineering :: Mathematics",
#   ]
# (Add these keys to the existing [project] table; do not duplicate name/version.)
```

```gitignore
# add to python/.gitignore
tax/_vendor/include/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_vendor.py -v`
Expected: PASS. Full suite green. (Optional manual check: `cd python && .venv/bin/pip wheel . -w /tmp/taxwheel --no-deps` then `unzip -l /tmp/taxwheel/tax-*.whl | grep _vendor/include/tax/tax.hpp` shows the header is bundled — a manual verification, not a CI test.)

- [ ] **Step 5: Commit**

```bash
git add python/tax/_vendor/__init__.py python/hatch_build.py python/pyproject.toml python/.gitignore python/tests/test_vendor.py
git commit -m "feat(py): vendor tax headers into the wheel (build hook) + package metadata"
```

---

### Task 3: Self-contained compile smoke (no `TAX_INCLUDE`)

**Files:**
- Test: `python/tests/test_self_contained.py`

**Interfaces:**
- Consumes: `tax._vendor.sync_from_repo`, the public surface, `tax._codegen.build`.
- Produces: the end-to-end vendoring guarantee — with the headers vendored and `TAX_INCLUDE` **unset**, a kernel compiles and runs correctly (the installed-package scenario, simulated in-repo).

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_self_contained.py
import numpy as np
import tax
import tax._vendor as vendor
from tax._codegen import build
from tests._helpers import needs_toolchain

@needs_toolchain
def test_compiles_from_vendored_headers_without_tax_include(tmp_path, monkeypatch):
    vendor.sync_from_repo()                               # populate tax/_vendor/include/tax
    monkeypatch.delenv("TAX_INCLUDE", raising=False)      # simulate an installed wheel
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    build.find_tax_include.cache_clear()
    from tax._frontend import eager
    eager._KERNEL_CACHE.clear()
    try:
        x = tax.variable(0.0, order=5)
        f = tax.sin(x) * tax.exp(x)                       # compiles using the vendored headers
        np.testing.assert_allclose(f.numpy(),
                                   [0, 1, 1, 1.0 / 3, 0, -1.0 / 30], atol=1e-12)
    finally:
        build.find_tax_include.cache_clear()              # don't leak the vendored path to other tests
```

- [ ] **Step 2: Run test to verify it fails (or passes once vendoring exists)**

Run: `cd python && .venv/bin/python -m pytest tests/test_self_contained.py -v`
Expected: PASS once Task 2's `sync_from_repo` exists and `find_tax_include` resolves the vendored path. If it FAILS with `TaxIncludeNotFound`, the vendored copy wasn't populated (debug Task 2's sync) or the cache wasn't cleared.

- [ ] **Step 3: (No new implementation — integration test over Task 1+2)**

The test is the deliverable. If it fails on compile, read the `JitCompileError` stderr — most likely a missing header in the vendored tree (confirm `sync_from_repo` copies the whole `include/tax` recursively).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_self_contained.py -v`
Expected: PASS. Full suite: `cd python && .venv/bin/python -m pytest -q` — no regressions (the `cache_clear()` in `finally` prevents leaking the vendored include path into other tests that rely on `TAX_INCLUDE`).

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_self_contained.py
git commit -m "test(py): self-contained kernel compile from vendored headers (no TAX_INCLUDE)"
```

---

### Task 4: README + usage docs

**Files:**
- Create: `python/README.md`
- Test: `python/tests/test_readme.py`

**Interfaces:**
- Produces: `python/README.md` — install prerequisites, quickstart (variables, eager, `@tax.jit`), eager-vs-jit guidance, the measured fusion win, the two-body example pointer, and the known limitations.
- Produces (test): `test_readme.py` — asserts the README exists and references the key public symbols, so it can't drift out of sync with the API.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_readme.py
import pathlib

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"

def test_readme_exists_and_covers_the_api():
    assert README.is_file()
    text = README.read_text()
    for token in ["tax.variable", "tax.variables", "@tax.jit", "tax.concatenate",
                  "jacobian", "C++23", "Eigen", "TAX_CXX"]:
        assert token in text, f"README missing reference to {token!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_readme.py -v`
Expected: FAIL — `AssertionError` / the README does not exist.

- [ ] **Step 3: Write the README**

```markdown
<!-- python/README.md -->
# tax (Python) — JIT-compiled Taylor expansions

A Python front-end for the **tax** C++ library: build truncated multivariate Taylor
expansions whose order, size, and axis names are chosen at *runtime*, while every
computation runs on the library's static `std::array` storage. The order/size you
pick in Python is JIT-compiled to a native kernel (reusing the C++ header kernels)
and cached on disk — so there is no fixed grid of pre-instantiated types and no
Python type explosion.

## Requirements

- Python ≥ 3.10, NumPy.
- A **C++23 compiler** discoverable at runtime (`TAX_CXX`, then `CXX`, then
  `c++`/`clang++`/`g++` on `PATH`).
- **Eigen** headers discoverable (`TAX_EIGEN_INCLUDE`, then `pkg-config eigen3`,
  then common system dirs).

The base package is pure-Python; the only native code is the JIT'd kernel, loaded
via `ctypes`. Compiled kernels are cached under `TAX_CACHE_DIR` (default
`~/.cache/tax`); the first use of a new computation pays a one-time compile (cut
~6× by a precompiled header — disable with `TAX_USE_PCH=0`).

## Quickstart

```python
import tax

# A univariate variable, order 5, expanded at x0 = 1.0
x = tax.variable(1.0, order=5)
f = tax.sin(x) * tax.exp(x)        # eager: each op is a cached kernel
f.value()                          # constant term
f.coeff(2)                         # raw Taylor coefficient
f.derivative(2)                    # k!-scaled derivative

# Multivariate coordinate variables + a vector map
X = tax.variables([1.0, 2.0], order=4)
Y = tax.concatenate([X[0] * X[1], X[0] / X[1]])
Y.value()                          # -> np.array([2.0, 0.5])
Y.jacobian()                       # -> 2x2 Jacobian

# Named axes
mu = tax.variable(398600.4418, order=4, name="mu")
xs = tax.variables([1.0, 0.0, 0.0, 1.0], order=4, name="x")
dx = -mu * xs[0]                   # composes into the union of axes {mu, x}
dx.jacobian("x")                   # derivatives w.r.t. the "x" axis
```

## Eager vs. `@tax.jit`

Eager mode runs each operation as its own cached kernel — ergonomic for
exploration. `@tax.jit` traces a whole function and fuses it into **one** kernel
(no per-op FFI, no intermediate buffers):

```python
@tax.jit
def rhs(t, x, mu):
    r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
    return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])

dx = rhs(0.0, X, 398600.4418)
```

Fusion is a real speedup — on the reference machine, the planar two-body RHS runs
at ≈106 µs/call eager, ≈30 µs/call under `@tax.jit`, vs ≈13 µs/call for a
hand-written C++ kernel. An explicit numba-style signature compiles at decoration:

```python
@tax.jit([tax.f64, tax.ArrayType(order=4, size=4, name="x"),
          tax.ExpansionType(order=4, name="mu")])
def rhs(t, x, mu): ...
```

See `examples/two_body.py` for the full worked example, and `bench/bench_two_body.py`
for the benchmark.

## Known limitations

- `@tax.jit` promotes its output to the union of *all* input schemes; a function
  that ignores an input still carries that input's axes (always-zero) in the
  result. For maps where every input interacts (e.g. an ODE RHS) this is exactly
  the natural scheme.
- Scalar arguments to `@tax.jit` are baked as compile-time constants (a different
  value re-traces). NumPy scalars (`np.float64`) work; pass Python `int`/`float`.
- `x ** p` with a real `p` requires a positive constant term; integer `p` works
  for any base.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_readme.py -v`
Expected: PASS. Full suite: `cd python && .venv/bin/python -m pytest -q` — all green.

- [ ] **Step 5: Commit**

```bash
git add python/README.md python/tests/test_readme.py
git commit -m "docs(py): README — install, quickstart, eager-vs-jit, perf, limitations"
```

---

## Self-Review

**Spec coverage (M6 = packaging, per the M5 roadmap):**
- Precompiled-header warm builds (cut first-touch compile) → Task 1. ✓
- Vendored `tax` headers in the wheel (works without `TAX_INCLUDE`) → Task 2. ✓
- Self-contained compile guarantee (the installed-wheel scenario) → Task 3. ✓
- Compiler/Eigen discovery + usage docs → Task 4 (README). ✓
- `pyproject.toml` finalization (build hook + metadata) → Task 2. ✓
- Pure-Python wheel preserved (no compiled extension on the call path) → all tasks (PCH/vendoring/docs add no extension; the hatch hook only copies headers). ✓
- Out of scope (documented): cffi *API-mode* FFI (breaks the pure-Python wheel; the M5 benchmark showed the residual jit↔C++ gap is Python-level — deferred); vendoring Eigen for a fully offline wheel; `batch=K`.

**Placeholder scan:** No "TBD"/"similar to Task N"/"add error handling". Every code step is complete; the PCH facts (matching `opt_flags`, `#pragma once` lets the kernel keep its include so `emit` is unchanged, key unchanged, graceful `None` fallback, `TAX_USE_PCH` opt-out) are verified on the reference machine; the vendoring path resolution (`parents[3]/include/tax`) and `find_tax_include`'s existing vendored fallback are confirmed.

**Type consistency:** `pch_path(cxx, includes, opt_flags) -> Path|None` and the `-include-pch` addition in `compile_kernel` use the existing `cache_dir`/`cache_key`/`compiler_id`/`flags_for_key`/`STD_FLAG`; `sync_from_repo() -> Path` and `find_tax_include`'s vendored fallback (`tax/_vendor/include`) match; the README test references the public symbols (`tax.variable`/`variables`/`jit`/`concatenate`/`jacobian`/`f64`/`ArrayType`/`ExpansionType`) defined in M1–M4. The `find_tax_include.cache_clear()` calls account for its M1 `@lru_cache`.

---

## Roadmap (after M6 — the layer is complete)

- **Optional — cffi FFI:** if per-call overhead matters more than a pure-Python wheel, an API-mode cffi shim (or an optimized ABI-mode path) can shave the ctypes dispatch cost. Tradeoff: a per-platform compiled wheel instead of `py3-none-any`.
- **Optional — vendor Eigen:** copy the needed Eigen subset into `_vendor` for a fully offline wheel (no system Eigen needed). Heavier wheel; document the MPL2 license.
- **Optional — `batch=K`:** lock-step multi-point evaluation (parameterize the emitted scalar type to `Batch<double,K>`).
- **Optional — bundled compiler / cibuildwheel:** publish wheels per platform; investigate shipping a minimal compiler for pip-install-anywhere.
```
