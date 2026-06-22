# Python JIT Layer — Foundation (M0 + M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the end-to-end JIT pipeline and a working eager **univariate** scalar expansion in Python — `tax.variable(...)` + the math surface compute correct Taylor coefficients by generating C++, compiling it once, caching the `.so`, and calling it via `ctypes`.

**Architecture:** A pure-Python package emits a small C++ translation unit that reuses the existing `tax` header kernels for a *fixed* scheme, compiles it at `-O3` with the system C++ compiler, caches the shared object on disk keyed by a graph+scheme+toolchain hash, and calls the `extern "C"` kernel with numpy coefficient buffers. Eager mode traces a single operation into a one-node graph; this same machinery scales to whole-function fusion later.

**Tech Stack:** Python ≥3.10 (stdlib + numpy + pytest), the system C++23 compiler invoked via `subprocess`, `ctypes` for loading, the existing header-only `tax` library + Eigen for the generated kernels.

## Global Constraints

- **C++ standard:** generated TUs compile with `-std=c++23` (verbatim). The library requires C++23.
- **Static storage preserved:** every generated kernel instantiates a *compile-time-fixed* `tax::IsotropicScheme<N, M>`, so all math runs in `std::array`-backed `TaylorExpansion`. No dynamic-order C++ type is ever created.
- **Pure-Python base package:** no compiled extension module on the import/call path. The only native code is the JIT'd kernel, loaded via `ctypes`.
- **Eigen is mandatory to compile any TU:** `tax/core/taylor_expansion.hpp` includes `tax/la/types.hpp`, so the Eigen include path must be discovered and passed to every compile.
- **Reuse the umbrella header:** generated TUs `#include <tax/tax.hpp>`.
- **Graded-lex coefficient layout is canonical:** index 0 is the value; for univariate, flat index `k` is the coefficient of `dx^k`.
- **Cache key composition (verbatim):** `sha256(ABI_VERSION ‖ tax_lib_version ‖ compiler_id ‖ flags ‖ scalar ‖ graph_canonical)` where `flags` excludes machine-specific `-I` paths.
- **Atomic cache writes:** compile to a temp path, then `os.replace` into the cache (race-safe; duplicate concurrent builds are harmless).
- **Scope of this plan:** **univariate** isotropic eager only. Multivariate coordinate variables, the graded-lex multi-index `flat_index`, the `Array` vector type, `norm`/`cross`/`dot`, named schemes, `tax.jit` fusion, signatures, and packaging are later plans (see Roadmap).

---

## File Structure

```
python/
├── pyproject.toml                     # pure-Python package (hatchling), pytest config
├── tax/
│   ├── __init__.py                    # public API: variable, Expansion, math fns, errors
│   ├── _errors.py                     # CompilerNotFound, EigenNotFound, TaxIncludeNotFound, JitCompileError, DomainError
│   ├── _frontend/
│   │   ├── __init__.py
│   │   ├── scheme.py                  # Isotropic descriptor, num_monomials
│   │   ├── ir.py                      # Var/Const/Op/Graph + canonical()
│   │   ├── types.py                   # Expansion handle + operators
│   │   ├── factories.py              # variable(x0, order)
│   │   ├── mathfns.py                 # sin/cos/exp/... free functions (table-driven)
│   │   └── eager.py                   # eager engine: run(opcode, operands) -> Expansion
│   └── _codegen/
│       ├── __init__.py
│       ├── emit_cpp.py                # emit(graph) -> C++ source string
│       ├── build.py                   # toolchain discovery, compile, cache key/dir
│       └── load.py                    # ctypes load + call
└── tests/
    ├── conftest.py                    # sets TAX_INCLUDE to repo include/, skips if no toolchain
    ├── test_scheme.py
    ├── test_build.py
    ├── test_load.py
    ├── test_spike_m0.py
    ├── test_ir.py
    ├── test_emit.py
    ├── test_expansion.py
    ├── test_factories.py
    └── test_eager.py
```

Run tests from `python/`: `cd python && python -m pytest -v`.

---

### Task 1: Package scaffold + errors

**Files:**
- Create: `python/pyproject.toml`
- Create: `python/tax/__init__.py`
- Create: `python/tax/_errors.py`
- Create: `python/tax/_frontend/__init__.py`
- Create: `python/tax/_codegen/__init__.py`
- Test: `python/tests/test_import.py`

**Interfaces:**
- Produces: package `tax`; exceptions `CompilerNotFound`, `EigenNotFound`, `TaxIncludeNotFound`, `JitCompileError`, `DomainError` (all subclasses of `TaxError`).

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_import.py
def test_package_imports_and_exposes_errors():
    import tax
    from tax import TaxError, CompilerNotFound, JitCompileError, DomainError
    assert issubclass(CompilerNotFound, TaxError)
    assert issubclass(JitCompileError, TaxError)
    assert issubclass(DomainError, TaxError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tax'`.

- [ ] **Step 3: Write the package files**

```toml
# python/pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tax"
version = "0.1.0"
description = "JIT-compiled Python layer for tax Taylor expansions"
requires-python = ">=3.10"
dependencies = ["numpy>=1.23"]

[project.optional-dependencies]
test = ["pytest>=7"]

[tool.hatch.build.targets.wheel]
packages = ["tax"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

```python
# python/tax/_errors.py
class TaxError(Exception):
    """Base class for all tax Python-layer errors."""

class CompilerNotFound(TaxError):
    """No C++ compiler could be discovered."""

class EigenNotFound(TaxError):
    """Eigen headers could not be located."""

class TaxIncludeNotFound(TaxError):
    """The tax header include directory could not be located."""

class JitCompileError(TaxError):
    """The generated translation unit failed to compile."""
    def __init__(self, cmd, stderr, source):
        self.cmd = cmd
        self.stderr = stderr
        self.source = source
        super().__init__(f"JIT compile failed:\n{' '.join(cmd)}\n\n{stderr}")

class DomainError(TaxError):
    """A kernel trapped a domain error at runtime (nonzero return code)."""
```

```python
# python/tax/_frontend/__init__.py
```
```python
# python/tax/_codegen/__init__.py
```
```python
# python/tax/__init__.py
from ._errors import (
    TaxError, CompilerNotFound, EigenNotFound, TaxIncludeNotFound,
    JitCompileError, DomainError,
)

__all__ = [
    "TaxError", "CompilerNotFound", "EigenNotFound", "TaxIncludeNotFound",
    "JitCompileError", "DomainError",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_import.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/pyproject.toml python/tax python/tests/test_import.py
git commit -m "feat(py): package scaffold and error hierarchy"
```

---

### Task 2: Isotropic scheme descriptor

**Files:**
- Create: `python/tax/_frontend/scheme.py`
- Test: `python/tests/test_scheme.py`

**Interfaces:**
- Produces:
  - `num_monomials(order: int, vars: int) -> int`
  - `Isotropic(order: int, vars: int)` (frozen dataclass) with `.n_coeff: int`, `.cpp_type_string() -> str`, `.descriptor_hash() -> str`, `.union(other: Isotropic) -> Isotropic`.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_scheme.py
import pytest
from tax._frontend.scheme import Isotropic, num_monomials

def test_num_monomials():
    assert num_monomials(5, 1) == 6          # univariate order 5 -> 6 coeffs
    assert num_monomials(4, 4) == 70

def test_isotropic_properties():
    s = Isotropic(5, 1)
    assert s.n_coeff == 6
    assert s.cpp_type_string() == "tax::IsotropicScheme<5, 1>"
    assert s.descriptor_hash() == "iso:5:1"

def test_isotropic_validation():
    with pytest.raises(ValueError):
        Isotropic(-1, 1)
    with pytest.raises(ValueError):
        Isotropic(3, 0)

def test_isotropic_union_promotes_order():
    assert Isotropic(3, 1).union(Isotropic(5, 1)) == Isotropic(5, 1)
    with pytest.raises(ValueError):
        Isotropic(3, 1).union(Isotropic(3, 2))   # differing vars
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_scheme.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tax._frontend.scheme'`.

- [ ] **Step 3: Write the implementation**

```python
# python/tax/_frontend/scheme.py
from __future__ import annotations
from dataclasses import dataclass
from math import comb

def num_monomials(order: int, vars: int) -> int:
    return comb(order + vars, vars)

@dataclass(frozen=True)
class Isotropic:
    order: int
    vars: int

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("Isotropic.order must be >= 0")
        if self.vars < 1:
            raise ValueError("Isotropic.vars must be >= 1")

    @property
    def n_coeff(self) -> int:
        return num_monomials(self.order, self.vars)

    def cpp_type_string(self) -> str:
        return f"tax::IsotropicScheme<{self.order}, {self.vars}>"

    def descriptor_hash(self) -> str:
        return f"iso:{self.order}:{self.vars}"

    def union(self, other: "Isotropic") -> "Isotropic":
        if self.vars != other.vars:
            raise ValueError(
                f"isotropic union requires equal vars ({self.vars} != {other.vars})"
            )
        return Isotropic(max(self.order, other.order), self.vars)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_scheme.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/scheme.py python/tests/test_scheme.py
git commit -m "feat(py): Isotropic scheme descriptor"
```

---

### Task 3: Toolchain discovery (compiler, Eigen, tax include)

**Files:**
- Create: `python/tax/_codegen/build.py`
- Create: `python/tests/_helpers.py`
- Test: `python/tests/test_build.py`

**Interfaces:**
- Produces (in `build.py`):
  - `find_compiler() -> str` (path; honors `TAX_CXX`, then `CXX`, then `c++`/`clang++`/`g++`; raises `CompilerNotFound`)
  - `compiler_id(cxx: str) -> str` (path + first `--version` line)
  - `find_eigen_include() -> str` (honors `TAX_EIGEN_INCLUDE`, then `pkg-config eigen3`, then common dirs; raises `EigenNotFound`)
  - `find_tax_include() -> str` (honors `TAX_INCLUDE`, then a vendored `_vendor/include`; raises `TaxIncludeNotFound`)
  - `include_dirs() -> list[str]` = `[find_tax_include(), find_eigen_include()]`
- Consumes: `tax._errors`.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/_helpers.py
import os, pathlib, pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# Set at IMPORT time (before any skipif marker below is evaluated) so the
# codegen finds the repo headers and writes the JIT cache to a gitignored
# scratch dir. A session fixture would run too late — markers evaluate at
# collection/import.
os.environ.setdefault("TAX_INCLUDE", str(REPO_ROOT / "include"))
os.environ.setdefault("TAX_CACHE_DIR", str(REPO_ROOT / "python" / ".tax_cache"))

def _have_toolchain() -> bool:
    from tax._codegen import build
    try:
        build.find_compiler(); build.find_eigen_include(); build.find_tax_include()
        return True
    except Exception:
        return False

needs_toolchain = pytest.mark.skipif(
    not _have_toolchain(), reason="C++ compiler / Eigen / tax headers not available"
)
```

```python
# python/tests/test_build.py
from tax._codegen import build
from tests._helpers import needs_toolchain   # importing also sets TAX_INCLUDE/TAX_CACHE_DIR

@needs_toolchain
def test_find_compiler_returns_path():
    cxx = build.find_compiler()
    assert isinstance(cxx, str) and cxx

@needs_toolchain
def test_compiler_id_is_stable_and_nonempty():
    cxx = build.find_compiler()
    cid = build.compiler_id(cxx)
    assert cxx in cid and len(cid) > len(cxx)

@needs_toolchain
def test_include_dirs_exist():
    import os
    for d in build.include_dirs():
        assert os.path.isdir(d), d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_build.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tax._codegen.build'`.

- [ ] **Step 3: Write the implementation**

```python
# python/tax/_codegen/build.py
from __future__ import annotations
import functools, os, pathlib, shutil, subprocess
from .._errors import CompilerNotFound, EigenNotFound, TaxIncludeNotFound

@functools.lru_cache(maxsize=None)
def find_compiler() -> str:
    for env in ("TAX_CXX", "CXX"):
        c = os.environ.get(env)
        if c:
            return c
    for name in ("c++", "clang++", "g++"):
        p = shutil.which(name)
        if p:
            return p
    raise CompilerNotFound("set TAX_CXX or put c++/clang++/g++ on PATH")

@functools.lru_cache(maxsize=None)
def compiler_id(cxx: str) -> str:
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True)
        first = out.stdout.splitlines()[0] if out.stdout else ""
    except OSError:
        first = ""
    return f"{cxx}|{first}"

@functools.lru_cache(maxsize=None)
def find_eigen_include() -> str:
    e = os.environ.get("TAX_EIGEN_INCLUDE")
    if e:
        return e
    pkg = shutil.which("pkg-config")
    if pkg:
        out = subprocess.run([pkg, "--cflags", "eigen3"], capture_output=True, text=True)
        if out.returncode == 0:
            for tok in out.stdout.split():
                if tok.startswith("-I"):
                    return tok[2:]
    for p in ("/usr/include/eigen3", "/usr/local/include/eigen3",
              "/opt/homebrew/include/eigen3"):
        if os.path.isdir(p):
            return p
    raise EigenNotFound("set TAX_EIGEN_INCLUDE or install eigen3")

@functools.lru_cache(maxsize=None)
def find_tax_include() -> str:
    t = os.environ.get("TAX_INCLUDE")
    if t:
        return t
    vendored = pathlib.Path(__file__).resolve().parents[1] / "_vendor" / "include"
    if vendored.is_dir():
        return str(vendored)
    raise TaxIncludeNotFound("set TAX_INCLUDE to the tax header directory")

def include_dirs() -> list[str]:
    return [find_tax_include(), find_eigen_include()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_build.py -v`
Expected: PASS (on a machine with a compiler + Eigen; otherwise these specific tests fail, signalling the dev environment needs the toolchain).

- [ ] **Step 5: Commit**

```bash
git add python/tax/_codegen/build.py python/tests/_helpers.py python/tests/test_build.py
git commit -m "feat(py): toolchain discovery (compiler, Eigen, tax headers)"
```

---

### Task 4: Compile + on-disk cache

**Files:**
- Modify: `python/tax/_codegen/build.py`
- Test: `python/tests/test_build.py`

**Interfaces:**
- Produces (in `build.py`):
  - `ABI_VERSION: str = "0"`, `TAX_LIB_VERSION: str = "0.1.0"`
  - `cache_dir() -> pathlib.Path` (honors `TAX_CACHE_DIR`, else `~/.cache/tax`)
  - `cache_key(canonical: str, *, cid: str, flags: str, scalar: str = "float64") -> str`
  - `compile_kernel(source: str, key: str, *, cxx: str, includes: list[str], opt_flags: list[str]) -> pathlib.Path`
- Consumes: `JitCompileError`.

- [ ] **Step 1: Write the failing test**

```python
# add to python/tests/test_build.py  (build + needs_toolchain already imported at the top from Task 3)

def test_cache_key_is_deterministic_and_sensitive():
    k1 = build.cache_key("g", cid="c", flags="-O3")
    k2 = build.cache_key("g", cid="c", flags="-O3")
    k3 = build.cache_key("h", cid="c", flags="-O3")
    assert k1 == k2 and k1 != k3
    assert len(k1) == 64   # sha256 hex digest

@needs_toolchain
def test_compile_kernel_builds_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    src = (
        'extern "C" int tax_kernel(const double* const* ins, double* const* outs)'
        ' noexcept { outs[0][0] = ins[0][0] + 1.0; return 0; }\n'
    )
    cxx = build.find_compiler()
    so = build.compile_kernel(src, "abc123", cxx=cxx, includes=build.include_dirs(),
                              opt_flags=["-O3"])
    assert so.exists() and so.suffix == ".so"
    # Second call is a cache hit (same path, no recompile needed).
    so2 = build.compile_kernel(src, "abc123", cxx=cxx, includes=build.include_dirs(),
                               opt_flags=["-O3"])
    assert so2 == so
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_build.py -k "cache or compile" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'cache_key'`.

- [ ] **Step 3: Write the implementation**

```python
# append to python/tax/_codegen/build.py
import hashlib, tempfile
from .._errors import JitCompileError

ABI_VERSION = "0"
TAX_LIB_VERSION = "0.1.0"

def cache_dir() -> pathlib.Path:
    d = os.environ.get("TAX_CACHE_DIR")
    base = pathlib.Path(d) if d else pathlib.Path.home() / ".cache" / "tax"
    return base

def cache_key(canonical: str, *, cid: str, flags: str, scalar: str = "float64") -> str:
    blob = "\x1f".join([ABI_VERSION, TAX_LIB_VERSION, cid, flags, scalar, canonical])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def compile_kernel(source: str, key: str, *, cxx: str, includes: list[str],
                   opt_flags: list[str]) -> pathlib.Path:
    out_dir = cache_dir()
    so_path = out_dir / f"{key}.so"
    if so_path.exists():
        return so_path
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        cpp = pathlib.Path(td) / "kernel.cpp"
        cpp.write_text(source)
        tmp_so = pathlib.Path(td) / "kernel.so"
        cmd = [cxx, "-std=c++23", *opt_flags, "-shared", "-fPIC",
               *[f"-I{i}" for i in includes], str(cpp), "-o", str(tmp_so)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise JitCompileError(cmd, proc.stderr, source)
        os.replace(tmp_so, so_path)   # atomic publish into the cache
    return so_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_build.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_codegen/build.py python/tests/test_build.py
git commit -m "feat(py): compile generated TU and cache .so atomically"
```

---

### Task 5: Kernel loader (ctypes)

**Files:**
- Create: `python/tax/_codegen/load.py`
- Test: `python/tests/test_load.py`

**Interfaces:**
- Produces:
  - `load_kernel(so_path) -> ctypes function` (sets `argtypes`/`restype`)
  - `call_kernel(fn, in_buffers: list[np.ndarray], out_sizes: list[int]) -> list[np.ndarray]` (raises `DomainError` on nonzero return)
- Consumes: `DomainError`.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_load.py
import numpy as np
from tax._codegen import build, load
from tests._helpers import needs_toolchain   # noqa

@needs_toolchain
def test_load_and_call_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    src = (
        'extern "C" int tax_kernel(const double* const* ins, double* const* outs)'
        ' noexcept { outs[0][0] = ins[0][0] * 2.0; outs[0][1] = ins[0][1] + 5.0;'
        ' return 0; }\n'
    )
    cxx = build.find_compiler()
    so = build.compile_kernel(src, "load_test", cxx=cxx,
                              includes=build.include_dirs(), opt_flags=["-O3"])
    fn = load.load_kernel(so)
    outs = load.call_kernel(fn, [np.array([3.0, 7.0])], [2])
    assert outs[0][0] == 6.0 and outs[0][1] == 12.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_load.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tax._codegen.load'`.

- [ ] **Step 3: Write the implementation**

```python
# python/tax/_codegen/load.py
from __future__ import annotations
import ctypes
import numpy as np
from .._errors import DomainError

_DBL_PP = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))

def load_kernel(so_path):
    lib = ctypes.CDLL(str(so_path))
    fn = lib.tax_kernel
    fn.argtypes = [_DBL_PP, _DBL_PP]
    fn.restype = ctypes.c_int
    return fn

def _as_pointer_array(buffers):
    arr = (ctypes.POINTER(ctypes.c_double) * len(buffers))()
    for i, b in enumerate(buffers):
        arr[i] = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return arr

def call_kernel(fn, in_buffers, out_sizes):
    ins = [np.ascontiguousarray(b, dtype=np.float64) for b in in_buffers]
    outs = [np.zeros(n, dtype=np.float64) for n in out_sizes]
    rc = fn(_as_pointer_array(ins), _as_pointer_array(outs))
    if rc != 0:
        raise DomainError(f"kernel returned {rc}")
    return outs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_load.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_codegen/load.py python/tests/test_load.py
git commit -m "feat(py): ctypes kernel loader and caller"
```

---

### Task 6: M0 spike — hand-written `sin(x)*exp(x)` end-to-end

**Files:**
- Test: `python/tests/test_spike_m0.py`

**Interfaces:**
- Consumes: `build.{find_compiler,include_dirs,compile_kernel}`, `load.{load_kernel,call_kernel}`.
- Produces: nothing new — this is the de-risking integration test that proves the architecture (real headers, `-O3`, `ctypes`, correct numerics) before building the frontend.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_spike_m0.py
import numpy as np
from tax._codegen import build, load
from tests._helpers import needs_toolchain   # noqa

SRC = r'''
#include <tax/tax.hpp>
#include <algorithm>
using namespace tax;
extern "C" int tax_kernel(const double* const* ins, double* const* outs) noexcept {
    using E = TaylorExpansion<double, IsotropicScheme<5, 1>>;
    E::Data d; std::copy_n(ins[0], 6, d.data());
    E x{d};
    E r = sin(x) * exp(x);
    std::copy_n(r.coefficients().data(), E::nCoefficients, outs[0]);
    return 0;
}
'''

@needs_toolchain
def test_m0_sin_times_exp(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    cxx = build.find_compiler()
    so = build.compile_kernel(SRC, "m0_spike", cxx=cxx,
                              includes=build.include_dirs(), opt_flags=["-O3"])
    fn = load.load_kernel(so)
    # x = 0 + 1*dx, order 5  -> seed [0, 1, 0, 0, 0, 0]
    (out,) = load.call_kernel(fn, [np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])], [6])
    expected = np.array([0.0, 1.0, 1.0, 1.0/3.0, 0.0, -1.0/30.0])
    np.testing.assert_allclose(out, expected, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_spike_m0.py -v`
Expected: FAIL initially only if a prior task is incomplete; otherwise this should PASS immediately and prove the pipeline. If it fails on compile, read the `JitCompileError` stderr (most likely a missing include path — fix discovery in Task 3).

- [ ] **Step 3: (No new implementation)**

The spike uses only Tasks 3–5. If it passes, the core architectural assumption (reuse headers + `-O3` + ctypes → correct coefficients) holds. If the compile is slow (> ~3 s), note it — a precompiled header is a later optimization (M6), not a blocker.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_spike_m0.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_spike_m0.py
git commit -m "test(py): M0 spike — sin*exp end-to-end via generated TU"
```

---

### Task 7: Graph IR + canonical form

**Files:**
- Create: `python/tax/_frontend/ir.py`
- Test: `python/tests/test_ir.py`

**Interfaces:**
- Produces:
  - `Var(slot: int, scheme)`, `Const(value: float, scheme)`, `Op(opcode: str, operands: tuple[int, ...], scheme)` — frozen dataclasses.
  - `Graph(nodes: list, outputs: list[int], n_inputs: int)` with `.canonical() -> str`.
  - `single_op_graph(opcode: str, operand_schemes: list, result_scheme) -> Graph` — builds N input `Var`s feeding one `Op`, output = the op.
- Consumes: scheme descriptors (only `.descriptor_hash()`).

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_ir.py
from tax._frontend.ir import Var, Const, Op, Graph, single_op_graph
from tax._frontend.scheme import Isotropic

def test_single_op_graph_shape():
    s = Isotropic(5, 1)
    g = single_op_graph("mul", [s, s], s)
    assert g.n_inputs == 2
    assert isinstance(g.nodes[0], Var) and isinstance(g.nodes[1], Var)
    assert isinstance(g.nodes[2], Op) and g.nodes[2].operands == (0, 1)
    assert g.outputs == [2]

def test_canonical_is_structural():
    s = Isotropic(5, 1)
    a = single_op_graph("sin", [s], s).canonical()
    b = single_op_graph("sin", [s], s).canonical()
    c = single_op_graph("cos", [s], s).canonical()
    assert a == b and a != c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_ir.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tax._frontend.ir'`.

- [ ] **Step 3: Write the implementation**

```python
# python/tax/_frontend/ir.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Var:
    slot: int
    scheme: object

@dataclass(frozen=True)
class Const:
    value: float
    scheme: object

@dataclass(frozen=True)
class Op:
    opcode: str
    operands: tuple
    scheme: object

@dataclass
class Graph:
    nodes: list
    outputs: list
    n_inputs: int

    def canonical(self) -> str:
        parts = []
        for i, n in enumerate(self.nodes):
            if isinstance(n, Var):
                parts.append(f"{i}=var({n.slot},{n.scheme.descriptor_hash()})")
            elif isinstance(n, Const):
                parts.append(f"{i}=const({float(n.value).hex()},{n.scheme.descriptor_hash()})")
            elif isinstance(n, Op):
                ops = ",".join(str(o) for o in n.operands)
                parts.append(f"{i}=op({n.opcode},{ops},{n.scheme.descriptor_hash()})")
            else:
                raise TypeError(f"unknown node type {type(n)!r}")
        parts.append("out:" + ",".join(str(o) for o in self.outputs))
        return ";".join(parts)

def single_op_graph(opcode: str, operand_schemes: list, result_scheme) -> Graph:
    nodes = [Var(slot=i, scheme=s) for i, s in enumerate(operand_schemes)]
    op_index = len(nodes)
    nodes.append(Op(opcode=opcode, operands=tuple(range(len(operand_schemes))),
                    scheme=result_scheme))
    return Graph(nodes=nodes, outputs=[op_index], n_inputs=len(operand_schemes))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_ir.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/ir.py python/tests/test_ir.py
git commit -m "feat(py): graph IR with canonical form"
```

---

### Task 8: C++ emitter `emit(graph)`

**Files:**
- Create: `python/tax/_codegen/emit_cpp.py`
- Test: `python/tests/test_emit.py`

**Interfaces:**
- Produces:
  - `CPP_EXPR: dict[str, str]` — opcode → C++ expression template using `{0}`, `{1}` for operand C++ variable names. Covers all M1 ops: `add sub mul div neg sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh exp log sqrt cbrt square cube erf reciprocal pow atan2`.
  - `emit(graph) -> str` — a complete `extern "C" int tax_kernel(...)` TU.
- Consumes: `ir.{Var,Const,Op}`.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_emit.py
from tax._frontend.ir import single_op_graph
from tax._frontend.scheme import Isotropic
from tax._codegen.emit_cpp import emit, CPP_EXPR

def test_emit_contains_scheme_and_signature():
    s = Isotropic(5, 1)
    src = emit(single_op_graph("sin", [s], s))
    assert "#include <tax/tax.hpp>" in src
    assert "tax::IsotropicScheme<5, 1>" in src
    assert 'extern "C" int tax_kernel' in src
    assert "sin(n0)" in src
    assert "std::copy_n(n1.coefficients().data()" in src

def test_emit_binary_mul():
    s = Isotropic(5, 1)
    src = emit(single_op_graph("mul", [s, s], s))
    assert "(n0 * n1)" in src

def test_cpp_expr_table_complete():
    for opc in ["add","sub","mul","div","neg","sin","cos","tan","asin","acos",
                "atan","sinh","cosh","tanh","asinh","acosh","atanh","exp","log",
                "sqrt","cbrt","square","cube","erf","reciprocal","pow","atan2"]:
        assert opc in CPP_EXPR
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_emit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tax._codegen.emit_cpp'`.

- [ ] **Step 3: Write the implementation**

```python
# python/tax/_codegen/emit_cpp.py
from __future__ import annotations
from .._frontend.ir import Var, Const, Op

CPP_EXPR = {
    "add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})",
    "div": "({0} / {1})", "neg": "(-{0})",
    "sin": "sin({0})", "cos": "cos({0})", "tan": "tan({0})",
    "asin": "asin({0})", "acos": "acos({0})", "atan": "atan({0})",
    "sinh": "sinh({0})", "cosh": "cosh({0})", "tanh": "tanh({0})",
    "asinh": "asinh({0})", "acosh": "acosh({0})", "atanh": "atanh({0})",
    "exp": "exp({0})", "log": "log({0})", "sqrt": "sqrt({0})", "cbrt": "cbrt({0})",
    "square": "square({0})", "cube": "cube({0})", "erf": "erf({0})",
    "reciprocal": "reciprocal({0})",
    "pow": "pow({0}, {1})", "atan2": "atan2({0}, {1})",
}

def emit(graph) -> str:
    lines = [
        "#include <tax/tax.hpp>",
        "#include <algorithm>",
        "using namespace tax;",
        'extern "C" int tax_kernel(const double* const* ins, '
        "double* const* outs) noexcept {",
    ]
    for i, node in enumerate(graph.nodes):
        cpp_type = node.scheme.cpp_type_string()
        if isinstance(node, Var):
            n = node.scheme.n_coeff
            lines.append(f"    {cpp_type}::Data d{i}; "
                         f"std::copy_n(ins[{node.slot}], {n}, d{i}.data());")
            lines.append(f"    {cpp_type} n{i}{{d{i}}};")
        elif isinstance(node, Const):
            lines.append(f"    auto n{i} = {cpp_type}::constant({node.value!r});")
        elif isinstance(node, Op):
            expr = CPP_EXPR[node.opcode].format(*[f"n{o}" for o in node.operands])
            lines.append(f"    auto n{i} = {expr};")
        else:
            raise TypeError(f"unknown node type {type(node)!r}")
    for j, o in enumerate(graph.outputs):
        n = graph.nodes[o].scheme.n_coeff
        lines.append(f"    std::copy_n(n{o}.coefficients().data(), {n}, outs[{j}]);")
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_emit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_codegen/emit_cpp.py python/tests/test_emit.py
git commit -m "feat(py): C++ emitter for isotropic single-output kernels"
```

---

### Task 9: `Expansion` handle (univariate accessors)

**Files:**
- Create: `python/tax/_frontend/types.py`
- Test: `python/tests/test_expansion.py`

**Interfaces:**
- Produces: `Expansion(coeffs, scheme)` with `.coeffs: np.ndarray`, `.scheme`, `.value() -> float`, `.numpy() -> np.ndarray`, `.coeff(k: int) -> float` (univariate flat index), `.derivative(k: int) -> float` (`k!`-scaled), and arithmetic dunders `__add__/__radd__/__sub__/__rsub__/__mul__/__rmul__/__truediv__/__rtruediv__/__neg__` that delegate to `eager.run` (wired in Task 11).
- Consumes: `scheme` (for `n_coeff`); later `eager.run`.

> This task implements the data handle + accessors and leaves the dunders importing `eager` lazily; `eager` is created in Task 11. The accessor tests below don't exercise the dunders, so they pass before Task 11.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_expansion.py
import math, numpy as np, pytest
from tax._frontend.types import Expansion
from tax._frontend.scheme import Isotropic

def test_value_and_numpy():
    e = Expansion([2.0, 1.0, 0.0], Isotropic(2, 1))
    assert e.value() == 2.0
    assert np.array_equal(e.numpy(), np.array([2.0, 1.0, 0.0]))

def test_coeff_and_derivative_univariate():
    # exp(x) at 0, order 3: coeffs [1, 1, 1/2, 1/6]; derivatives all 1
    e = Expansion([1.0, 1.0, 0.5, 1.0/6.0], Isotropic(3, 1))
    assert e.coeff(2) == 0.5
    assert math.isclose(e.derivative(2), 1.0)   # 2! * 1/2
    assert math.isclose(e.derivative(3), 1.0)   # 3! * 1/6

def test_coeff_out_of_range():
    e = Expansion([1.0, 1.0], Isotropic(1, 1))
    with pytest.raises(IndexError):
        e.coeff(5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_expansion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tax._frontend.types'`.

- [ ] **Step 3: Write the implementation**

```python
# python/tax/_frontend/types.py
from __future__ import annotations
import math
import numpy as np

class Expansion:
    __slots__ = ("coeffs", "scheme")

    def __init__(self, coeffs, scheme):
        self.coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)
        self.scheme = scheme
        if self.coeffs.shape != (scheme.n_coeff,):
            raise ValueError(
                f"coeffs length {self.coeffs.shape} != scheme.n_coeff {scheme.n_coeff}"
            )

    def value(self) -> float:
        return float(self.coeffs[0])

    def numpy(self) -> np.ndarray:
        return self.coeffs.copy()

    def coeff(self, k: int) -> float:
        if self.scheme.vars != 1:
            raise NotImplementedError("multivariate coeff() arrives in M2")
        if not (0 <= k < self.scheme.n_coeff):
            raise IndexError(f"coeff index {k} out of range [0, {self.scheme.n_coeff})")
        return float(self.coeffs[k])

    def derivative(self, k: int) -> float:
        return self.coeff(k) * math.factorial(k)

    # --- arithmetic: delegate to the eager engine (Task 11) ---
    def __add__(self, other):
        from .eager import binary
        return binary("add", self, other)

    def __radd__(self, other):
        from .eager import binary
        return binary("add", other, self)

    def __sub__(self, other):
        from .eager import binary
        return binary("sub", self, other)

    def __rsub__(self, other):
        from .eager import binary
        return binary("sub", other, self)

    def __mul__(self, other):
        from .eager import binary
        return binary("mul", self, other)

    def __rmul__(self, other):
        from .eager import binary
        return binary("mul", other, self)

    def __truediv__(self, other):
        from .eager import binary
        return binary("div", self, other)

    def __rtruediv__(self, other):
        from .eager import binary
        return binary("div", other, self)

    def __neg__(self):
        from .eager import unary
        return unary("neg", self)

    def __repr__(self):
        return f"Expansion(scheme={self.scheme}, value={self.value()})"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_expansion.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/types.py python/tests/test_expansion.py
git commit -m "feat(py): Expansion handle with univariate accessors"
```

---

### Task 10: `variable` factory (univariate)

**Files:**
- Create: `python/tax/_frontend/factories.py`
- Modify: `python/tax/__init__.py`
- Test: `python/tests/test_factories.py`

**Interfaces:**
- Produces: `variable(x0: float, order: int) -> Expansion` — seeds `[x0, 1, 0, ..., 0]` in `Isotropic(order, 1)` (pure Python; no kernel).
- Consumes: `Expansion`, `Isotropic`.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_factories.py
import numpy as np
import tax
from tax._frontend.scheme import Isotropic

def test_variable_seeds_linear_term():
    x = tax.variable(2.5, order=4)
    assert x.scheme == Isotropic(4, 1)
    assert np.array_equal(x.numpy(), np.array([2.5, 1.0, 0.0, 0.0, 0.0]))

def test_variable_order_zero_has_no_linear_slot():
    x = tax.variable(2.5, order=0)
    assert np.array_equal(x.numpy(), np.array([2.5]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_factories.py -v`
Expected: FAIL — `AttributeError: module 'tax' has no attribute 'variable'`.

- [ ] **Step 3: Write the implementation**

```python
# python/tax/_frontend/factories.py
from __future__ import annotations
import numpy as np
from .scheme import Isotropic
from .types import Expansion

def variable(x0: float, order: int) -> Expansion:
    scheme = Isotropic(order, 1)
    coeffs = np.zeros(scheme.n_coeff, dtype=np.float64)
    coeffs[0] = float(x0)
    if order >= 1:
        coeffs[1] = 1.0
    return Expansion(coeffs, scheme)
```

```python
# add to python/tax/__init__.py
from ._frontend.factories import variable
from ._frontend.types import Expansion

__all__ += ["variable", "Expansion"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_factories.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/factories.py python/tax/__init__.py python/tests/test_factories.py
git commit -m "feat(py): univariate variable() factory"
```

---

### Task 11: Eager engine + math surface

**Files:**
- Create: `python/tax/_frontend/eager.py`
- Create: `python/tax/_frontend/mathfns.py`
- Modify: `python/tax/__init__.py`
- Test: `python/tests/test_eager.py`

**Interfaces:**
- Produces (in `eager.py`):
  - `run(graph, in_buffers, out_sizes) -> list[np.ndarray]` — cache-keyed compile+load+call (in-process LRU on the cache key).
  - `unary(opcode, x) -> Expansion`, `binary(opcode, a, b) -> Expansion` — build the one-op graph, embed operands to the result scheme, call `run`, wrap.
  - `_as_expansion(value, ref_scheme) -> Expansion` — promote a Python scalar to a constant `Expansion`.
- Produces (in `mathfns.py`): the unary free functions `sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, log, sqrt, cbrt, square, cube, erf, reciprocal` and binaries `pow, atan2` — each `f(x) -> Expansion`.
- Consumes: `ir.single_op_graph`, `emit_cpp.emit`, `build.*`, `load.*`, `Expansion`, `Isotropic`.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_eager.py
import numpy as np
import tax
from tests._helpers import needs_toolchain   # noqa

@needs_toolchain
def test_eager_sin_univariate():
    x = tax.variable(0.0, order=5)
    f = tax.sin(x)
    expected = np.array([0, 1, 0, -1/6, 0, 1/120], dtype=float)
    np.testing.assert_allclose(f.numpy(), expected, atol=1e-12)

@needs_toolchain
def test_eager_exp_univariate():
    x = tax.variable(0.0, order=5)
    f = tax.exp(x)
    expected = np.array([1, 1, 1/2, 1/6, 1/24, 1/120], dtype=float)
    np.testing.assert_allclose(f.numpy(), expected, atol=1e-12)

@needs_toolchain
def test_eager_scalar_broadcast_mul():
    x = tax.variable(0.0, order=3)
    f = 2.0 * x            # exercises __rmul__ + _as_expansion
    np.testing.assert_allclose(f.numpy(), np.array([0, 2, 0, 0], dtype=float), atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_eager.py -v`
Expected: FAIL — `AttributeError: module 'tax' has no attribute 'sin'`.

- [ ] **Step 3: Write the implementation**

```python
# python/tax/_frontend/eager.py
from __future__ import annotations
import numpy as np
from .ir import single_op_graph
from .scheme import Isotropic
from .types import Expansion
from .._codegen.emit_cpp import emit
from .._codegen import build, load

_KERNEL_CACHE: dict[str, tuple] = {}

def _embed(x: Expansion, target: Isotropic) -> np.ndarray:
    """Embed an isotropic expansion into a (>=) order target via the graded-lex prefix."""
    if x.scheme == target:
        return x.coeffs
    if x.scheme.vars != target.vars or x.scheme.order > target.order:
        raise ValueError(f"cannot embed {x.scheme} into {target}")
    out = np.zeros(target.n_coeff, dtype=np.float64)
    out[: x.scheme.n_coeff] = x.coeffs        # univariate / graded-lex prefix
    return out

def _as_expansion(value, ref_scheme: Isotropic) -> Expansion:
    if isinstance(value, Expansion):
        return value
    coeffs = np.zeros(ref_scheme.n_coeff, dtype=np.float64)
    coeffs[0] = float(value)
    return Expansion(coeffs, ref_scheme)

def run(graph, in_buffers, out_sizes):
    canon = graph.canonical()
    cxx = build.find_compiler()
    cid = build.compiler_id(cxx)
    flags = "-std=c++23 -O3"
    key = build.cache_key(canon, cid=cid, flags=flags)
    cached = _KERNEL_CACHE.get(key)
    if cached is None:
        so = build.compile_kernel(emit(graph), key, cxx=cxx,
                                  includes=build.include_dirs(), opt_flags=["-O3"])
        cached = load.load_kernel(so)
        _KERNEL_CACHE[key] = cached
    return load.call_kernel(cached, in_buffers, out_sizes)

def unary(opcode: str, x) -> Expansion:
    if not isinstance(x, Expansion):
        raise TypeError(f"{opcode}: expected Expansion, got {type(x)!r}")
    result_scheme = x.scheme
    graph = single_op_graph(opcode, [result_scheme], result_scheme)
    (out,) = run(graph, [x.coeffs], [result_scheme.n_coeff])
    return Expansion(out, result_scheme)

def binary(opcode: str, a, b) -> Expansion:
    ref = a.scheme if isinstance(a, Expansion) else b.scheme
    ea, eb = _as_expansion(a, ref), _as_expansion(b, ref)
    result_scheme = ea.scheme.union(eb.scheme)
    graph = single_op_graph(opcode, [result_scheme, result_scheme], result_scheme)
    ba, bb = _embed(ea, result_scheme), _embed(eb, result_scheme)
    (out,) = run(graph, [ba, bb], [result_scheme.n_coeff])
    return Expansion(out, result_scheme)
```

```python
# python/tax/_frontend/mathfns.py
from __future__ import annotations
from .eager import unary, binary

_UNARY = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
          "asinh", "acosh", "atanh", "exp", "log", "sqrt", "cbrt", "square",
          "cube", "erf", "reciprocal"]

def _make_unary(opcode):
    def fn(x):
        return unary(opcode, x)
    fn.__name__ = opcode
    return fn

for _name in _UNARY:
    globals()[_name] = _make_unary(_name)

def pow(x, y):
    return binary("pow", x, y)

def atan2(y, x):
    return binary("atan2", y, x)

__all__ = _UNARY + ["pow", "atan2"]
```

```python
# add to python/tax/__init__.py
from ._frontend import mathfns as _mathfns
for _n in _mathfns.__all__:
    globals()[_n] = getattr(_mathfns, _n)
__all__ += list(_mathfns.__all__)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/test_eager.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/eager.py python/tax/_frontend/mathfns.py python/tax/__init__.py python/tests/test_eager.py
git commit -m "feat(py): eager engine and math surface (univariate isotropic)"
```

---

### Task 12: End-to-end composition + cache-hit

**Files:**
- Test: `python/tests/test_eager.py` (extend)

**Interfaces:**
- Consumes: the full public surface (`tax.variable`, `tax.sin`, `tax.exp`, operators).
- Produces: the M1 acceptance gate — composed eager numerics match the oracle and repeated evaluation hits the in-process kernel cache (no second compile).

- [ ] **Step 1: Write the failing test**

```python
# add to python/tests/test_eager.py
@needs_toolchain
def test_eager_sin_times_exp_composition():
    import numpy as np, tax
    x = tax.variable(0.0, order=5)
    f = tax.sin(x) * tax.exp(x)
    expected = np.array([0, 1, 1, 1/3, 0, -1/30], dtype=float)
    np.testing.assert_allclose(f.numpy(), expected, atol=1e-12)

@needs_toolchain
def test_kernel_cache_avoids_recompile(monkeypatch):
    import tax
    from tax._frontend import eager
    from tax._codegen import build

    calls = {"n": 0}
    real = build.compile_kernel
    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(build, "compile_kernel", counting)

    x = tax.variable(0.0, order=4)
    tax.sin(x)                 # may compile (cold) or hit on-disk cache
    before = calls["n"]
    tax.sin(x)                 # in-process cache -> no compile call at all
    assert calls["n"] == before
```

Both tests require the toolchain (the cold path compiles a kernel); they are marked `@needs_toolchain`, imported at the top of the file from Task 11. The cache assertion holds whether the first call compiles (cold) or hits the on-disk cache: it captures the compile count after the first call and asserts the second call adds none, because the in-process `_KERNEL_CACHE` short-circuits before `compile_kernel`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/test_eager.py -k "composition or cache" -v`
Expected: FAIL on `composition` until `mul` over two univariate kernels works end-to-end; the `cache` test fails if `compile_kernel` is invoked twice.

- [ ] **Step 3: (No new implementation expected)**

If `composition` fails, debug the `mul` path (Tasks 8/11). If `cache` fails with two compiles, confirm `_KERNEL_CACHE` is keyed by `graph.canonical()` and that identical ops produce identical canonical strings (Task 7).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/ -v`
Expected: PASS (whole suite).

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_eager.py
git commit -m "test(py): M1 gate — composed eager numerics + cache hit"
```

---

## Self-Review

**Spec coverage (this plan = M0 + M1 only):**
- Trace→emit-C++→compile→cache→ctypes pipeline → Tasks 3–8, 11. ✓
- Static `std::array` storage via fixed `IsotropicScheme<N,M>` → Tasks 6, 8. ✓
- Pure-Python package, `ctypes` call path → Tasks 1, 5. ✓
- Scheme descriptor / IR / emitter / build-cache / loader / eager engine (the reusable spine) → Tasks 2, 7, 8, 4, 5, 11. ✓
- Cache key composition + atomic publish → Task 4. ✓
- Eigen-required compile + toolchain discovery → Task 3. ✓
- M0 de-risk spike (numerics + fusion note) → Task 6. ✓
- Deferred to later plans (explicitly out of this plan's scope): multivariate + graded-lex `flat_index`, `Array`/vectors, `norm`/`cross`/`dot`, named schemes, `tax.jit` + signatures + options + fusion, two-body targets, regression/perf, packaging/wheel/PCH, cffi/nanobind fast-core. These are the Roadmap below.

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step shows complete code. ✓

**Type consistency:** `Isotropic`, `Graph`, `single_op_graph`, `emit`, `CPP_EXPR`, `compile_kernel`, `cache_key`, `load_kernel`, `call_kernel`, `Expansion`, `variable`, `run`, `unary`, `binary`, `_embed`, `_as_expansion` are used with identical names/signatures across the tasks that produce and consume them. The eager `run(graph, in_buffers, out_sizes)` signature matches its callers in Task 11. ✓

---

## Roadmap (subsequent plans, each its own working/testable slice)

- **M2 — Multivariate + vectors:** graded-lex `flat_index`/`unflat_index` (cross-checked against C++), multivariate coordinate variables, the `Array` type (contiguous `K×nCoeff`), `concatenate`/`stack`/indexing/elementwise, `dot`/`cross`/`norm`, `value`/`eval`/`jacobian`/`hessian`.
- **M3 — Named schemes:** the `Named` scheme descriptor (single global order) → `NamedTaylorExpansion<T,N,Axes…>`, axis-union promotion, `name=` factories.
- **M4 — `tax.jit` fusion:** tracer over the full function, multi-scheme/multi-output `emit`, options (`opt/cache/compiler/scalar/batch/static_argnums/dump`), and explicit numba-style signatures (eager decoration-time compile).
- **M5 — Targets + regression + perf:** both two-body RHS maps (named and unnamed, bare and pinned-signature) as e2e tests; DACE/C++ accuracy regression; eager-vs-jit-vs-C++ benchmarks.
- **M6 — Packaging:** vendored `tax`/Eigen headers in the wheel, `cffi` FFI upgrade, precompiled-header warm builds, compiler-discovery docs, examples.
