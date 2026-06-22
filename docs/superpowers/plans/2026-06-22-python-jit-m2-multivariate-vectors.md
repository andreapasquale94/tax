# Python JIT Layer — M2: Multivariate + Vectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the M1 foundation to **multivariate** isotropic expansions and a first-class **`Array`** vector type — coordinate-variable factories, multi-index accessors (`coeff`/`derivative`/`gradient`/`hessian`/`eval`), and vector ops (`concatenate`/`stack`/`dot`/`cross`/`norm`/elementwise math, `jacobian`/`hessian`) — all eager.

**Architecture:** The M1 eager engine already JIT-compiles kernels for any `IsotropicScheme<N,M>`, so **multivariate arithmetic/math works unchanged** — M2 adds only the Python frontend: a graded-lex `flat_index`/`unflat_index` that exactly mirrors C++ `tax::flatIndex`/`unflatIndex` (cross-checked by a generated probe), multi-index accessors built on it, and an `Array` handle (a contiguous `(K, nCoeff)` buffer over one shared scheme) whose vector ops decompose into per-row M1 eager operations. Vector reductions (`dot`/`norm`/`cross`) are composite — no new kernels.

**Tech Stack:** Python ≥3.10 (stdlib `math` + numpy + pytest), the M1 pipeline (`tax._frontend` / `tax._codegen`), the system C++23 compiler + Eigen + `tax` headers (only for the cross-check probe and the eager numeric tests).

## Global Constraints

- **Builds on M1** (branch `feature/python-jit-expansions`, all M1 tasks merged). Reuse the existing modules; do not duplicate them.
- **Test runner:** `cd /Users/andrea/Documents/Codes/tax/python && .venv/bin/python -m pytest ...` (a venv with pytest+numpy already exists at `python/.venv`). Toolchain-dependent tests import `needs_toolchain` from `tests._helpers` (which sets `TAX_INCLUDE`/`TAX_CACHE_DIR` at import).
- **`flat_index`/`unflat_index` MUST match C++ `tax::flatIndex<M>`/`tax::unflatIndex<M>` exactly** (graded-lex; degree blocks contiguous; within a degree, ordered by first-variable exponent descending). This is verified by a generated probe kernel (Task 1).
- **`Array` is a contiguous `(K, nCoeff)` float64 buffer over a single SHARED scheme.** Indexing returns an `Expansion` (a row) or an `Array` (a slice).
- **Multivariate eager reuses the M1 engine unmodified.** `Isotropic(order, M>1)` flows through `single_op_graph` → `emit` → `compile_kernel` exactly as univariate did. Do not change `eager.py`'s `run`/`unary`/`binary` semantics.
- **Vector reductions are composite:** `dot`/`norm`/`cross` are built from per-row `eager.unary`/`eager.binary` calls (`mul`/`add`/`sub`/`sqrt`) — **no new kernels, no new opcodes**.
- **numpy shape conventions:** `dot`/`norm` → `Expansion`; `cross` → a scalar `Expansion` for 2-vectors, a 3-row `Array` for 3-vectors; `Array.jacobian()` → `(K, M)`; `Array.hessian()` → `(K, M, M)`; `Array.value()` → `(K,)`.
- **`coeff`/`derivative` out-of-box semantics match C++:** a multi-index whose total degree exceeds the scheme order returns `0.0` (the C++ `kNotInBox` behavior), NOT an exception. Wrong arity or a negative exponent raises `ValueError`. (This generalizes — and slightly changes — the M1 univariate `coeff(k)`; Task 2 updates the affected M1 test.)
- **Static storage, pure-Python base package, graded-lex, cache key** — all M1 constraints still hold.

---

## File Structure

```
python/tax/
├── _frontend/
│   ├── scheme.py        # MODIFY: add _binom, flat_index(alpha), unflat_index(k, vars)
│   ├── types.py         # MODIFY: Expansion.coeff/derivative -> multi-index; add gradient/hessian/eval
│   ├── factories.py     # MODIFY: add variables(point, order) -> Array
│   ├── array.py         # CREATE: Array type + concatenate/stack/dot/cross/norm
│   ├── mathfns.py       # MODIFY: dispatch unary/binary over Array (map per row)
│   └── eager.py         # (unchanged; reused as-is)
└── __init__.py          # MODIFY: export Array, variables, concatenate, stack, dot, cross, norm
python/tests/
├── test_layout.py       # CREATE: flat/unflat self-consistency + C++ cross-check
├── test_expansion.py    # MODIFY: multivariate coeff/derivative/gradient/hessian/eval
├── test_array.py        # CREATE: Array core, elementwise, eval/jacobian/hessian
├── test_vector_ops.py   # CREATE: concatenate/stack/dot/cross/norm
├── test_factories.py    # MODIFY: variables(point, order)
└── test_m2_gate.py      # CREATE: vector-map end-to-end gate
```

---

### Task 1: Graded-lex layout (`flat_index`/`unflat_index`) + C++ cross-check

**Files:**
- Modify: `python/tax/_frontend/scheme.py`
- Test: `python/tests/test_layout.py`

**Interfaces:**
- Produces (in `scheme.py`): `_binom(n: int, k: int) -> int`; `flat_index(alpha) -> int` (alpha = sequence of `M` ints); `unflat_index(k: int, vars: int) -> tuple[int, ...]`.
- Consumes: `num_monomials`, `Isotropic` (existing); `build`/`load`/`emit`-free hand-written probe kernel via `tax._codegen.build`/`load`.

- [ ] **Step 1: Write the failing tests**

```python
# python/tests/test_layout.py
import numpy as np
from tax._frontend.scheme import flat_index, unflat_index, num_monomials
from tax._codegen import build, load
from tests._helpers import needs_toolchain   # sets TAX_INCLUDE/TAX_CACHE_DIR

def test_flat_unflat_roundtrip_and_bijection():
    for N in range(0, 6):
        for M in range(1, 5):
            n = num_monomials(N, M)
            seen = set()
            for k in range(n):
                a = unflat_index(k, M)
                assert len(a) == M
                assert sum(a) <= N
                assert flat_index(a) == k          # round-trip
                seen.add(a)
            assert len(seen) == n                  # bijection onto [0, n)

def test_linear_slots_are_i_plus_one():
    # coordinate variable i's linear monomial e_i lands at flat index i+1
    for M in (2, 3, 4):
        for i in range(M):
            e = tuple(1 if j == i else 0 for j in range(M))
            assert flat_index(e) == i + 1

# C++ cross-check: a probe kernel writes encode(unflatIndex<M>(k)) for every k;
# Python must compute the same encoding from its own unflat_index.
_PROBE = r'''
#include <tax/tax.hpp>
extern "C" int tax_kernel(const double* const*, double* const* outs) noexcept {{
    constexpr int N = {N}, M = {M};
    constexpr std::size_t n = tax::numMonomials(N, M);
    for (std::size_t k = 0; k < n; ++k) {{
        auto a = tax::unflatIndex<M>(k);
        double e = 0.0, base = 1.0;
        for (int i = 0; i < M; ++i) {{ e += double(a[std::size_t(i)]) * base; base *= double(N + 1); }}
        outs[0][k] = e;
    }}
    return 0;
}}
'''

def _encode(alpha, N):
    e, base = 0.0, 1.0
    for a in alpha:
        e += a * base
        base *= (N + 1)
    return e

@needs_toolchain
@pytest.mark.parametrize("N,M", [(5, 1), (3, 2), (4, 3)])
def test_layout_matches_cpp(N, M, tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    src = _PROBE.format(N=N, M=M)
    so = build.compile_kernel(src, f"layout_probe_{N}_{M}", cxx=build.find_compiler(),
                              includes=build.include_dirs(), opt_flags=["-O3"])
    n = num_monomials(N, M)
    (out,) = load.call_kernel(load.load_kernel(so), [], [n])
    expected = np.array([_encode(unflat_index(k, M), N) for k in range(n)])
    np.testing.assert_array_equal(out, expected)
```

Add `import pytest` at the top of the file (the parametrize/needs_toolchain decorators need it).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_layout.py -v`
Expected: FAIL — `ImportError: cannot import name 'flat_index' from 'tax._frontend.scheme'`.

- [ ] **Step 3: Add the layout functions to `scheme.py`**

```python
# add to python/tax/_frontend/scheme.py
import math

def _binom(n: int, k: int) -> int:
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)

def flat_index(alpha) -> int:
    """Graded-lex flat index of multi-index `alpha` (mirrors C++ tax::flatIndex)."""
    alpha = tuple(alpha)
    M = len(alpha)
    d = sum(alpha)
    idx = _binom(d + M - 1, M)
    rem = d
    for i in range(M - 1):
        idx += _binom(rem - alpha[i] + (M - 2 - i), M - 1 - i)
        rem -= alpha[i]
    return idx

def unflat_index(k: int, vars: int) -> tuple:
    """Inverse of flat_index for `vars` variables (mirrors C++ tax::unflatIndex)."""
    M = vars
    alpha = [0] * M
    d = 0
    while _binom(d + M, M) <= k:
        d += 1
    rank = k - _binom(d + M - 1, M)
    rem = d
    for i in range(M - 1):
        vars_left = M - i
        for ai in range(rem, -1, -1):
            block = _binom(rem - ai + vars_left - 2, vars_left - 2)
            if rank < block:
                alpha[i] = ai
                rem -= ai
                break
            rank -= block
    alpha[M - 1] = rem
    return tuple(alpha)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_layout.py -v`
Expected: PASS (incl. the 3 parametrized C++ cross-check cases, which RUN on this machine).
Then full suite: `cd python && .venv/bin/python -m pytest -q` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/scheme.py python/tests/test_layout.py
git commit -m "feat(py): graded-lex flat_index/unflat_index + C++ layout cross-check"
```

---

### Task 2: Expansion multivariate accessors

**Files:**
- Modify: `python/tax/_frontend/types.py`
- Modify: `python/tests/test_expansion.py`
- Test: `python/tests/test_expansion.py`

**Interfaces:**
- Produces (on `Expansion`): `coeff(*alpha) -> float` (multi-index; out-of-box → 0.0; wrong arity / negative → ValueError); `derivative(*alpha) -> float` (`coeff * Π aᵢ!`); `gradient() -> np.ndarray` shape `(M,)`; `hessian() -> np.ndarray` shape `(M, M)`; `eval(dx) -> float`.
- Consumes: `scheme.flat_index`, `scheme.unflat_index` (Task 1).

- [ ] **Step 1: Write the failing tests + update the M1 out-of-range test**

```python
# REPLACE the body of test_coeff_out_of_range in python/tests/test_expansion.py with:
def test_coeff_out_of_box_returns_zero_and_validates_arity():
    import pytest
    from tax._frontend.types import Expansion
    from tax._frontend.scheme import Isotropic
    e = Expansion([1.0, 1.0], Isotropic(1, 1))
    assert e.coeff(5) == 0.0            # degree 5 > order 1 -> not in box -> 0 (C++ kNotInBox)
    with pytest.raises(ValueError):
        e.coeff(1, 2)                   # wrong arity (vars == 1)
    with pytest.raises(ValueError):
        e.coeff(-1)                     # negative exponent
```

```python
# ADD to python/tests/test_expansion.py
import numpy as np
from tax._frontend.types import Expansion
from tax._frontend.scheme import Isotropic

def test_multivariate_coeff_and_derivative():
    # f = x0*x1 expanded at (1,2), order 2, M=2:
    #   flat layout [ (0,0),(1,0),(0,1),(2,0),(1,1),(0,2) ] = [2, 2, 1, 0, 1, 0]
    f = Expansion([2.0, 2.0, 1.0, 0.0, 1.0, 0.0], Isotropic(2, 2))
    assert f.coeff(0, 0) == 2.0
    assert f.coeff(1, 0) == 2.0
    assert f.coeff(0, 1) == 1.0
    assert f.coeff(1, 1) == 1.0
    assert f.coeff(2, 0) == 0.0
    assert f.derivative(1, 1) == 1.0    # mixed partial of x0*x1 = 1
    assert f.derivative(2, 0) == 0.0    # coeff(2,0)=0 -> 0

def test_gradient_and_hessian():
    f = Expansion([2.0, 2.0, 1.0, 0.0, 1.0, 0.0], Isotropic(2, 2))   # x0*x1 at (1,2)
    assert np.array_equal(f.gradient(), np.array([2.0, 1.0]))        # [x1, x0] = [2, 1]
    assert np.array_equal(f.hessian(), np.array([[0.0, 1.0], [1.0, 0.0]]))

def test_eval_multivariate():
    f = Expansion([2.0, 2.0, 1.0, 0.0, 1.0, 0.0], Isotropic(2, 2))   # (1+dx0)(2+dx1)
    # exact at order 2: (1.1)*(2.2) = 2.42
    assert abs(f.eval([0.1, 0.2]) - 2.42) < 1e-12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_expansion.py -v`
Expected: FAIL — `gradient`/`hessian`/`eval` missing; `coeff(0,0)` arity errors against the old univariate `coeff(k)`.

- [ ] **Step 3: Generalize the accessors in `types.py`**

Replace the existing `coeff` and `derivative` methods, and add `gradient`/`hessian`/`eval`:

```python
# in python/tax/_frontend/types.py — add near the top:
from .scheme import flat_index, unflat_index

# replace coeff() and derivative() with:
    def coeff(self, *alpha) -> float:
        if len(alpha) != self.scheme.vars:
            raise ValueError(
                f"coeff expects {self.scheme.vars} exponents, got {len(alpha)}"
            )
        if any(a < 0 for a in alpha):
            raise ValueError("coeff: negative exponent")
        k = flat_index(alpha)
        return float(self.coeffs[k]) if k < self.scheme.n_coeff else 0.0

    def derivative(self, *alpha) -> float:
        fac = 1
        for a in alpha:
            fac *= math.factorial(a)
        return self.coeff(*alpha) * fac

# add new methods:
    def gradient(self) -> np.ndarray:
        M = self.scheme.vars
        g = np.empty(M, dtype=np.float64)
        for i in range(M):
            e = tuple(1 if j == i else 0 for j in range(M))
            g[i] = self.coeff(*e)            # ∂f/∂xᵢ = coeff(eᵢ) · 1!
        return g

    def hessian(self) -> np.ndarray:
        M = self.scheme.vars
        H = np.empty((M, M), dtype=np.float64)
        for i in range(M):
            for j in range(M):
                a = [0] * M
                a[i] += 1
                a[j] += 1
                H[i, j] = self.derivative(*a)
        return H

    def eval(self, dx) -> float:
        M = self.scheme.vars
        if len(dx) != M:
            raise ValueError(f"eval expects {M} displacements, got {len(dx)}")
        total = 0.0
        for k in range(self.scheme.n_coeff):
            a = unflat_index(k, M)
            term = float(self.coeffs[k])
            for i in range(M):
                term *= dx[i] ** a[i]
            total += term
        return total
```

(`math` is already imported in `types.py` from M1.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_expansion.py -v`
Expected: PASS. Full suite: `cd python && .venv/bin/python -m pytest -q` — no regressions (the univariate M1 tests still pass: `coeff(2)` for `M=1` is multi-index `(2,)`).

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/types.py python/tests/test_expansion.py
git commit -m "feat(py): multivariate Expansion accessors (coeff/derivative/gradient/hessian/eval)"
```

---

### Task 3: `Array` core

**Files:**
- Create: `python/tax/_frontend/array.py`
- Modify: `python/tax/__init__.py`
- Test: `python/tests/test_array.py`

**Interfaces:**
- Produces: `Array(coeffs, scheme)` where `coeffs` is shape `(K, nCoeff)` float64; `.coeffs`, `.scheme`, `.__len__() -> K`, `.__getitem__(i)` (int → `Expansion`, slice → `Array`), `.value() -> np.ndarray (K,)`, `.numpy() -> np.ndarray (K, nCoeff)` (copy), `._rows() -> list[Expansion]`.
- Consumes: `Expansion` (from `types`).

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_array.py
import numpy as np
from tax._frontend.array import Array
from tax._frontend.types import Expansion
from tax._frontend.scheme import Isotropic

def test_array_construction_and_indexing():
    s = Isotropic(2, 2)
    data = np.array([[1.0, 2.0, 3.0, 0, 0, 0], [4.0, 5.0, 6.0, 0, 0, 0]])
    a = Array(data, s)
    assert len(a) == 2
    assert isinstance(a[0], Expansion)
    assert np.array_equal(a[0].coeffs, data[0])
    assert np.array_equal(a.value(), np.array([1.0, 4.0]))
    sub = a[0:1]
    assert isinstance(sub, Array) and len(sub) == 1

def test_array_numpy_is_copy():
    s = Isotropic(1, 1)
    a = Array(np.array([[1.0, 2.0]]), s)
    out = a.numpy()
    out[0, 0] = 99.0
    assert a.value()[0] == 1.0

def test_array_shape_validation():
    import pytest
    with pytest.raises(ValueError):
        Array(np.zeros((2, 5)), Isotropic(2, 2))   # nCoeff(2,2)=6, not 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_array.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tax._frontend.array'`.

- [ ] **Step 3: Implement `array.py` core + export it**

```python
# python/tax/_frontend/array.py
from __future__ import annotations
import numpy as np
from .types import Expansion

class Array:
    """A vector of K expansions over one shared scheme; contiguous (K, nCoeff) buffer."""
    __slots__ = ("coeffs", "scheme")

    def __init__(self, coeffs, scheme):
        self.coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)
        self.scheme = scheme
        if self.coeffs.ndim != 2 or self.coeffs.shape[1] != scheme.n_coeff:
            raise ValueError(
                f"Array coeffs shape {self.coeffs.shape} != (K, {scheme.n_coeff})"
            )

    def __len__(self) -> int:
        return self.coeffs.shape[0]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Array(self.coeffs[i], self.scheme)
        return Expansion(self.coeffs[i], self.scheme)

    def _rows(self) -> list:
        return [Expansion(self.coeffs[i], self.scheme) for i in range(len(self))]

    def value(self) -> np.ndarray:
        return self.coeffs[:, 0].copy()

    def numpy(self) -> np.ndarray:
        return self.coeffs.copy()

    def __repr__(self):
        return f"Array(K={len(self)}, scheme={self.scheme})"
```

```python
# add to python/tax/__init__.py
from ._frontend.array import Array
__all__ += ["Array"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_array.py -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/array.py python/tax/__init__.py python/tests/test_array.py
git commit -m "feat(py): Array vector handle (construction, indexing, value/numpy)"
```

---

### Task 4: `variables(point, order)` factory + multivariate eager numerics

**Files:**
- Modify: `python/tax/_frontend/factories.py`
- Modify: `python/tax/__init__.py`
- Modify: `python/tests/test_factories.py`
- Test: `python/tests/test_factories.py`

**Interfaces:**
- Produces: `variables(point, order) -> Array` — `M = len(point)` coordinate variables over `Isotropic(order, M)`; row `i` seeds `coeffs[0] = point[i]` and (if `order >= 1`) `coeffs[i+1] = 1.0` (since `flat_index(eᵢ) = i+1`).
- Consumes: `Isotropic`, `Array`, M1's eager engine (for the numeric test).

- [ ] **Step 1: Write the failing tests**

```python
# add to python/tests/test_factories.py
import numpy as np
import tax
from tax._frontend.array import Array
from tax._frontend.scheme import Isotropic
from tests._helpers import needs_toolchain

def test_variables_seeds_coordinate_rows():
    X = tax.variables([1.0, 2.0], order=2)
    assert isinstance(X, Array)
    assert X.scheme == Isotropic(2, 2)
    # row 0: x0 = 1 + dx0  -> [1, 1, 0, 0, 0, 0]; row 1: x1 = 2 + dx1 -> [2, 0, 1, 0, 0, 0]
    assert np.array_equal(X[0].coeffs, np.array([1.0, 1.0, 0, 0, 0, 0]))
    assert np.array_equal(X[1].coeffs, np.array([2.0, 0.0, 1.0, 0, 0, 0]))

@needs_toolchain
def test_multivariate_eager_product():
    X = tax.variables([1.0, 2.0], order=2)
    f = X[0] * X[1]                       # eager mul over IsotropicScheme<2,2>
    # (1+dx0)(2+dx1) = 2 + 2 dx0 + 1 dx1 + dx0 dx1
    np.testing.assert_allclose(
        f.numpy(), np.array([2.0, 2.0, 1.0, 0.0, 1.0, 0.0]), atol=1e-12
    )
    assert np.array_equal(f.gradient(), np.array([2.0, 1.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_factories.py -v`
Expected: FAIL — `AttributeError: module 'tax' has no attribute 'variables'`.

- [ ] **Step 3: Implement `variables` and export it**

```python
# add to python/tax/_frontend/factories.py
import numpy as np
from .array import Array

def variables(point, order) -> Array:
    point = list(point)
    M = len(point)
    if M < 1:
        raise ValueError("variables(): point must have at least one element")
    scheme = Isotropic(order, M)
    data = np.zeros((M, scheme.n_coeff), dtype=np.float64)
    for i in range(M):
        data[i, 0] = float(point[i])
        if order >= 1:
            data[i, i + 1] = 1.0          # flat_index(e_i) == i + 1
    return Array(data, scheme)
```

```python
# update python/tax/__init__.py
from ._frontend.factories import variable, variables
__all__ += ["variables"]
```

(The existing `from ._frontend.factories import variable` line becomes `import variable, variables`; keep `Expansion` import as-is. `__all__` already has `variable`; append `variables`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_factories.py -v`
Expected: PASS (the `@needs_toolchain` product test RUNS and matches the oracle). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/factories.py python/tax/__init__.py python/tests/test_factories.py
git commit -m "feat(py): variables(point, order) coordinate-variable Array + multivariate eager"
```

---

### Task 5: Array elementwise math + arithmetic

**Files:**
- Modify: `python/tax/_frontend/array.py`
- Modify: `python/tax/_frontend/mathfns.py`
- Test: `python/tests/test_array.py`

**Interfaces:**
- Produces (on `Array`): `_map_unary(op) -> Array`; `_map_binary(op, other) -> Array` (other = Array | Expansion | scalar); arithmetic dunders `__add__/__radd__/__sub__/__rsub__/__mul__/__rmul__/__truediv__/__rtruediv__/__neg__`.
- Produces (in `mathfns`): the unary free functions and `pow`/`atan2` now accept an `Array` (map per row).
- Consumes: `eager.unary`/`eager.binary` (M1), `Expansion`.

- [ ] **Step 1: Write the failing tests**

```python
# add to python/tests/test_array.py
import tax
from tests._helpers import needs_toolchain

@needs_toolchain
def test_array_elementwise_math():
    X = tax.variables([0.0, 0.0], order=3)
    S = tax.sin(X)                         # elementwise sin over the 2-vector
    # each row depends only on its own variable: sin(dx_i)
    # row 0 = sin(dx0): coeff(1,0)=1, coeff(3,0)=-1/6
    np.testing.assert_allclose(S[0].coeff(1, 0), 1.0, atol=1e-12)
    np.testing.assert_allclose(S[0].coeff(3, 0), -1.0 / 6.0, atol=1e-12)
    np.testing.assert_allclose(S[1].coeff(0, 1), 1.0, atol=1e-12)

@needs_toolchain
def test_array_arithmetic_and_broadcast():
    X = tax.variables([1.0, 2.0], order=2)
    Y = 2.0 * X + X                        # scalar broadcast + elementwise add -> 3*X
    np.testing.assert_allclose(Y.value(), np.array([3.0, 6.0]), atol=1e-12)
    Z = X + X[0]                            # broadcast an Expansion over the Array
    np.testing.assert_allclose(Z.value(), np.array([2.0, 3.0]), atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_array.py -k "elementwise or arithmetic" -v`
Expected: FAIL — `tax.sin(Array)` errors (mathfns assumes Expansion); `Array` has no `__mul__`.

- [ ] **Step 3: Implement Array mapping + dunders, and Array-aware mathfns**

```python
# add to python/tax/_frontend/array.py (inside class Array)
    def _map_unary(self, op):
        from . import eager
        results = [eager.unary(op, r) for r in self._rows()]
        return Array(np.stack([r.coeffs for r in results]), results[0].scheme)

    def _map_binary(self, op, other):
        from . import eager
        rows = self._rows()
        if isinstance(other, Array):
            if len(other) != len(self):
                raise ValueError(f"Array length mismatch: {len(self)} vs {len(other)}")
            results = [eager.binary(op, a, b) for a, b in zip(rows, other._rows())]
        else:                                  # Expansion or Python scalar -> broadcast
            results = [eager.binary(op, a, other) for a in rows]
        return Array(np.stack([r.coeffs for r in results]), results[0].scheme)

    def __add__(self, other): return self._map_binary("add", other)
    def __radd__(self, other): return self._map_binary("add", other)
    def __sub__(self, other): return self._map_binary("sub", other)
    def __rsub__(self, other):
        from . import eager
        rows = self._rows()
        results = [eager.binary("sub", other, a) for a in rows]
        return Array(np.stack([r.coeffs for r in results]), results[0].scheme)
    def __mul__(self, other): return self._map_binary("mul", other)
    def __rmul__(self, other): return self._map_binary("mul", other)
    def __truediv__(self, other): return self._map_binary("div", other)
    def __rtruediv__(self, other):
        from . import eager
        rows = self._rows()
        results = [eager.binary("div", other, a) for a in rows]
        return Array(np.stack([r.coeffs for r in results]), results[0].scheme)
    def __neg__(self): return self._map_unary("neg")
```

```python
# modify python/tax/_frontend/mathfns.py
# At the top, import Array for dispatch:
from .array import Array
from .eager import unary, binary

def _make_unary(opcode):
    def fn(x):
        if isinstance(x, Array):
            return x._map_unary(opcode)
        return unary(opcode, x)
    fn.__name__ = opcode
    return fn

# (the loop building globals()[_name] = _make_unary(_name) is unchanged)

def pow(x, y):
    if isinstance(x, Array):
        return x._map_binary("pow", y)
    return binary("pow", x, y)

def atan2(y, x):
    if isinstance(y, Array):
        return y._map_binary("atan2", x)
    return binary("atan2", y, x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_array.py -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/array.py python/tax/_frontend/mathfns.py python/tests/test_array.py
git commit -m "feat(py): Array elementwise math + arithmetic (per-row eager dispatch)"
```

---

### Task 6: `concatenate` / `stack`

**Files:**
- Modify: `python/tax/_frontend/array.py`
- Modify: `python/tax/__init__.py`
- Test: `python/tests/test_vector_ops.py`

**Interfaces:**
- Produces: `concatenate(items) -> Array` (items = list of `Expansion` and/or `Array`; flattens `Array`s into rows; embeds all to the union scheme via the graded-lex prefix); `stack = concatenate` (alias).
- Consumes: `eager._embed` (M1 — graded-lex prefix embed), `scheme.union`, `Expansion`.

- [ ] **Step 1: Write the failing tests**

```python
# python/tests/test_vector_ops.py
import numpy as np
import tax
from tax._frontend.array import Array
from tests._helpers import needs_toolchain

@needs_toolchain
def test_concatenate_scalars_into_vector():
    X = tax.variables([1.0, 2.0], order=2)
    Y = tax.concatenate([X[0] * X[1], X[0] + X[1]])
    assert isinstance(Y, Array) and len(Y) == 2
    np.testing.assert_allclose(Y.value(), np.array([2.0, 3.0]), atol=1e-12)

@needs_toolchain
def test_concatenate_flattens_arrays():
    X = tax.variables([1.0, 2.0], order=2)
    Y = tax.concatenate([X, X[0] * X[1]])     # 2 rows from X + 1 row -> length 3
    assert len(Y) == 3
    np.testing.assert_allclose(Y.value(), np.array([1.0, 2.0, 2.0]), atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_vector_ops.py -v`
Expected: FAIL — `AttributeError: module 'tax' has no attribute 'concatenate'`.

- [ ] **Step 3: Implement `concatenate`/`stack` and export them**

```python
# add to python/tax/_frontend/array.py (module-level functions, after the class)
def concatenate(items) -> Array:
    from . import eager
    exps = []
    for it in items:
        if isinstance(it, Array):
            exps.extend(it._rows())
        else:
            exps.append(it)            # Expansion
    if not exps:
        raise ValueError("concatenate(): empty input")
    target = exps[0].scheme
    for e in exps[1:]:
        target = target.union(e.scheme)
    rows = [eager._embed(e, target) for e in exps]
    return Array(np.stack(rows), target)

stack = concatenate
```

```python
# add to python/tax/__init__.py
from ._frontend.array import Array, concatenate, stack
__all__ += ["concatenate", "stack"]
```

(Merge with the existing `from ._frontend.array import Array` line from Task 3 — final import: `from ._frontend.array import Array, concatenate, stack`. `Array` stays in `__all__`; append `concatenate`, `stack`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_vector_ops.py -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/array.py python/tax/__init__.py python/tests/test_vector_ops.py
git commit -m "feat(py): concatenate/stack expansions into an Array (union-scheme embed)"
```

---

### Task 7: `dot` / `norm` / `cross`

**Files:**
- Modify: `python/tax/_frontend/array.py`
- Modify: `python/tax/__init__.py`
- Test: `python/tests/test_vector_ops.py`

**Interfaces:**
- Produces: `dot(a, b) -> Expansion`; `norm(a) -> Expansion` (`sqrt(dot(a, a))`); `cross(a, b)` → `Expansion` for 2-vectors (scalar z-component), `Array` (3 rows) for 3-vectors. All composite (per-row `eager` `mul`/`add`/`sub`/`sqrt`).
- Consumes: `eager.unary`/`eager.binary`, `Array`.

- [ ] **Step 1: Write the failing tests**

```python
# add to python/tests/test_vector_ops.py
@needs_toolchain
def test_dot_and_norm():
    X = tax.variables([3.0, 4.0], order=2)
    d = tax.dot(X, X)                      # x0^2 + x1^2, value 9+16=25
    np.testing.assert_allclose(d.value(), 25.0, atol=1e-12)
    n = tax.norm(X)                        # sqrt(25) = 5
    np.testing.assert_allclose(n.value(), 5.0, atol=1e-12)
    # d/dx0 (x0^2+x1^2) = 2 x0 = 6 ; d/dx1 = 2 x1 = 8
    np.testing.assert_allclose(d.gradient(), np.array([6.0, 8.0]), atol=1e-12)

@needs_toolchain
def test_cross_3d_matches_numpy():
    X = tax.variables([1.0, 2.0, 3.0], order=1)
    C = tax.concatenate([X[0] * 0.0 + 0.0, X[0] * 0.0 + 1.0, X[0] * 0.0 + 0.0])  # const [0,1,0]
    R = tax.cross(X, C)
    # cross([1,2,3],[0,1,0]) = [2*0-3*1, 3*0-1*0, 1*1-2*0] = [-3, 0, 1]
    np.testing.assert_allclose(R.value(), np.array([-3.0, 0.0, 1.0]), atol=1e-12)

@needs_toolchain
def test_cross_2d_is_scalar():
    X = tax.variables([1.0, 2.0], order=1)
    C = tax.concatenate([X[0] * 0.0 + 3.0, X[0] * 0.0 + 4.0])   # const [3,4]
    z = tax.cross(X, C)                    # x0*4 - x1*3, value 1*4 - 2*3 = -2
    np.testing.assert_allclose(z.value(), -2.0, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_vector_ops.py -k "dot or norm or cross" -v`
Expected: FAIL — `AttributeError: module 'tax' has no attribute 'dot'`.

- [ ] **Step 3: Implement `dot`/`norm`/`cross` and export them**

```python
# add to python/tax/_frontend/array.py (module-level, after concatenate)
def dot(a, b):
    from . import eager
    ra, rb = a._rows(), b._rows()
    if len(ra) != len(rb):
        raise ValueError(f"dot: length mismatch {len(ra)} vs {len(rb)}")
    acc = eager.binary("mul", ra[0], rb[0])
    for i in range(1, len(ra)):
        acc = eager.binary("add", acc, eager.binary("mul", ra[i], rb[i]))
    return acc                              # Expansion

def norm(a):
    from . import eager
    return eager.unary("sqrt", dot(a, a))   # Expansion

def cross(a, b):
    from . import eager
    ra, rb = a._rows(), b._rows()
    if len(ra) != len(rb) or len(ra) not in (2, 3):
        raise ValueError("cross requires two 2- or 3-vectors of equal length")
    mul = lambda x, y: eager.binary("mul", x, y)
    sub = lambda x, y: eager.binary("sub", x, y)
    if len(ra) == 2:
        return sub(mul(ra[0], rb[1]), mul(ra[1], rb[0]))     # scalar Expansion
    c0 = sub(mul(ra[1], rb[2]), mul(ra[2], rb[1]))
    c1 = sub(mul(ra[2], rb[0]), mul(ra[0], rb[2]))
    c2 = sub(mul(ra[0], rb[1]), mul(ra[1], rb[0]))
    return Array(np.stack([c0.coeffs, c1.coeffs, c2.coeffs]), c0.scheme)
```

```python
# add to python/tax/__init__.py
from ._frontend.array import Array, concatenate, stack, dot, cross, norm
__all__ += ["dot", "cross", "norm"]
```

(Merge into the single `from ._frontend.array import ...` line.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_vector_ops.py -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/array.py python/tax/__init__.py python/tests/test_vector_ops.py
git commit -m "feat(py): dot/norm/cross composite vector reductions"
```

---

### Task 8: Array `eval` / `jacobian` / `hessian`

**Files:**
- Modify: `python/tax/_frontend/array.py`
- Test: `python/tests/test_array.py`

**Interfaces:**
- Produces (on `Array`): `eval(dx) -> np.ndarray (K,)`; `jacobian() -> np.ndarray (K, M)` (row i = `self[i].gradient()`); `hessian() -> np.ndarray (K, M, M)`.
- Consumes: `Expansion.eval`/`gradient`/`hessian` (Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# add to python/tests/test_array.py
@needs_toolchain
def test_array_jacobian_and_eval():
    X = tax.variables([1.0, 2.0], order=2)
    Y = tax.concatenate([X[0] * X[1], X[0] + X[1]])    # [x0*x1, x0+x1]
    # value [2, 3]; jacobian [[x1, x0],[1,1]] = [[2,1],[1,1]]
    np.testing.assert_allclose(Y.value(), np.array([2.0, 3.0]), atol=1e-12)
    np.testing.assert_allclose(Y.jacobian(), np.array([[2.0, 1.0], [1.0, 1.0]]), atol=1e-12)
    # eval at dx=(0.1,0.2): row0 (1.1)(2.2)=2.42 ; row1 1.1+2.2=3.3
    np.testing.assert_allclose(Y.eval([0.1, 0.2]), np.array([2.42, 3.3]), atol=1e-12)

@needs_toolchain
def test_array_hessian_shape_and_values():
    X = tax.variables([1.0, 2.0], order=2)
    Y = tax.concatenate([X[0] * X[1], X[0] + X[1]])
    H = Y.hessian()
    assert H.shape == (2, 2, 2)
    np.testing.assert_allclose(H[0], np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)  # x0*x1
    np.testing.assert_allclose(H[1], np.zeros((2, 2)), atol=1e-12)                    # x0+x1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_array.py -k "jacobian or hessian" -v`
Expected: FAIL — `Array` has no `jacobian`/`eval`/`hessian`.

- [ ] **Step 3: Implement the Array accessors**

```python
# add to python/tax/_frontend/array.py (inside class Array)
    def eval(self, dx) -> np.ndarray:
        return np.array([r.eval(dx) for r in self._rows()], dtype=np.float64)

    def jacobian(self) -> np.ndarray:
        return np.stack([r.gradient() for r in self._rows()])

    def hessian(self) -> np.ndarray:
        return np.stack([r.hessian() for r in self._rows()])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_array.py -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/array.py python/tests/test_array.py
git commit -m "feat(py): Array eval/jacobian/hessian (per-row)"
```

---

### Task 9: M2 gate — vector map end-to-end

**Files:**
- Test: `python/tests/test_m2_gate.py`

**Interfaces:**
- Consumes: the full M2 public surface (`tax.variables`, `tax.concatenate`, elementwise ops, `tax.dot`/`norm`, `Array.jacobian`/`value`/`numpy`).
- Produces: the M2 acceptance gate — a multivariate vector map computed eagerly with correct values, Jacobian, and numpy interop.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_m2_gate.py
import numpy as np
import tax
from tax._frontend.array import Array
from tests._helpers import needs_toolchain

@needs_toolchain
def test_vector_map_value_jacobian_numpy():
    X = tax.variables([1.0, 2.0], order=4)
    Y = tax.concatenate([X[0] * X[1], X[0] / X[1]])     # [x0*x1, x0/x1] at (1,2)
    assert isinstance(Y, Array)
    np.testing.assert_allclose(Y.value(), np.array([2.0, 0.5]), atol=1e-12)
    # ∂(x0*x1) = [x1, x0] = [2, 1]; ∂(x0/x1) = [1/x1, -x0/x1^2] = [0.5, -0.25]
    np.testing.assert_allclose(
        Y.jacobian(), np.array([[2.0, 1.0], [0.5, -0.25]]), atol=1e-12
    )
    assert Y.numpy().shape == (2, tax._frontend.scheme.num_monomials(4, 2))

@needs_toolchain
def test_norm_of_vector_map():
    X = tax.variables([3.0, 4.0], order=3)
    r = tax.norm(X)                                      # sqrt(x0^2 + x1^2), value 5
    np.testing.assert_allclose(r.value(), 5.0, atol=1e-12)
    # d/dx0 sqrt(x0^2+x1^2) = x0/r = 3/5 ; d/dx1 = 4/5
    np.testing.assert_allclose(r.gradient(), np.array([0.6, 0.8]), atol=1e-12)
```

`tax._frontend.scheme.num_monomials` is importable; if you prefer, replace the shape assertion with the literal `np.testing.assert_array_equal(Y.numpy().shape, (2, 15))` since `num_monomials(4, 2) = 15`.

- [ ] **Step 2: Run test to verify it fails (or passes immediately)**

Run: `cd python && .venv/bin/python -m pytest tests/test_m2_gate.py -v`
Expected: PASS if Tasks 1–8 are complete (this is an integration gate over existing surface). If it fails, debug the specific op (e.g. `/` path, or `norm`'s `sqrt` on the constant term).

- [ ] **Step 3: (No new implementation expected)**

If `test_norm_of_vector_map` fails on `sqrt`, confirm the constant term of `dot(X, X)` is positive (25 here) so `seriesSqrt` is well-defined. If the Jacobian of `x0/x1` is wrong, confirm `eager.binary("div", ...)` routes through the real `operator/` kernel.

- [ ] **Step 4: Run the whole suite — the M2 gate**

Run: `cd python && .venv/bin/python -m pytest -v`
Expected: PASS (all M1 + M2 tests).

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_m2_gate.py
git commit -m "test(py): M2 gate — multivariate vector map value/jacobian/norm"
```

---

## Self-Review

**Spec coverage (M2 = multivariate + vectors, per the design spec §6.1–6.2 and the M1 roadmap):**
- graded-lex `flat_index`/`unflat_index` cross-checked against C++ → Task 1. ✓
- multivariate coordinate variables → Task 4. ✓
- multi-index `coeff`/`derivative`, `gradient`/`hessian`, `eval` → Task 2 (Expansion), Task 8 (Array). ✓
- `Array` (contiguous `K×nCoeff`, shared scheme), indexing, `value`/`numpy` → Task 3. ✓
- elementwise arithmetic + math (broadcast scalar/Expansion) → Task 5. ✓
- `concatenate`/`stack` → Task 6. ✓
- `dot`/`cross`/`norm` (composite, numpy shape conventions) → Task 7. ✓
- `jacobian`/`hessian`/`eval` on `Array` → Task 8. ✓
- end-to-end vector map → Task 9. ✓
- Out of scope (later plans): named schemes (M3), `tax.jit` fusion + signatures (M4), two-body targets (M5), packaging (M6), batched/one-kernel Array ops (perf; current Array ops are per-row).

**Placeholder scan:** No "TBD"/"similar to Task N"/"add error handling". Every step has complete code and concrete oracles (all hand-derived: `x0*x1` → `[ab,b,a,0,1,0]`; jacobians; `norm` gradient `x/r`).

**Type consistency:** `flat_index(alpha)`, `unflat_index(k, vars)`, `Expansion.coeff(*alpha)`/`derivative(*alpha)`/`gradient()`/`hessian()`/`eval(dx)`, `Array(coeffs, scheme)` with `_rows()`/`_map_unary`/`_map_binary`/`value`/`numpy`/`eval`/`jacobian`/`hessian`, `variables(point, order) -> Array`, `concatenate`/`stack`/`dot`/`norm`/`cross` are used with identical names/signatures across producing and consuming tasks. `eager._embed`/`eager.unary`/`eager.binary` are reused from M1 unchanged. The `coeff` semantics change (Task 2) is explicitly reconciled by rewriting the affected M1 test in the same task.

**Note on a known M1 follow-up:** `tax.pow(x, integer)` still NaNs for non-positive bases (tracked from M1). M2's `pow` dispatch over `Array` inherits this; not addressed here.

---

## Roadmap (after M2)

- **M3 — Named schemes:** `Named(order, axes)` descriptor → `NamedTaylorExpansion<T,N,Axes…>`; `name=` factories; axis-union promotion; named `coeff(x=…, y=…)`, `jacobian("x")`.
- **M4 — `tax.jit` fusion:** whole-function tracer, multi-scheme/multi-output `emit`, options, numba-style signatures; also fix the integer-`pow` lowering (the tracked NaN follow-up).
- **M5 — Targets + regression + perf:** both two-body RHS maps; DACE/C++ regression; benchmarks.
- **M6 — Packaging:** vendored headers, cffi FFI, PCH warm builds, docs.
