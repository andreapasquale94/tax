# Python JIT Layer — M3: Named Schemes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add **named axes** to the Python JIT layer — `tax.variable(x0, order, name="mu")` / `tax.variables(point, order, name="x")`, expansions that compose across axis sets into their union, and axis-addressed accessors (`coeff(x=…)`, `gradient("x")`, `Array.jacobian("x")`) — all eager, single global order (joint-simplex).

**Architecture:** A named expansion's coefficient buffer is **bit-identical to an isotropic expansion over `vars = Σ axis.dim` variables** (C++ `NamedTaylorExpansion<T,N,Axes…>`'s `Inner` is `IsotropicScheme<N, Σdim>`, named.hpp:318). So names are pure Python-side bookkeeping: a `Named` scheme descriptor carries the canonical sorted axes and delegates `cpp_type_string`/`n_coeff`/`descriptor_hash` to its isotropic twin, so the existing eager engine compiles and caches the *same* isotropic kernels. The only named-specific runtime logic is an **axis-remap embed** (scatter an operand's variables into the union's variable layout) — the Python mirror of C++ `NamedTaylorExpansion::embed`. No codegen changes.

**Tech Stack:** Python ≥3.10 (stdlib `math` + numpy + pytest), the M1+M2 pipeline (`tax._frontend` / `tax._codegen`), the system C++23 compiler + Eigen + `tax` headers (for the cross-check probe and eager numeric tests).

## Global Constraints

- **Builds on M1+M2** (branch `feature/python-jit-expansions`, all M1+M2 tasks merged; 54 tests passing). Reuse existing modules; do not duplicate.
- **Test runner:** `cd /Users/andrea/Documents/Codes/tax/python && .venv/bin/python -m pytest ...`. Toolchain-dependent tests import `needs_toolchain` from `tests._helpers`.
- **Single global order (joint-simplex).** All axes of a named expansion share one order N; combining different orders promotes to the max (NOT per-axis / mixed — that stays C++-only). A named expansion maps to C++ `NamedTaylorExpansion<T, N, Axis<name,dim>…>`.
- **Canonical axis order = sorted by name, unique, with consistent dim.** Must match C++ `IsCanonical`: names compared as unsigned char (≡ Python `str` `<` for ASCII names — names MUST be ASCII), shorter-is-less on prefix tie. The same name with two different dims is an error.
- **Named buffer layout == `IsotropicScheme<N, Σdim>`.** Axis blocks occupy variables in canonical (sorted-name) order: the first axis occupies vars `[0, dim0)`, the next `[dim0, dim0+dim1)`, etc. So `Named.var_offset(name) = Σ dims of axes sorted before it`, matching C++ `OffsetOf`.
- **`Named` delegates `cpp_type_string()`/`n_coeff`/`descriptor_hash()` to `Isotropic(order, vars)`** — so `emit`/`run`/the cache key treat a named scheme exactly like its isotropic twin, and named & unnamed of the same `(N, vars)` share the cached `.so`. The only named-specific path is `_embed` (axis-remap).
- **Axis-union promotion:** binary ops embed both operands into the union scheme (union of axes, max order) before the isotropic kernel runs — mirroring C++ `MergedNamedTaylorExpansion` + `embed`.
- **No new kernels / opcodes / `CPP_EXPR` entries.** Named arithmetic/math reuses the isotropic eager kernels verbatim.
- **Static storage, pure-Python base package, graded-lex** — all prior constraints still hold.
- **Scope:** named factories, composition/promotion, and axis-addressed `coeff`/`gradient`/`jacobian`. Out of scope (later/never here): symbolic `deriv`/`integ`/`slice` by axis in Python (C++-only for now), and per-axis *mixed* orders (M-mixed, C++-only).

---

## File Structure

```
python/tax/
├── _frontend/
│   ├── scheme.py        # MODIFY: add Axis, Named (canonical, union, var_offset, axis_var_map, isotropic delegation)
│   ├── eager.py         # MODIFY: _embed gains a Named (axis-remap) branch; _as_expansion unchanged
│   ├── factories.py     # MODIFY: variable()/variables() gain name= (produce Named Expansion/Array)
│   ├── types.py         # MODIFY: Expansion.coeff(**axes) keyword form; gradient(name=None)
│   └── array.py         # MODIFY: Array.jacobian(name=None)
python/tests/
├── test_named_scheme.py # CREATE: Named descriptor (canonical, union, offsets, var-map)
├── test_named.py        # CREATE: factories + named eager + C++ cross-check + accessors
└── test_m3_gate.py      # CREATE: named two-body RHS gate
```

---

### Task 1: `Named` scheme descriptor

**Files:**
- Modify: `python/tax/_frontend/scheme.py`
- Test: `python/tests/test_named_scheme.py`

**Interfaces:**
- Produces (in `scheme.py`):
  - `Axis(name: str, dim: int)` — frozen dataclass; `dim >= 1` else ValueError.
  - `Named(order: int, axes: tuple[Axis, ...])` — frozen dataclass; `__post_init__` requires `axes` already canonical (sorted by name, unique) else ValueError (mirrors the C++ static_assert).
  - `Named.of(order, axes_iterable) -> Named` — classmethod that sorts by name + checks duplicate-name dim consistency (raises on conflict), then constructs.
  - Properties/methods: `.vars -> int` (Σ dim), `.n_coeff -> int`, `.isotropic() -> Isotropic`, `.cpp_type_string() -> str` (delegates), `.descriptor_hash() -> str` (delegates), `.var_offset(name) -> int`, `.dim_of(name) -> int`, `.axis_names() -> tuple[str,...]`, `.union(other: Named) -> Named`, `.axis_var_map(target: Named) -> list[int]`.
- Consumes: `Isotropic`, `num_monomials` (existing).

- [ ] **Step 1: Write the failing tests**

```python
# python/tests/test_named_scheme.py
import pytest
from tax._frontend.scheme import Axis, Named, Isotropic

def test_axis_validation():
    with pytest.raises(ValueError):
        Axis("x", 0)

def test_named_requires_canonical():
    Named(4, (Axis("mu", 1), Axis("x", 4)))          # ok: sorted
    with pytest.raises(ValueError):
        Named(4, (Axis("x", 4), Axis("mu", 1)))      # not sorted
    with pytest.raises(ValueError):
        Named(4, (Axis("x", 4), Axis("x", 4)))       # duplicate name

def test_named_of_sorts_and_checks_dims():
    n = Named.of(4, [Axis("x", 4), Axis("mu", 1)])
    assert n.axes == (Axis("mu", 1), Axis("x", 4))   # sorted mu < x
    with pytest.raises(ValueError):
        Named.of(4, [Axis("x", 4), Axis("x", 2)])    # same name, conflicting dim

def test_vars_ncoeff_and_isotropic_delegation():
    n = Named.of(4, [Axis("mu", 1), Axis("x", 4)])
    assert n.vars == 5
    assert n.n_coeff == Isotropic(4, 5).n_coeff      # 126
    assert n.isotropic() == Isotropic(4, 5)
    assert n.cpp_type_string() == "tax::IsotropicScheme<4, 5>"
    assert n.descriptor_hash() == Isotropic(4, 5).descriptor_hash()

def test_var_offset_and_dim():
    n = Named.of(4, [Axis("mu", 1), Axis("x", 4)])
    assert n.var_offset("mu") == 0
    assert n.var_offset("x") == 1                     # mu(dim1) precedes x
    assert n.dim_of("x") == 4
    with pytest.raises(KeyError):
        n.var_offset("nope")

def test_union_and_var_map():
    x = Named.of(4, [Axis("x", 4)])
    mu = Named.of(2, [Axis("mu", 1)])
    u = x.union(mu)
    assert u.axes == (Axis("mu", 1), Axis("x", 4))    # union, sorted
    assert u.order == 4                               # max order
    assert x.axis_var_map(u) == [1, 2, 3, 4]          # x's vars -> union vars 1..4
    assert mu.axis_var_map(u) == [0]
    with pytest.raises(ValueError):
        Named.of(3, [Axis("x", 4)]).union(Named.of(3, [Axis("x", 2)]))  # dim conflict
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_named_scheme.py -v`
Expected: FAIL — `ImportError: cannot import name 'Axis'`.

- [ ] **Step 3: Add `Axis` and `Named` to `scheme.py`**

```python
# add to python/tax/_frontend/scheme.py
@dataclass(frozen=True)
class Axis:
    name: str
    dim: int

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("Axis.dim must be >= 1")

@dataclass(frozen=True)
class Named:
    order: int
    axes: tuple  # tuple[Axis, ...], canonical: sorted by name, unique

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("Named.order must be >= 0")
        names = [a.name for a in self.axes]
        if names != sorted(names):
            raise ValueError("Named.axes must be sorted by name (use Named.of)")
        if len(set(names)) != len(names):
            raise ValueError("Named.axes has duplicate axis names")

    @classmethod
    def of(cls, order: int, axes) -> "Named":
        by_name: dict[str, Axis] = {}
        for a in axes:
            if a.name in by_name and by_name[a.name].dim != a.dim:
                raise ValueError(f"axis {a.name!r} used with conflicting dim")
            by_name[a.name] = a
        ordered = tuple(sorted(by_name.values(), key=lambda a: a.name))
        return cls(order, ordered)

    @property
    def vars(self) -> int:
        return sum(a.dim for a in self.axes)

    @property
    def n_coeff(self) -> int:
        return num_monomials(self.order, self.vars)

    def isotropic(self) -> Isotropic:
        return Isotropic(self.order, self.vars)

    def cpp_type_string(self) -> str:
        return self.isotropic().cpp_type_string()

    def descriptor_hash(self) -> str:
        # Identical emitted C++ to the isotropic twin -> share the cached .so.
        return self.isotropic().descriptor_hash()

    def axis_names(self) -> tuple:
        return tuple(a.name for a in self.axes)

    def dim_of(self, name: str) -> int:
        for a in self.axes:
            if a.name == name:
                return a.dim
        raise KeyError(name)

    def var_offset(self, name: str) -> int:
        off = 0
        for a in self.axes:
            if a.name == name:
                return off
            off += a.dim
        raise KeyError(name)

    def union(self, other: "Named") -> "Named":
        merged = Named.of(max(self.order, other.order), (*self.axes, *other.axes))
        return merged

    def axis_var_map(self, target: "Named") -> list:
        """Map each of this scheme's variable indices to the target's variable layout."""
        m = [0] * self.vars
        src = 0
        for a in self.axes:
            to = target.var_offset(a.name)
            for l in range(a.dim):
                m[src] = to + l
                src += 1
        return m
```

`Axis`/`Named` use `@dataclass(frozen=True)` — `from dataclasses import dataclass` is already imported in `scheme.py` (used by `Isotropic`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_named_scheme.py -v`
Expected: PASS. Full suite: `cd python && .venv/bin/python -m pytest -q` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/scheme.py python/tests/test_named_scheme.py
git commit -m "feat(py): Named scheme descriptor (canonical axes, union, var-map, isotropic delegation)"
```

---

### Task 2: `name=` factories

**Files:**
- Modify: `python/tax/_frontend/factories.py`
- Test: `python/tests/test_factories.py`

**Interfaces:**
- Produces:
  - `variable(x0, order, name=None) -> Expansion` — when `name` is given, a 1-D named axis `Named.of(order, [Axis(name, 1)])`; row seeds `[x0, 1, 0, …]` (var 0 → flat index 1). When `name is None`, the existing M1 univariate isotropic behavior (unchanged).
  - `variables(point, order, name=None) -> Array` — when `name` is given, a single `M`-dim named axis `Named.of(order, [Axis(name, M)])` (M = len(point)); row i seeds index 0 = point[i] and (order≥1) index i+1 = 1.0 (var i → flat i+1). When `name is None`, the existing M2 isotropic behavior (unchanged).
- Consumes: `Named`, `Axis`, `Isotropic`, `Expansion`, `Array`.

- [ ] **Step 1: Write the failing tests**

```python
# add to python/tests/test_factories.py
import numpy as np
import tax
from tax._frontend.scheme import Named, Axis

def test_named_variable_1d():
    mu = tax.variable(398600.4418, order=4, name="mu")
    assert mu.scheme == Named.of(4, [Axis("mu", 1)])
    expected = np.zeros(mu.scheme.n_coeff)
    expected[0] = 398600.4418
    expected[1] = 1.0                       # var 0 -> flat 1
    assert np.array_equal(mu.numpy(), expected)

def test_named_variables_axis():
    X = tax.variables([1.0, 0.0, 0.0, 1.0], order=4, name="x")
    assert X.scheme == Named.of(4, [Axis("x", 4)])
    assert len(X) == 4
    assert X[0].numpy()[0] == 1.0 and X[0].numpy()[1] == 1.0      # x0 = 1 + dx0
    assert X[3].numpy()[0] == 1.0 and X[3].numpy()[4] == 1.0      # x3 = 1 + dx3 (var3 -> flat 4)

def test_name_none_still_isotropic():
    x = tax.variable(2.5, order=4)           # unchanged M1 path
    from tax._frontend.scheme import Isotropic
    assert x.scheme == Isotropic(4, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_factories.py -k named -v`
Expected: FAIL — `variable() got an unexpected keyword argument 'name'`.

- [ ] **Step 3: Add `name=` to the factories**

```python
# python/tax/_frontend/factories.py — replace variable() and variables() with:
from __future__ import annotations
import numpy as np
from .scheme import Isotropic, Named, Axis
from .types import Expansion
from .array import Array

def variable(x0, order, name=None):
    if name is None:
        scheme = Isotropic(order, 1)
        coeffs = np.zeros(scheme.n_coeff, dtype=np.float64)
        coeffs[0] = float(x0)
        if order >= 1:
            coeffs[1] = 1.0
        return Expansion(coeffs, scheme)
    scheme = Named.of(order, [Axis(name, 1)])
    coeffs = np.zeros(scheme.n_coeff, dtype=np.float64)
    coeffs[0] = float(x0)
    if order >= 1:
        coeffs[1] = 1.0                       # the axis's single var -> flat index 1
    return Expansion(coeffs, scheme)

def variables(point, order, name=None):
    point = list(point)
    M = len(point)
    if M < 1:
        raise ValueError("variables(): point must have at least one element")
    if name is None:
        scheme = Isotropic(order, M)
    else:
        scheme = Named.of(order, [Axis(name, M)])
    data = np.zeros((M, scheme.n_coeff), dtype=np.float64)
    for i in range(M):
        data[i, 0] = float(point[i])
        if order >= 1:
            data[i, i + 1] = 1.0              # var i -> flat index i+1 (single axis block)
    return Array(data, scheme)
```

(The isotropic branches reproduce the existing M1/M2 behavior verbatim — keep them identical so the `name is None` tests still pass.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_factories.py -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/factories.py python/tests/test_factories.py
git commit -m "feat(py): name= on variable()/variables() (named-axis factories)"
```

---

### Task 3: Named eager embed + C++ cross-check

**Files:**
- Modify: `python/tax/_frontend/eager.py`
- Test: `python/tests/test_named.py`

**Interfaces:**
- Produces: `eager._embed(x, target)` gains a `Named` branch — when `target` is a `Named`, scatter `x`'s coefficients into the target's variable layout via `x.scheme.axis_var_map(target)` (the Python mirror of C++ `NamedTaylorExpansion::embed`). The isotropic branch (graded-lex prefix) is preserved unchanged. `eager.unary`/`binary` then flow named operands through transparently (because `Named` delegates `cpp_type_string`/`n_coeff`/`descriptor_hash`).
- Consumes: `scheme.flat_index`, `scheme.unflat_index`, `scheme.Named`.

- [ ] **Step 1: Write the failing tests**

```python
# python/tests/test_named.py
import numpy as np
import pytest
import tax
from tax._frontend import eager
from tax._frontend.scheme import Named, Axis, flat_index, unflat_index
from tax._frontend.types import Expansion
from tax._codegen import build, load
from tests._helpers import needs_toolchain

def test_named_embed_scatters_axis_block():
    # x-only expansion (axis x, dim 2, order 2): put a marker on x's var 1 (flat index 2)
    src = Named.of(2, [Axis("x", 2)])
    coeffs = np.zeros(src.n_coeff)
    coeffs[flat_index((0, 1))] = 7.0          # exponent on x's 2nd coordinate
    e = Expansion(coeffs, src)
    target = src.union(Named.of(2, [Axis("mu", 1)]))   # {mu:1, x:2}: mu var0, x vars1-2
    out = eager._embed(e, target)
    # x's var1 -> union var2; the marker must land at flat_index of e_(union var2)
    dst = [0, 0, 0]
    dst[target.var_offset("x") + 1] = 1
    assert out[flat_index(tuple(dst))] == 7.0
    assert out.shape == (target.n_coeff,)

# --- C++ cross-check: a named product in Python must match the compiled C++ named product ---
_PROBE = r'''
#include <tax/tax.hpp>
#include <algorithm>
#include <array>
extern "C" int tax_kernel(const double* const*, double* const* outs) noexcept {
    using namespace tax;
    std::array<double, 4> x0{1.0, 0.0, 0.0, 1.0};
    auto xs = variables<"x", 4>(x0);            // NE<4, Axis<"x",4>>[4]
    auto mu = variable<"mu", 4>(398600.4418);   // NE<4, Axis<"mu",1>>
    auto f = mu * xs[0];                          // NE<4, Axis<"mu",1>, Axis<"x",4>>
    using F = decltype(f);
    std::copy_n(f.inner().coefficients().data(), F::nCoefficients, outs[0]);
    return 0;
}
'''

@needs_toolchain
def test_named_product_matches_cpp(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    so = build.compile_kernel(_PROBE, "named_probe_mu_x", cxx=build.find_compiler(),
                              includes=build.include_dirs(), opt_flags=["-O3"])
    n = Named.of(4, [Axis("mu", 1), Axis("x", 4)]).n_coeff       # 126
    (expected,) = load.call_kernel(load.load_kernel(so), [], [n])
    mu = tax.variable(398600.4418, order=4, name="mu")
    xs = tax.variables([1.0, 0.0, 0.0, 1.0], order=4, name="x")
    f = mu * xs[0]
    assert f.scheme == Named.of(4, [Axis("mu", 1), Axis("x", 4)])
    np.testing.assert_allclose(f.numpy(), expected, rtol=1e-12, atol=1e-9)

@needs_toolchain
def test_named_unary_preserves_axes():
    x = tax.variable(0.0, order=5, name="t")
    f = tax.sin(x)
    assert f.scheme == Named.of(5, [Axis("t", 1)])
    np.testing.assert_allclose(f.coeff(1) if False else f.numpy()[1], 1.0, atol=1e-12)
```

(The last test's odd-looking line keeps it to a positional buffer check; `coeff` keyword form arrives in Task 4.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_named.py -v`
Expected: FAIL — `test_named_embed_scatters_axis_block` fails because `_embed`'s isotropic branch raises `cannot embed` on differing `vars` (4 vs 3... here 2 vs 3), i.e. there is no Named branch yet.

- [ ] **Step 3: Add the Named branch to `_embed`**

```python
# python/tax/_frontend/eager.py — replace _embed with:
def _embed(x: Expansion, target) -> np.ndarray:
    """Embed an expansion into `target`. Isotropic: graded-lex prefix. Named: axis-remap scatter."""
    if x.scheme == target:
        return x.coeffs
    from .scheme import Named
    if isinstance(target, Named):
        src = x.scheme  # a Named scheme over a subset of target's axes
        vmap = src.axis_var_map(target)
        out = np.zeros(target.n_coeff, dtype=np.float64)
        for k in range(src.n_coeff):
            c = x.coeffs[k]
            if c == 0.0:
                continue
            a_src = unflat_index(k, src.vars)
            a_dst = [0] * target.vars
            for j in range(src.vars):
                a_dst[vmap[j]] = a_src[j]
            out[flat_index(a_dst)] = c
        return out
    if x.scheme.vars != target.vars or x.scheme.order > target.order:
        raise ValueError(f"cannot embed {x.scheme} into {target}")
    out = np.zeros(target.n_coeff, dtype=np.float64)
    out[: x.scheme.n_coeff] = x.coeffs        # univariate / graded-lex prefix
    return out
```

Add the import at the top of `eager.py` (next to the existing `from .scheme import Isotropic`):

```python
from .scheme import Isotropic, flat_index, unflat_index
```

No change is needed to `unary`/`binary`/`run`: a `Named` result scheme delegates `cpp_type_string`/`n_coeff`/`descriptor_hash`, so `single_op_graph`/`emit`/the cache key treat it as its isotropic twin, while `union` (Named→Named) and the new `_embed` Named branch handle promotion.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_named.py -v`
Expected: PASS (incl. the `@needs_toolchain` C++ cross-check, which RUNS). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/eager.py python/tests/test_named.py
git commit -m "feat(py): named axis-remap embed; named eager matches compiled C++"
```

---

### Task 4: Named accessors (`coeff(**axes)`, `gradient(name)`, `Array.jacobian(name)`)

**Files:**
- Modify: `python/tax/_frontend/types.py`
- Modify: `python/tax/_frontend/array.py`
- Test: `python/tests/test_named.py`

**Interfaces:**
- Produces:
  - `Expansion.coeff(*alpha, **axes)` — positional multi-index (existing) OR keyword axis form (named only). Each kwarg names a **1-D** axis with an int exponent; missing axes default 0; raises if positional+keyword mixed, if the scheme isn't Named, if a name is unknown, or if a named axis has dim>1 (use positional).
  - `Expansion.gradient(name=None)` — full gradient `(vars,)` if `name is None`; else the `(dim,)` slice `∂f/∂(axis's vars)` (named only).
  - `Array.jacobian(name=None)` — full `(K, vars)` if `name is None`; else the `(K, dim)` column slice for that axis (named only).
- Consumes: `Named.var_offset`/`dim_of`, the existing positional `coeff`/`gradient`.

- [ ] **Step 1: Write the failing tests**

```python
# add to python/tests/test_named.py
@needs_toolchain
def test_named_coeff_keyword_and_gradient_axis():
    # f = a*b for two 1-D axes a, b at (a0,b0) = (2, 3), order 2 -> axes {a, b}
    a = tax.variable(2.0, order=2, name="a")
    b = tax.variable(3.0, order=2, name="b")
    f = a * b
    # union {a, b}: a var0, b var1; f = (2+da)(3+db)
    assert f.coeff(a=0, b=0) == 6.0
    assert f.coeff(a=1, b=0) == 3.0       # ∂/∂a coeff = b0 = 3
    assert f.coeff(a=1, b=1) == 1.0       # mixed da*db coeff
    assert np.allclose(f.gradient(), [3.0, 2.0])      # [b0, a0]
    assert np.allclose(f.gradient("a"), [3.0])
    assert np.allclose(f.gradient("b"), [2.0])

def test_named_coeff_keyword_validation():
    X = tax.variables([1.0, 2.0], order=2, name="x")   # x is dim 2 (not 1-D)
    f = X[0]
    with pytest.raises(ValueError):
        f.coeff(x=1)                       # dim>1 axis via keyword -> error
    with pytest.raises(ValueError):
        f.coeff(0, 0, x=1)                 # positional + keyword mixed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_named.py -k "keyword or gradient_axis" -v`
Expected: FAIL — `coeff()` takes no keyword arguments / `gradient()` takes no `name`.

- [ ] **Step 3: Extend `coeff`/`gradient` in `types.py` and `jacobian` in `array.py`**

```python
# python/tax/_frontend/types.py — replace coeff() with:
    def coeff(self, *alpha, **axes):
        if axes:
            if alpha:
                raise ValueError("coeff: pass positional exponents OR axis keywords, not both")
            from .scheme import Named
            if not isinstance(self.scheme, Named):
                raise ValueError("coeff(**axes): keyword form requires a named expansion")
            a = [0] * self.scheme.vars
            for name, e in axes.items():
                if self.scheme.dim_of(name) != 1:
                    raise ValueError(
                        f"coeff keyword form supports 1-D axes only; {name!r} is multi-dim "
                        "— use positional coeff(*exponents)"
                    )
                a[self.scheme.var_offset(name)] = int(e)
            alpha = tuple(a)
        if len(alpha) != self.scheme.vars:
            raise ValueError(
                f"coeff expects {self.scheme.vars} exponents, got {len(alpha)}"
            )
        if any(x < 0 for x in alpha):
            raise ValueError("coeff: negative exponent")
        k = flat_index(alpha)
        return float(self.coeffs[k]) if k < self.scheme.n_coeff else 0.0
```

```python
# python/tax/_frontend/types.py — replace gradient() with:
    def gradient(self, name=None) -> np.ndarray:
        M = self.scheme.vars
        g = np.empty(M, dtype=np.float64)
        for i in range(M):
            e = tuple(1 if j == i else 0 for j in range(M))
            g[i] = self.coeff(*e)
        if name is None:
            return g
        from .scheme import Named
        if not isinstance(self.scheme, Named):
            raise ValueError("gradient(name): requires a named expansion")
        off = self.scheme.var_offset(name)
        return g[off: off + self.scheme.dim_of(name)]
```

(The positional `coeff` body is unchanged from M2 — it's just now reachable after the keyword-to-positional conversion. `derivative`/`hessian`/`eval` continue to call `coeff` positionally and are unaffected.)

```python
# python/tax/_frontend/array.py — replace jacobian() with:
    def jacobian(self, name=None) -> np.ndarray:
        J = np.stack([r.gradient() for r in self._rows()])   # (K, vars)
        if name is None:
            return J
        from .scheme import Named
        if not isinstance(self.scheme, Named):
            raise ValueError("jacobian(name): requires a named Array")
        off = self.scheme.var_offset(name)
        return J[:, off: off + self.scheme.dim_of(name)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_named.py -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/types.py python/tax/_frontend/array.py python/tests/test_named.py
git commit -m "feat(py): named accessors — coeff(**axes), gradient(name), Array.jacobian(name)"
```

---

### Task 5: `__pow__` operator (prerequisite for the `** 1.5` in the gate)

**Context:** M1 added the `tax.pow(x, y)` free function and the `pow` opcode, but never the `__pow__` operator on `Expansion`/`Array`. The design's two-body examples (and the gate in Task 6) use `(…) ** 1.5`, so `**` must work. This is a thin sugar over the existing, verified `pow` path — no new kernel/opcode.

**Files:**
- Modify: `python/tax/_frontend/types.py`
- Modify: `python/tax/_frontend/array.py`
- Test: `python/tests/test_array.py`

**Interfaces:**
- Produces: `Expansion.__pow__(self, p) -> Expansion` (delegates to `eager.binary("pow", self, p)`); `Array.__pow__(self, p) -> Array` (delegates to `self._map_binary("pow", p)`). `p` is an int or float exponent (promoted to a constant expansion by the existing `binary`/`_map_binary` path). No `__rpow__` (scalar ** expansion is out of scope).
- Consumes: `eager.binary` (Expansion), `Array._map_binary` (M2 Task 5).

- [ ] **Step 1: Write the failing tests**

```python
# add to python/tests/test_array.py
@needs_toolchain
def test_pow_operator_matches_pow_function():
    import numpy as np, tax
    x = tax.variable(0.0, order=4)
    b = x * x + 1.0                       # 1 + dx^2 (value 1 > 0)
    np.testing.assert_allclose((b ** 1.5).numpy(), tax.pow(b, 1.5).numpy(), atol=1e-12)
    np.testing.assert_allclose(((x + 2.0) ** 2).numpy(), tax.pow(x + 2.0, 2).numpy(), atol=1e-12)

@needs_toolchain
def test_array_pow_operator_elementwise():
    import numpy as np, tax
    X = tax.variables([2.0, 3.0], order=2)
    Y = X ** 2                            # elementwise square
    np.testing.assert_allclose(Y.value(), np.array([4.0, 9.0]), atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && .venv/bin/python -m pytest tests/test_array.py -k pow -v`
Expected: FAIL — `unsupported operand type(s) for ** : 'Expansion'` / `'Array'`.

- [ ] **Step 3: Add `__pow__` to both classes**

```python
# add to python/tax/_frontend/types.py, inside class Expansion (next to the other dunders)
    def __pow__(self, p):
        from .eager import binary
        return binary("pow", self, p)
```

```python
# add to python/tax/_frontend/array.py, inside class Array (next to the other dunders)
    def __pow__(self, p):
        return self._map_binary("pow", p)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && .venv/bin/python -m pytest tests/test_array.py -k pow -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/types.py python/tax/_frontend/array.py python/tests/test_array.py
git commit -m "feat(py): __pow__ operator on Expansion/Array (sugar over the pow opcode)"
```

---

### Task 6: M3 gate — named two-body RHS

**Files:**
- Test: `python/tests/test_m3_gate.py`

**Interfaces:**
- Consumes: the full named surface (`variable`/`variables` with `name=`, composition, `tax.concatenate`, elementwise math, `**`, `Array.value`/`jacobian(name)`).
- Produces: the M3 acceptance gate — the named planar two-body RHS map evaluated eagerly, with correct value, ∂/∂x Jacobian block, and parameter-sensitivity ∂/∂mu.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_m3_gate.py
import numpy as np
import tax
from tax._frontend.array import Array
from tax._frontend.scheme import Named, Axis
from tests._helpers import needs_toolchain

MU = 398600.4418

@needs_toolchain
def test_named_two_body_rhs():
    x = tax.variables([1.0, 0.0, 0.0, 1.0], order=4, name="x")    # rx, ry, vx, vy
    mu = tax.variable(MU, order=4, name="mu")

    def rhs(x, mu):
        r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
        return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])

    dx = rhs(x, mu)
    assert isinstance(dx, Array)
    assert dx.scheme == Named.of(4, [Axis("mu", 1), Axis("x", 4)])   # union, M=5

    # value of the RHS at the state (r = 1): [vx, vy, -mu*rx, -mu*ry] = [0, 1, -mu, 0]
    np.testing.assert_allclose(dx.value(), np.array([0.0, 1.0, -MU, 0.0]), rtol=1e-9, atol=1e-6)

    # ∂(rhs)/∂x : the state-transition block (4x4), x = [rx, ry, vx, vy] at (1,0,0,1)
    jac_x = dx.jacobian("x")
    expected_x = np.array([
        [0.0, 0.0, 1.0, 0.0],     # ∂vx
        [0.0, 0.0, 0.0, 1.0],     # ∂vy
        [2 * MU, 0.0, 0.0, 0.0],  # ∂(-mu rx / r^3) ; at r=1: -mu(1 - 3 rx^2) = 2mu
        [0.0, -MU, 0.0, 0.0],     # ∂(-mu ry / r^3) ; at r=1: -mu
    ])
    np.testing.assert_allclose(jac_x, expected_x, rtol=1e-9, atol=1e-6)

    # ∂(rhs)/∂mu : parameter sensitivity (4x1) = [0, 0, -rx/r^3, -ry/r^3] = [0, 0, -1, 0]
    jac_mu = dx.jacobian("mu")
    np.testing.assert_allclose(jac_mu, np.array([[0.0], [0.0], [-1.0], [0.0]]),
                               rtol=1e-9, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails (or passes immediately)**

Run: `cd python && .venv/bin/python -m pytest tests/test_m3_gate.py -v`
Expected: PASS if Tasks 1–5 are complete (this is an integration gate over the named surface). If it fails, debug the specific op — most likely the union/embed for the mixed `{x}`-vs-`{mu,x}` operands in `concatenate`, or `jacobian("x")` column slicing. (`** 1.5` requires Task 5's `__pow__`.)

- [ ] **Step 3: (No new implementation expected)**

If the value is right but the Jacobian is shifted, confirm the axis-block slice in `Array.jacobian("x")` uses `var_offset("x") == 1` (mu occupies var 0). If `concatenate` errors, confirm its `_embed` reaches the Named branch for the `x[2]`/`x[3]` operands (axes `{x}`) being lifted into the union `{mu, x}`.

- [ ] **Step 4: Run the whole suite — the M3 gate**

Run: `cd python && .venv/bin/python -m pytest -v`
Expected: PASS (all M1 + M2 + M3 tests).

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_m3_gate.py
git commit -m "test(py): M3 gate — named two-body RHS value + jacobian(x) + jacobian(mu)"
```

---

## Self-Review

**Spec coverage (M3 = named schemes, single global order — design §6.1–6.2 + M2 roadmap):**
- `Named` scheme descriptor (canonical sorted axes, single order) → Task 1. ✓
- Maps to `NamedTaylorExpansion<T,N,Axes…>` semantics, reusing the isotropic buffer/codegen → Tasks 1 (delegation) + 3 (embed). ✓
- `name=` factories (`variable`, `variables`) → Task 2. ✓
- Axis-union promotion (compose across axis sets) → Task 3 (`_embed` Named branch + `union`). ✓
- Named eager == compiled C++ (cross-check) → Task 3. ✓
- Axis-addressed accessors (`coeff(x=…)`, `gradient("x")`, `jacobian("x")`) → Task 4. ✓
- `__pow__` operator (`** 1.5`), prerequisite gap found during execution → Task 5. ✓
- Named two-body RHS target → Task 6. ✓
- Out of scope (documented): symbolic `deriv`/`integ`/`slice` by axis in Python; per-axis *mixed* orders (C++-only). The M1 `pow(x, integer)` NaN follow-up is unrelated (named `**1.5` is real-pow with positive base — fine).

**Placeholder scan:** No "TBD"/"similar to Task N"/"add error handling". Every step has complete code and concrete oracles (named product cross-checked vs compiled C++; two-body Jacobian hand-derived: `∂(-mu·rx/r³)/∂rx = -mu(r⁻³ - 3rx²r⁻⁵) = 2mu` at r=1, etc.).

**Type consistency:** `Axis(name, dim)`, `Named(order, axes)`/`Named.of`, `.vars`/`.n_coeff`/`.isotropic()`/`.cpp_type_string()`/`.descriptor_hash()`/`.var_offset`/`.dim_of`/`.union`/`.axis_var_map`, `variable(x0, order, name=)`, `variables(point, order, name=)`, `Expansion.coeff(*alpha, **axes)`/`gradient(name=None)`, `Array.jacobian(name=None)`, `eager._embed(x, target)` are used with identical names/signatures across producing and consuming tasks. `Named` delegation makes `eager.unary`/`binary`/`run`/`emit`/`single_op_graph` work unchanged — verified against the M1/M2 implementations (`emit` uses only `scheme.cpp_type_string()`/`.n_coeff`; `graph.canonical()` uses `.descriptor_hash()`; `binary` uses `.union` + `_embed`).

---

## Roadmap (after M3)

- **M4 — `tax.jit` fusion:** whole-function tracer, multi-scheme/multi-output `emit`, options (`opt/cache/compiler/scalar/batch/static_argnums/dump`), numba-style explicit signatures; also the integer-`pow` lowering fix (tracked from M1). Both two-body RHS maps run under `@tax.jit` (bare and pinned-signature).
- **M5 — Targets + regression + perf:** both two-body maps as e2e tests; DACE/C++ regression; eager-vs-jit-vs-C++ benchmarks.
- **M6 — Packaging:** vendored `tax`/Eigen headers, cffi FFI, PCH warm builds, docs.
```
