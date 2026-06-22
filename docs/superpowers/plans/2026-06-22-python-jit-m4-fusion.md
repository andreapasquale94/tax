# Python JIT Layer — M4: `tax.jit` Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `@tax.jit` — trace a whole Python function into one graph, fuse it into a single compiled kernel (no per-op FFI, no intermediate buffers), with trace-on-first-call **and** numba-style explicit signatures; plus the integer-exponent `pow` lowering (so `x ** <int>` is correct for any base).

**Architecture:** The M1 backend (`emit`/`run`/`call_kernel`/cache) is already fully general — it lowers an arbitrary multi-node, multi-input, multi-output graph and the kernel ABI is `int tax_kernel(const double* const* ins, double* const* outs)`. So M4 is **purely additive Python**: a tracer that records a function's ops into one `Graph`, and a `jit` driver that compiles+caches it and marshals buffers. Because `emit` types every node's C++ by its scheme (an `Op`'s operands must share a C++ type), the whole trace runs in **one global scheme** G = the union of the inputs' schemes; inputs are embedded into G in Python at call time (reusing `eager._embed`), and one single-scheme kernel is emitted.

**Tech Stack:** Python ≥3.10 (numpy + pytest), the M1–M3 pipeline (`tax._frontend`/`tax._codegen`), the system C++23 compiler + Eigen + `tax` headers (for the numeric tests).

## Global Constraints

- **Builds on M1+M2+M3** (branch `feature/python-jit-expansions`, all merged; 73 tests passing). Reuse existing modules; do not duplicate.
- **Test runner:** `cd /Users/andrea/Documents/Codes/tax/python && .venv/bin/python -m pytest ...`. Toolchain tests import `needs_toolchain` from `tests._helpers`.
- **No backend changes.** `emit`/`run`/`call_kernel`/`build`/the cache key are reused verbatim. M4 adds a tracer + jit driver on top. The ONE codegen addition is the `powint:{n}` opcode (Task 5).
- **One dispatch layer, two modes.** `eager.unary`/`binary` become polymorphic: if any operand is a `Tracer` → record a graph node and return a `Tracer`; else → the existing eager compile+run. Every op built on `unary`/`binary` (operators, `mathfns`, `dot`/`norm`) then works in both modes for free.
- **Trace in one global scheme.** `G = ` fold of `.union` over all `Expansion`/`Array` input schemes (Isotropic or Named). Every traced node has scheme G. Inputs are embedded to G at call via `eager._embed` (identity when already G; graded-lex prefix / named axis-remap otherwise). The jit output scheme is G. *Documented consequence:* if a jit function ignores some input, the output is promoted to the full input-union G (broader than eager would produce for that specific function). For functions where all inputs interact — including both two-body RHS maps — G equals eager's result scheme exactly, so **jit numerics == eager numerics**.
- **Scalars are baked.** A plain Python `int`/`float` argument is a compile-time constant: when combined with a tracer it becomes a `Const` node (captured in `graph.canonical()` → re-traced/re-compiled if its value changes). Scalars contribute no input buffer and no axes to G.
- **Cache key** is the existing `sha256(ABI ‖ lib_version ‖ compiler_id ‖ flags ‖ scalar ‖ graph_canonical)` — the fused graph's `canonical()` already captures the whole computation, so identical traced functions share a `.so`.
- **Static storage, pure-Python base, graded-lex** — all prior constraints hold.
- **Scope:** lazy `@tax.jit` + explicit signatures, integer-`pow` lowering, the `dump` option, and both two-body RHS maps under jit. Out of scope (later/never): runtime-scalar kernel parameters (scalars are baked, not passed live), `batch=K`/`scalar=float32`/`march_native` options (the option *plumbing* is added but only `opt`/`dump`/`cache` wired; batch is M5+), and an MLIR backend.

---

## File Structure

```
python/tax/_frontend/
├── trace.py      # CREATE: TraceBuilder, Tracer, ArrayTracer, trace_function
├── jit.py        # CREATE: tax.jit decorator (lazy + signatures), type specs (f64/Expansion/Array), call marshaling
├── eager.py      # MODIFY: unary/binary become Tracer-polymorphic
├── mathfns.py    # MODIFY: dispatch ArrayTracer (vector) vs Tracer/Expansion (scalar)
├── array.py      # MODIFY: concatenate/cross trace-aware (group tracers); __pow__ int path (Task 5)
├── types.py      # MODIFY: Expansion.__pow__ int path (Task 5)
└── ...
python/tax/_codegen/emit_cpp.py  # MODIFY (Task 5): powint:{n} opcode
python/tax/__init__.py            # MODIFY: export jit, f64, Expansion-spec, Array-spec
python/tests/
├── test_trace.py     # CREATE: tracer + graph-structure (no toolchain)
├── test_jit.py       # CREATE: jit numerics == eager; signatures; dump
├── test_pow_int.py   # CREATE: integer-pow correctness incl. negative base
└── test_m4_gate.py   # CREATE: both two-body RHS under @tax.jit
```

---

### Task 1: `Tracer` + polymorphic op dispatch

**Files:**
- Create: `python/tax/_frontend/trace.py`
- Modify: `python/tax/_frontend/eager.py`
- Modify: `python/tax/_frontend/mathfns.py`
- Test: `python/tests/test_trace.py`

**Interfaces:**
- Produces (in `trace.py`): `TraceBuilder` (`.nodes: list`, `.add(node) -> int`); `Tracer(builder, node: int, scheme)` with `.builder`/`.node`/`.scheme` and arithmetic dunders (`__add__/__radd__/__sub__/__rsub__/__mul__/__rmul__/__truediv__/__rtruediv__/__neg__`) that route to `eager.binary`/`eager.unary`.
- Produces (in `eager.py`): `unary`/`binary` gain a leading `Tracer` branch — append an `Op` node (scalars → a `Const` node in the operand's scheme) and return a `Tracer`; the existing eager body is the `else`.
- Consumes: `ir.{Var, Const, Op}`, `Expansion` (eager path unchanged).

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_trace.py
from tax._frontend.trace import TraceBuilder, Tracer
from tax._frontend.ir import Var, Const, Op
from tax._frontend.scheme import Isotropic
import tax

def test_tracer_records_ops_not_eager():
    b = TraceBuilder()
    s = Isotropic(5, 1)
    x = Tracer(b, b.add(Var(0, s)), s)
    y = tax.sin(x) * tax.exp(x)                 # 2 unary ops + 1 mul
    assert isinstance(y, Tracer)
    kinds = [type(n).__name__ for n in b.nodes]
    assert kinds == ["Var", "Op", "Op", "Op"]
    assert b.nodes[1].opcode == "sin" and b.nodes[1].operands == (0,)
    assert b.nodes[2].opcode == "exp" and b.nodes[2].operands == (0,)
    assert b.nodes[3].opcode == "mul" and b.nodes[3].operands == (1, 2)
    assert y.node == 3

def test_tracer_scalar_becomes_const():
    b = TraceBuilder()
    s = Isotropic(3, 1)
    x = Tracer(b, b.add(Var(0, s)), s)
    y = 2.0 * x                                 # scalar -> Const node, then mul
    assert isinstance(b.nodes[1], Const) and b.nodes[1].value == 2.0
    assert b.nodes[2].opcode == "mul"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_trace.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tax._frontend.trace'`.

- [ ] **Step 3: Implement `trace.py` (scalar Tracer) and make dispatch polymorphic**

```python
# python/tax/_frontend/trace.py
from __future__ import annotations
from .ir import Var, Const, Op, Graph

class TraceBuilder:
    def __init__(self):
        self.nodes: list = []

    def add(self, node) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

class Tracer:
    """A symbolic stand-in for a scalar Expansion during tracing."""
    __slots__ = ("builder", "node", "scheme")

    def __init__(self, builder, node, scheme):
        self.builder = builder
        self.node = node
        self.scheme = scheme

    def __add__(self, o):
        from .eager import binary; return binary("add", self, o)
    def __radd__(self, o):
        from .eager import binary; return binary("add", o, self)
    def __sub__(self, o):
        from .eager import binary; return binary("sub", self, o)
    def __rsub__(self, o):
        from .eager import binary; return binary("sub", o, self)
    def __mul__(self, o):
        from .eager import binary; return binary("mul", self, o)
    def __rmul__(self, o):
        from .eager import binary; return binary("mul", o, self)
    def __truediv__(self, o):
        from .eager import binary; return binary("div", self, o)
    def __rtruediv__(self, o):
        from .eager import binary; return binary("div", o, self)
    def __neg__(self):
        from .eager import unary; return unary("neg", self)
    def __pow__(self, p):
        from .eager import binary; return binary("pow", self, p)

def _trace_operand(builder, v, scheme) -> int:
    """A Tracer contributes its node; a Python scalar becomes a Const node."""
    if isinstance(v, Tracer):
        return v.node
    return builder.add(Const(float(v), scheme))

def trace_binary(opcode, a, b) -> Tracer:
    builder = a.builder if isinstance(a, Tracer) else b.builder
    scheme = a.scheme if isinstance(a, Tracer) else b.scheme
    na = _trace_operand(builder, a, scheme)
    nb = _trace_operand(builder, b, scheme)
    return Tracer(builder, builder.add(Op(opcode, (na, nb), scheme)), scheme)

def trace_unary(opcode, x) -> Tracer:
    return Tracer(x.builder, x.builder.add(Op(opcode, (x.node,), x.scheme)), x.scheme)
```

```python
# python/tax/_frontend/eager.py — modify unary() and binary():
# At the TOP of binary(), before the eager body, add:
def binary(opcode: str, a, b):
    from .trace import Tracer, trace_binary
    if isinstance(a, Tracer) or isinstance(b, Tracer):
        return trace_binary(opcode, a, b)
    # ... existing eager body unchanged (ref/_as_expansion/union/_embed/run) ...

# At the TOP of unary(), before the eager body, add:
def unary(opcode: str, x):
    from .trace import Tracer, trace_unary
    if isinstance(x, Tracer):
        return trace_unary(opcode, x)
    # ... existing eager body unchanged ...
```

(Keep the existing eager bodies verbatim below the new guard. The lazy `from .trace import …` avoids an import cycle: `trace.py` imports nothing from `eager` at module load.)

```python
# python/tax/_frontend/mathfns.py — make the unary dispatch Tracer-aware.
# Current _make_unary does: if isinstance(x, Array): x._map_unary(op); else unary(op, x).
# `unary` now handles Tracer, so the scalar branch already covers Tracer.
# Only the VECTOR branch must also recognise ArrayTracer (added in Task 2). For Task 1,
# no change is required here — a scalar Tracer flows through `unary(op, x)` correctly.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_trace.py -v`
Expected: PASS. Full suite: `cd python && .venv/bin/python -m pytest -q` — no regressions (the eager path is unchanged; the new branch only fires for `Tracer` operands).

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/trace.py python/tax/_frontend/eager.py python/tests/test_trace.py
git commit -m "feat(py): scalar Tracer + Tracer-polymorphic op dispatch"
```

---

### Task 2: `ArrayTracer` + trace-aware vector ops

**Files:**
- Modify: `python/tax/_frontend/trace.py`
- Modify: `python/tax/_frontend/mathfns.py`
- Modify: `python/tax/_frontend/array.py`
- Test: `python/tests/test_trace.py`

**Interfaces:**
- Produces (in `trace.py`): `ArrayTracer(rows: list[Tracer], scheme)` with `._rows()`, `__len__`, `__getitem__` (int → `Tracer`, slice → `ArrayTracer`), `_map_unary(op)`/`_map_binary(op, other)` (per-row, returning `ArrayTracer`), and elementwise arithmetic dunders.
- Produces: `array.concatenate` and `array.cross` gain a trace branch — when any item/operand is a `Tracer`/`ArrayTracer`, group the tracers (no buffer ops) into an `ArrayTracer`; `dot`/`norm` already work (built on `binary`/`unary`). `mathfns` unary dispatch recognises `ArrayTracer` as a vector.
- Consumes: `Tracer`, `trace_binary`/`trace_unary`.

- [ ] **Step 1: Write the failing test**

```python
# add to python/tests/test_trace.py
from tax._frontend.trace import ArrayTracer

def test_array_tracer_concatenate_and_dot():
    b = TraceBuilder()
    s = Isotropic(2, 2)
    X = ArrayTracer([Tracer(b, b.add(Var(i, s)), s) for i in range(2)], s)
    assert isinstance(X[0], Tracer) and len(X) == 2
    Y = tax.concatenate([X[0] * X[1], X[0] + X[1]])
    assert isinstance(Y, ArrayTracer) and len(Y) == 2
    d = tax.dot(X, X)                                  # x0*x0 + x1*x1 -> scalar Tracer
    assert isinstance(d, Tracer)

def test_array_tracer_elementwise_and_math():
    b = TraceBuilder()
    s = Isotropic(3, 2)
    X = ArrayTracer([Tracer(b, b.add(Var(i, s)), s) for i in range(2)], s)
    S = tax.sin(X)                                     # elementwise -> ArrayTracer
    assert isinstance(S, ArrayTracer) and len(S) == 2
    Z = 2.0 * X + X                                    # broadcast scalar + elementwise add
    assert isinstance(Z, ArrayTracer) and len(Z) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_trace.py -k tracer -v`
Expected: FAIL — `ImportError: cannot import name 'ArrayTracer'`.

- [ ] **Step 3: Implement `ArrayTracer` + trace branches**

```python
# add to python/tax/_frontend/trace.py
class ArrayTracer:
    """A symbolic stand-in for an Array during tracing: a list of scalar Tracers."""
    __slots__ = ("rows", "scheme")

    def __init__(self, rows, scheme):
        self.rows = list(rows)
        self.scheme = scheme

    def __len__(self):
        return len(self.rows)

    def _rows(self):
        return self.rows

    def __getitem__(self, i):
        if isinstance(i, slice):
            return ArrayTracer(self.rows[i], self.scheme)
        return self.rows[i]

    def _map_unary(self, op):
        return ArrayTracer([trace_unary(op, r) for r in self.rows], self.scheme)

    def _map_binary(self, op, other):
        if isinstance(other, ArrayTracer):
            if len(other) != len(self):
                raise ValueError(f"ArrayTracer length mismatch: {len(self)} vs {len(other)}")
            return ArrayTracer([trace_binary(op, a, b) for a, b in zip(self.rows, other.rows)],
                               self.scheme)
        return ArrayTracer([trace_binary(op, a, other) for a in self.rows], self.scheme)

    def __add__(self, o): return self._map_binary("add", o)
    def __radd__(self, o): return self._map_binary("add", o)
    def __sub__(self, o): return self._map_binary("sub", o)
    def __rsub__(self, o): return ArrayTracer([trace_binary("sub", o, a) for a in self.rows], self.scheme)
    def __mul__(self, o): return self._map_binary("mul", o)
    def __rmul__(self, o): return self._map_binary("mul", o)
    def __truediv__(self, o): return self._map_binary("div", o)
    def __rtruediv__(self, o): return ArrayTracer([trace_binary("div", o, a) for a in self.rows], self.scheme)
    def __neg__(self): return self._map_unary("neg")
    def __pow__(self, p): return self._map_binary("pow", p)
```

```python
# python/tax/_frontend/mathfns.py — recognise ArrayTracer as a vector in _make_unary:
def _make_unary(opcode):
    def fn(x):
        from .array import Array
        from .trace import ArrayTracer
        if isinstance(x, (Array, ArrayTracer)):
            return x._map_unary(opcode)
        return unary(opcode, x)           # Expansion or Tracer
    fn.__name__ = opcode
    return fn
# and in pow()/atan2(): dispatch (Array, ArrayTracer) -> _map_binary; else binary.
def pow(x, y):
    from .array import Array
    from .trace import ArrayTracer
    if isinstance(x, (Array, ArrayTracer)):
        return x._map_binary("pow", y)
    return binary("pow", x, y)
def atan2(y, x):
    from .array import Array
    from .trace import ArrayTracer
    if isinstance(y, (Array, ArrayTracer)):
        return y._map_binary("atan2", x)
    return binary("atan2", y, x)
```

```python
# python/tax/_frontend/array.py — add a trace branch at the TOP of concatenate() and cross().
# concatenate(): if any item is a Tracer/ArrayTracer, group symbolically (all share scheme G):
def concatenate(items):
    from .trace import Tracer, ArrayTracer
    if any(isinstance(it, (Tracer, ArrayTracer)) for it in items):
        rows = []
        for it in items:
            if isinstance(it, ArrayTracer):
                rows.extend(it._rows())
            else:
                rows.append(it)             # a scalar Tracer
        return ArrayTracer(rows, rows[0].scheme)
    # ... existing eager body unchanged ...

# cross(): if operands are tracers, build the components symbolically and return an ArrayTracer.
def cross(a, b):
    from .trace import Tracer, ArrayTracer
    if isinstance(a, ArrayTracer) or isinstance(b, ArrayTracer):
        from . import eager
        ra, rb = a._rows(), b._rows()
        if len(ra) != len(rb) or len(ra) not in (2, 3):
            raise ValueError("cross requires two 2- or 3-vectors of equal length")
        mul = lambda x, y: eager.binary("mul", x, y)
        sub = lambda x, y: eager.binary("sub", x, y)
        if len(ra) == 2:
            return sub(mul(ra[0], rb[1]), mul(ra[1], rb[0]))     # scalar Tracer
        c0 = sub(mul(ra[1], rb[2]), mul(ra[2], rb[1]))
        c1 = sub(mul(ra[2], rb[0]), mul(ra[0], rb[2]))
        c2 = sub(mul(ra[0], rb[1]), mul(ra[1], rb[0]))
        return ArrayTracer([c0, c1, c2], c0.scheme)
    # ... existing eager body unchanged ...
```

(`dot`/`norm` need no change: they iterate `._rows()` and call `eager.binary`/`eager.unary`, which now dispatch on `Tracer`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_trace.py -v`
Expected: PASS. Full suite green (eager `concatenate`/`cross`/`mathfns` paths unchanged).

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/trace.py python/tax/_frontend/mathfns.py python/tax/_frontend/array.py python/tests/test_trace.py
git commit -m "feat(py): ArrayTracer + trace-aware concatenate/cross/mathfns"
```

---

### Task 3: Trace driver (`trace_function`)

**Files:**
- Modify: `python/tax/_frontend/trace.py`
- Test: `python/tests/test_trace.py`

**Interfaces:**
- Produces: `trace_function(fn, arg_specs) -> TraceResult` where `arg_specs` is a list of `("exp", scheme)` / `("arr", scheme, nrows)` / `("scalar", value)`. `TraceResult` (a frozen dataclass) carries `.graph: Graph`, `.global_scheme`, `.out_kind: str` (`"exp"`/`"arr"`), `.out_nrows: int`. Builds G = fold of `.union` over the exp/arr schemes, creates input `Var`s (all scheme G) in argument order, runs `fn` with `Tracer`/`ArrayTracer`/raw-scalar args, and collects the result's node(s) as `graph.outputs`.
- Consumes: `TraceBuilder`, `Tracer`, `ArrayTracer`, `ir.{Var, Graph}`.

- [ ] **Step 1: Write the failing test**

```python
# add to python/tests/test_trace.py
from tax._frontend.trace import trace_function

def test_trace_function_global_scheme_and_outputs():
    s = Isotropic(4, 4)
    specs = [("scalar", 0.0), ("arr", s, 4), ("scalar", 398600.4418)]
    def rhs(t, x, mu):
        r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
        return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])
    tr = trace_function(rhs, specs)
    assert tr.global_scheme == s                  # only the Array contributes axes
    assert tr.out_kind == "arr" and tr.out_nrows == 4
    assert len(tr.graph.outputs) == 4
    assert tr.graph.n_inputs == 4                 # 4 Var slots (the Array rows); scalars baked

def test_trace_function_named_union():
    from tax._frontend.scheme import Named, Axis
    xs = Named.of(4, [Axis("x", 4)])
    mus = Named.of(4, [Axis("mu", 1)])
    specs = [("arr", xs, 4), ("exp", mus)]
    def f(x, mu):
        return mu * x[0]
    tr = trace_function(f, specs)
    assert tr.global_scheme == Named.of(4, [Axis("mu", 1), Axis("x", 4)])   # union
    assert tr.out_kind == "exp"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_trace.py -k trace_function -v`
Expected: FAIL — `ImportError: cannot import name 'trace_function'`.

- [ ] **Step 3: Implement `trace_function`**

```python
# add to python/tax/_frontend/trace.py
from dataclasses import dataclass

@dataclass(frozen=True)
class TraceResult:
    graph: object
    global_scheme: object
    out_kind: str          # "exp" or "arr"
    out_nrows: int

def _global_scheme(arg_specs):
    schemes = [spec[1] for spec in arg_specs if spec[0] in ("exp", "arr")]
    if not schemes:
        raise ValueError("tax.jit: at least one Expansion/Array argument is required")
    g = schemes[0]
    for s in schemes[1:]:
        g = g.union(s)
    return g

def trace_function(fn, arg_specs) -> TraceResult:
    g = _global_scheme(arg_specs)
    builder = TraceBuilder()
    slot = 0
    call_args = []
    for spec in arg_specs:
        kind = spec[0]
        if kind == "exp":
            call_args.append(Tracer(builder, builder.add(Var(slot, g)), g)); slot += 1
        elif kind == "arr":
            nrows = spec[2]
            rows = [Tracer(builder, builder.add(Var(slot + r, g)), g) for r in range(nrows)]
            slot += nrows
            call_args.append(ArrayTracer(rows, g))
        elif kind == "scalar":
            call_args.append(spec[1])            # raw scalar, baked when used
        else:
            raise ValueError(f"unknown arg spec kind {kind!r}")
    result = fn(*call_args)
    if isinstance(result, ArrayTracer):
        outputs = [t.node for t in result._rows()]
        out_kind, out_nrows = "arr", len(result)
    elif isinstance(result, Tracer):
        outputs = [result.node]
        out_kind, out_nrows = "exp", 1
    else:
        raise TypeError("tax.jit: function must return an Expansion or Array (got "
                        f"{type(result).__name__})")
    graph = Graph(builder.nodes, outputs, slot)
    return TraceResult(graph, g, out_kind, out_nrows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_trace.py -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/trace.py python/tests/test_trace.py
git commit -m "feat(py): trace_function — build a fused graph in one global scheme"
```

---

### Task 4: `tax.jit` decorator (lazy trace-on-first-call)

**Files:**
- Create: `python/tax/_frontend/jit.py`
- Modify: `python/tax/__init__.py`
- Test: `python/tests/test_jit.py`

**Interfaces:**
- Produces: `jit(fn)` (decorator) → a callable that on each call classifies its args, memoizes a `TraceResult` by the input signature (per-arg `("exp", scheme)` / `("arr", scheme, nrows)` / `("scalar", value)`), and on a miss traces+caches; then embeds inputs into the global scheme, runs the fused kernel once, and reconstructs the output(s) as `Expansion`/`Array`.
- Produces: `_classify_args(args) -> list[spec]` and the marshaling helpers.
- Consumes: `trace_function`, `eager.run`, `eager._embed`, `Expansion`, `Array`.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_jit.py
import numpy as np
import tax
from tests._helpers import needs_toolchain

@needs_toolchain
def test_jit_matches_eager_scalar():
    x = tax.variable(0.0, order=5)
    @tax.jit
    def f(x):
        return tax.sin(x) * tax.exp(x)
    got = f(x)
    want = tax.sin(x) * tax.exp(x)                 # eager
    np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-12)

@needs_toolchain
def test_jit_matches_eager_vector_map():
    X = tax.variables([1.0, 2.0], order=4)
    @tax.jit
    def g(X):
        return tax.concatenate([X[0] * X[1], X[0] / X[1]])
    got = g(X)
    want = tax.concatenate([X[0] * X[1], X[0] / X[1]])
    assert isinstance(got, tax.Array) and len(got) == 2
    np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-12)
    np.testing.assert_allclose(got.jacobian(), want.jacobian(), atol=1e-12)

@needs_toolchain
def test_jit_retraces_on_new_signature_but_reuses_match(monkeypatch):
    from tax._frontend import trace as tracemod
    calls = {"n": 0}
    real = tracemod.trace_function
    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(tracemod, "trace_function", counting)
    import importlib
    from tax._frontend import jit as jitmod
    monkeypatch.setattr(jitmod, "trace_function", counting)

    x4 = tax.variable(0.0, order=4)
    @tax.jit
    def f(x):
        return tax.exp(x)
    f(x4); f(x4)                                    # second call -> memo hit, no re-trace
    assert calls["n"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_jit.py -v`
Expected: FAIL — `AttributeError: module 'tax' has no attribute 'jit'`.

- [ ] **Step 3: Implement `jit.py` and export it**

```python
# python/tax/_frontend/jit.py
from __future__ import annotations
import numpy as np
from .types import Expansion
from .array import Array
from .trace import trace_function
from . import eager

def _classify_args(args):
    specs = []
    for a in args:
        if isinstance(a, Array):
            specs.append(("arr", a.scheme, len(a)))
        elif isinstance(a, Expansion):
            specs.append(("exp", a.scheme))
        elif isinstance(a, (int, float)):
            specs.append(("scalar", float(a)))
        else:
            raise TypeError(f"tax.jit: unsupported argument type {type(a).__name__}")
    return specs

def _sig_key(specs):
    # Hashable signature: schemes are frozen dataclasses; scalars by value.
    return tuple(specs)

def jit(fn):
    cache: dict = {}

    def wrapper(*args):
        specs = _classify_args(args)
        key = _sig_key(specs)
        tr = cache.get(key)
        if tr is None:
            tr = trace_function(fn, specs)
            cache[key] = tr
        g = tr.global_scheme
        in_buffers = []
        for a, spec in zip(args, specs):
            if spec[0] == "scalar":
                continue                            # baked
            if spec[0] == "exp":
                in_buffers.append(eager._embed(a, g))
            else:                                   # "arr"
                in_buffers.extend(eager._embed(row, g) for row in a._rows())
        out_sizes = [g.n_coeff] * tr.out_nrows
        outs = eager.run(tr.graph, in_buffers, out_sizes)
        if tr.out_kind == "exp":
            return Expansion(outs[0], g)
        return Array(np.stack(outs), g)

    wrapper.__name__ = getattr(fn, "__name__", "jitted")
    wrapper._tax_jit = True
    return wrapper
```

```python
# add to python/tax/__init__.py
from ._frontend.jit import jit
__all__ += ["jit"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_jit.py -v`
Expected: PASS (jit numerics match eager; the memo avoids re-tracing). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/jit.py python/tax/__init__.py python/tests/test_jit.py
git commit -m "feat(py): tax.jit lazy decorator (trace-on-first-call, fused kernel)"
```

---

### Task 5: Integer-`pow` lowering (`powint:{n}`)

**Files:**
- Modify: `python/tax/_codegen/emit_cpp.py`
- Modify: `python/tax/_frontend/types.py`
- Modify: `python/tax/_frontend/array.py`
- Modify: `python/tax/_frontend/mathfns.py`
- Test: `python/tests/test_pow_int.py`

**Interfaces:**
- Produces: a parameterized unary opcode `f"powint:{n}"` (n an int). `emit` lowers it to the C++ integer-power overload `pow(operand, n)` (= `tax::seriesPowInt`, correct for any base). `Expansion.__pow__`/`Array.__pow__`/`mathfns.pow` route an **int** exponent to `unary(f"powint:{n}", base)` (works in both eager and trace), and a **float** exponent to the existing `binary("pow", base, p)`.
- Consumes: `eager.unary` (already Tracer-polymorphic), `CPP_EXPR`.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_pow_int.py
import numpy as np
import tax
from tests._helpers import needs_toolchain

@needs_toolchain
def test_pow_int_negative_base_eager():
    x = tax.variable(0.0, order=3)
    f = (x - 2.0) ** 2                              # (-2 + dx)^2 = 4 - 4 dx + dx^2
    np.testing.assert_allclose(f.numpy(), np.array([4.0, -4.0, 1.0, 0.0]), atol=1e-12)

@needs_toolchain
def test_pow_int_matches_under_jit():
    x = tax.variable(0.0, order=3)
    @tax.jit
    def f(x):
        return (x - 2.0) ** 2
    np.testing.assert_allclose(f(x).numpy(), np.array([4.0, -4.0, 1.0, 0.0]), atol=1e-12)

@needs_toolchain
def test_pow_real_nonpositive_still_guarded():
    import pytest
    x = tax.variable(0.0, order=3)
    with pytest.raises(ValueError):
        (x - 2.0) ** 1.5                            # real exponent, base < 0 -> still raises
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_pow_int.py -v`
Expected: FAIL — `(x-2)**2` currently routes to `binary("pow", …)` and the M3 guard raises `ValueError` on the negative base (it should now compute correctly).

- [ ] **Step 3: Add the `powint` opcode and route ints to it**

```python
# python/tax/_codegen/emit_cpp.py — handle the powint:{n} opcode in emit()'s Op branch.
# Replace the Op handling line with:
        elif isinstance(node, Op):
            if node.opcode.startswith("powint:"):
                n = int(node.opcode.split(":", 1)[1])
                expr = f"pow(n{node.operands[0]}, {n})"     # C++ pow(TE, int) = seriesPowInt
            else:
                expr = CPP_EXPR[node.opcode].format(*[f"n{o}" for o in node.operands])
            lines.append(f"    auto n{i} = {expr};")
```

```python
# python/tax/_frontend/types.py — Expansion.__pow__ routes int -> powint:
    def __pow__(self, p):
        from .eager import unary, binary
        if isinstance(p, int) and not isinstance(p, bool):
            return unary(f"powint:{p}", self)
        return binary("pow", self, p)
```

```python
# python/tax/_frontend/array.py — Array.__pow__ routes int -> powint per row:
    def __pow__(self, p):
        if isinstance(p, int) and not isinstance(p, bool):
            return self._map_unary(f"powint:{p}")
        return self._map_binary("pow", p)
```

```python
# python/tax/_frontend/mathfns.py — pow() routes int -> powint (covers Array/ArrayTracer too):
def pow(x, y):
    from .array import Array
    from .trace import ArrayTracer
    if isinstance(y, int) and not isinstance(y, bool):
        if isinstance(x, (Array, ArrayTracer)):
            return x._map_unary(f"powint:{y}")
        return unary(f"powint:{y}", x)
    if isinstance(x, (Array, ArrayTracer)):
        return x._map_binary("pow", y)
    return binary("pow", x, y)
```

(`ArrayTracer.__pow__` already routes an int through `_map_binary("pow", p)` from Task 2 — update it to mirror `Array.__pow__`: `if isinstance(p, int) and not isinstance(p, bool): return self._map_unary(f"powint:{p}")` else `self._map_binary("pow", p)`.)

```python
# python/tax/_frontend/trace.py — ArrayTracer.__pow__:
    def __pow__(self, p):
        if isinstance(p, int) and not isinstance(p, bool):
            return self._map_unary(f"powint:{p}")
        return self._map_binary("pow", p)
```

The M3 eager `pow` guard in `eager.binary` (`if opcode == "pow" and ea.value() <= 0.0: raise`) stays — it now only fires for **float** exponents (which genuinely need a positive base), since int exponents no longer reach `binary("pow", …)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_pow_int.py -v`
Expected: PASS (int powers correct for a negative base, eager and jit; the float-exponent guard still raises). Full suite green — confirm the existing pow tests (M1 `tax.pow(b,1.5)`, M3 `**` with positive base) still pass: `b ** 1.5` with positive base routes to `binary("pow",…)` unchanged; `x ** 2` now routes to `powint`.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_codegen/emit_cpp.py python/tax/_frontend/types.py python/tax/_frontend/array.py python/tax/_frontend/mathfns.py python/tax/_frontend/trace.py python/tests/test_pow_int.py
git commit -m "feat(py): integer-pow lowering (powint opcode) — correct for any base, eager+jit"
```

---

### Task 6: numba-style signatures + `dump`

**Files:**
- Modify: `python/tax/_frontend/jit.py`
- Modify: `python/tax/__init__.py`
- Test: `python/tests/test_jit.py`

**Interfaces:**
- Produces: type-spec constructors `tax.f64` (a sentinel for a runtime scalar — for M4, a scalar spec with a placeholder value), `tax.ExpansionType(order, name=None)`, `tax.ArrayType(order, size, name=None)` — each lowering to the same `("exp"/"arr"/"scalar", …)` spec the lazy path produces. `jit(fn, signature=None, dump=False)`: with `signature` (a list of specs), trace+compile **at decoration** (eager) and validate calls against it; `dump=True` also returns/records the generated C++ for inspection. With no signature, the lazy path (Task 4) applies. `@tax.jit([...])` (list as the sole arg) is sugar for `signature=[...]`.
- Consumes: the Task 4 machinery; `Isotropic`/`Named`/`Axis` for spec→scheme.

- [ ] **Step 1: Write the failing test**

```python
# add to python/tests/test_jit.py
@needs_toolchain
def test_jit_explicit_signature_compiles_and_matches_lazy():
    sig = [tax.ArrayType(order=4, size=2)]
    @tax.jit(sig)
    def g(X):
        return tax.concatenate([X[0] * X[1], X[0] + X[1]])
    X = tax.variables([1.0, 2.0], order=4)
    got = g(X)
    want = tax.concatenate([X[0] * X[1], X[0] + X[1]])
    np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-12)

@needs_toolchain
def test_jit_dump_returns_source():
    @tax.jit(dump=True)
    def f(x):
        return tax.sin(x)
    x = tax.variable(0.0, order=4)
    f(x)
    assert "tax_kernel" in f.dump_source()         # the generated TU is retrievable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && .venv/bin/python -m pytest tests/test_jit.py -k "signature or dump" -v`
Expected: FAIL — `AttributeError: module 'tax' has no attribute 'ArrayType'`.

- [ ] **Step 3: Add type specs, signatures, and dump**

```python
# python/tax/_frontend/jit.py — add spec constructors and extend jit().
from dataclasses import dataclass
from .scheme import Isotropic, Named, Axis
from .._codegen.emit_cpp import emit

@dataclass(frozen=True)
class _ScalarSpec:
    pass
f64 = _ScalarSpec()

@dataclass(frozen=True)
class ExpansionType:
    order: int
    name: object = None
    def _spec(self):
        sch = Named.of(self.order, [Axis(self.name, 1)]) if self.name else Isotropic(self.order, 1)
        return ("exp", sch)

@dataclass(frozen=True)
class ArrayType:
    order: int
    size: int
    name: object = None
    def _spec(self):
        sch = (Named.of(self.order, [Axis(self.name, self.size)]) if self.name
               else Isotropic(self.order, self.size))
        return ("arr", sch, self.size)

def _spec_of_type(t):
    if isinstance(t, (ExpansionType, ArrayType)):
        return t._spec()
    if isinstance(t, _ScalarSpec):
        return ("scalar", 0.0)            # placeholder; M4 bakes scalars, value supplied at call
    raise TypeError(f"tax.jit signature: unsupported type spec {t!r}")

def jit(fn=None, signature=None, dump=False):
    # Support @tax.jit, @tax.jit([...]), @tax.jit(signature=[...], dump=True)
    if isinstance(fn, list):               # @tax.jit([...])
        signature, fn = fn, None
    if fn is None:
        return lambda f: _build(f, signature, dump)
    return _build(fn, signature, dump)

def _build(fn, signature, dump):
    cache: dict = {}
    state = {"last_source": None}

    def _trace_and_record(specs):
        tr = trace_function(fn, specs)
        if dump:
            state["last_source"] = emit(tr.graph)
        return tr

    if signature is not None:
        sig_specs = [_spec_of_type(t) for t in signature]
        cache[_sig_key(sig_specs)] = _trace_and_record(sig_specs)   # eager trace at decoration

    def wrapper(*args):
        specs = _classify_args(args)
        if signature is not None:
            want = [_spec_of_type(t) for t in signature]
            # Validate schemes match the pinned signature (scalars compare loosely).
            for got, exp in zip(specs, want):
                if got[0] != exp[0] or (got[0] != "scalar" and got[1] != exp[1]):
                    raise TypeError(f"tax.jit signature mismatch: {got} vs declared {exp}")
        key = _sig_key(specs)
        tr = cache.get(key)
        if tr is None:
            tr = _trace_and_record(specs)
            cache[key] = tr
        g = tr.global_scheme
        in_buffers = []
        for a, spec in zip(args, specs):
            if spec[0] == "scalar":
                continue
            if spec[0] == "exp":
                in_buffers.append(eager._embed(a, g))
            else:
                in_buffers.extend(eager._embed(row, g) for row in a._rows())
        outs = eager.run(tr.graph, in_buffers, [g.n_coeff] * tr.out_nrows)
        return Expansion(outs[0], g) if tr.out_kind == "exp" else Array(np.stack(outs), g)

    wrapper.__name__ = getattr(fn, "__name__", "jitted")
    wrapper._tax_jit = True
    wrapper.dump_source = lambda: state["last_source"]
    return wrapper
```

(This replaces the Task 4 `jit`/`wrapper`; the lazy behavior is preserved when `signature is None`.)

```python
# add to python/tax/__init__.py
from ._frontend.jit import f64, ExpansionType, ArrayType
__all__ += ["f64", "ExpansionType", "ArrayType"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && .venv/bin/python -m pytest tests/test_jit.py -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add python/tax/_frontend/jit.py python/tax/__init__.py python/tests/test_jit.py
git commit -m "feat(py): numba-style jit signatures (eager decoration-time compile) + dump"
```

---

### Task 7: M4 gate — both two-body RHS maps under `@tax.jit`

**Files:**
- Test: `python/tests/test_m4_gate.py`

**Interfaces:**
- Consumes: the full jit surface (`@tax.jit` bare + `@tax.jit(signature)`, `concatenate`, `**`, `Array.value`/`jacobian`).
- Produces: the M4 acceptance gate — the unnamed and named two-body RHS maps under jit, with jit value/Jacobian matching the eager result (and the named map also under a pinned signature).

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_m4_gate.py
import numpy as np
import tax
from tests._helpers import needs_toolchain

MU = 398600.4418

def _two_body(x, mu):
    r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
    return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])

@needs_toolchain
def test_jit_unnamed_two_body_matches_eager():
    x = tax.variables([1.0, 0.0, 0.0, 1.0], order=4)     # isotropic, mu baked as float

    @tax.jit
    def rhs(t, x, mu):
        return _two_body(x, mu)

    dx = rhs(0.0, x, MU)
    want = _two_body(x, MU)                               # eager
    np.testing.assert_allclose(dx.value(), np.array([0.0, 1.0, -MU, 0.0]), rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(dx.numpy(), want.numpy(), rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(dx.jacobian(), want.jacobian(), rtol=1e-9, atol=1e-6)

@needs_toolchain
def test_jit_named_two_body_bare_and_signature():
    from tax._frontend.scheme import Named, Axis
    x = tax.variables([1.0, 0.0, 0.0, 1.0], order=4, name="x")
    mu = tax.variable(MU, order=4, name="mu")

    @tax.jit
    def rhs_bare(t, x, mu):
        return _two_body(x, mu)

    dx = rhs_bare(0.0, x, mu)
    want = _two_body(x, mu)                               # eager
    assert dx.scheme == Named.of(4, [Axis("mu", 1), Axis("x", 4)])
    np.testing.assert_allclose(dx.numpy(), want.numpy(), rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(dx.jacobian("x"), want.jacobian("x"), rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(dx.jacobian("mu"), want.jacobian("mu"), rtol=1e-9, atol=1e-9)

    # same RHS, pinned with an explicit numba-style signature (compiles at decoration)
    @tax.jit([tax.f64, tax.ArrayType(order=4, size=4, name="x"),
              tax.ExpansionType(order=4, name="mu")])
    def rhs_sig(t, x, mu):
        return _two_body(x, mu)

    dx2 = rhs_sig(0.0, x, mu)
    np.testing.assert_allclose(dx2.numpy(), want.numpy(), rtol=1e-9, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails (or passes immediately)**

Run: `cd python && .venv/bin/python -m pytest tests/test_m4_gate.py -v`
Expected: PASS if Tasks 1–6 are complete (an integration gate). If it fails, debug the specific stage — most likely the global-scheme embed (named `{x}`/`{mu}` → `{mu,x}`) in the jit marshaling, or the `** 1.5` (which routes to `binary("pow", …)` — base `rx²+ry²=1 > 0`, so it must not hit the guard).

- [ ] **Step 3: (No new implementation expected)**

If jit output differs from eager, confirm the global scheme G equals eager's result scheme for these maps (it does — every input interacts, so G = `{mu,x}` named / `{x}` unnamed). If the named Jacobian columns are off, confirm `eager._embed` lifts each input into G before the call (jit marshaling) exactly as eager does per-op.

- [ ] **Step 4: Run the whole suite — the M4 gate**

Run: `cd python && .venv/bin/python -m pytest -v`
Expected: PASS (all M1–M4 tests).

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_m4_gate.py
git commit -m "test(py): M4 gate — unnamed + named two-body RHS under @tax.jit (bare + signature)"
```

---

## Self-Review

**Spec coverage (M4 = `tax.jit` fusion — design §6.3/§6.8, the spec's `@tax.jit` examples, and the M-roadmap):**
- Whole-function trace → one fused kernel (no per-op FFI / intermediates) → Tasks 1–4. ✓
- Trace-on-first-call (lazy), signature-memoized → Task 4. ✓
- numba-style explicit signatures (eager decoration-time compile) + `f64`/`ExpansionType`/`ArrayType` → Task 6. ✓
- Multi-input / multi-output maps (vector RHS) → Tasks 2–4 (ArrayTracer + multi-output graph; emit/run already general). ✓
- Integer-`pow` lowering (tracked from M1/M3 — `x ** <int>` correct for any base) → Task 5. ✓
- `dump` option → Task 6. ✓
- Both two-body RHS maps under jit (bare + signature) → Task 7. ✓
- Reuses the M1 backend (`emit`/`run`/`call_kernel`/cache) verbatim — verified general (multi-node/in/out). ✓
- Out of scope (documented): runtime-scalar kernel params (scalars baked); `batch=K`/`scalar=float32`/`march_native`/`static_argnums` option plumbing (only `signature`/`dump` wired); MLIR backend; the global-scheme over-promotion for functions that ignore an input (documented — does not affect the two-body, where G = eager's scheme).

**Placeholder scan:** No "TBD"/"similar to Task N"/"add error handling". Every step has complete code and concrete oracles (jit == eager via assert_allclose; integer-pow `(-2+dx)²=[4,-4,1,0]`; two-body value/Jacobian reused from the M2/M3 gates).

**Type consistency:** `TraceBuilder`/`Tracer`/`ArrayTracer`/`trace_function`/`TraceResult` (`.graph`/`.global_scheme`/`.out_kind`/`.out_nrows`), `eager.unary`/`binary` (Tracer-polymorphic), `jit`/`_classify_args`/`_sig_key`/`f64`/`ExpansionType`/`ArrayType`, and the `powint:{n}` opcode are used with identical names/signatures across producing and consuming tasks. The jit driver consumes `eager.run`/`eager._embed` (M1/M3, unchanged) and `Expansion`/`Array` (M1/M2). `emit`/`run`/`call_kernel` are reused without signature changes (confirmed general).

---

## Roadmap (after M4)

- **M5 — Targets + regression + perf:** both two-body maps as documented examples + end-to-end tests; DACE/C++ accuracy regression for the jit path; eager-vs-jit-vs-hand-written-C++ benchmarks (quantify the fusion win and the first-touch compile cost). Optionally `batch=K` (lock-step multi-point via `Batch<double,K>`).
- **M6 — Packaging:** vendored `tax`/Eigen headers in a `py3-none-any` wheel, `cffi` (API mode) FFI upgrade, a shipped precompiled header for warm builds, compiler-discovery docs, and the worked two-body examples.
```
