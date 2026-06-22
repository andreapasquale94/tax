# JIT-compiled Python layer for static Taylor expansions — design

**Date:** 2026-06-22
**Status:** Approved for planning (brainstorming complete)
**Topic:** A Python front-end for `tax` that lets the user set the expansion **order**, **size**, and **axis names** at runtime while every computation still runs on the library's **static `std::array` dense storage** — achieved by **JIT-compiling generated C++ that reuses the existing header kernels** and caching the resulting shared objects. Two execution modes share one pipeline: **eager** (DACE-like; each op is a cached kernel) and **`tax.jit`** (JAX-like; a whole function is traced and fused into one kernel). Exposes a **single pair of Python types** (`Expansion`, `Array`) — no per-`(N,M)` Python type explosion.

## Motivation

`tax`'s dense type is `TaylorExpansion<T, Scheme, storage::Dense>`, whose storage is `std::array<T, Scheme::nCoeff>` with `Scheme::nCoeff = numMonomials(N, M)` a **compile-time constant**. Order, variable count, and (for named/mixed schemes) the axis layout are all baked into the C++ type. That is exactly what makes the dense path fast and allocation-free — and exactly what makes it hard to expose to Python, where the user wants to choose order and size at runtime.

The only ways to reconcile "runtime-chosen scheme" with "compile-time-sized storage" are: (a) pre-instantiate a bounded grid of `(N,M)` at build time, (b) abandon static storage for a runtime/heap representation, or (c) **compile the specific instantiation on demand** — a JIT. Prior exploration on three branches mapped the first two and one extreme of the third; none occupies the point this design targets:

| Prior branch | One Python type | Runtime scheme | Static `std::array` | JIT | Reuses header lib |
|---|---|---|---|---|---|
| `claude/nanobind-tax-bindings` | ✓ `Sparse` | ✓ | ✗ (heap sparse, runtime kernels) | ✗ | ✗ (new dynamic type) |
| `claude/add-python-bindings` (pybind11) | ✓ `TE` | ✓ but **bounded** (N≤20, M≤10) | ✓ | ✗ (**pre-instantiated grid**) | ✓ |
| `claude/taylor-polynomial-mlir` | ✗ (C++ only) | per-function | dense memref | ✓ (LLVM/MLIR) | ✗ (**replaces** lib) |

The pybind11 grid branch already proves "one Python type + static dense storage + reuse the header library," but pays for runtime `(N,M)` by pre-instantiating *every* `(N,M)` at build time — the fat-binary / bounded-grid / slow-compile cost this design exists to escape. This design is that branch's properties **with a JIT replacing the pre-built grid**, so only the schemes actually used are compiled, the grid is unbounded, and the base wheel stays small. The MLIR branch sits at the opposite extreme — a custom compiler that discards the templates; we instead **reuse 100% of the kernels** and let the C++ optimizer do the fusion.

## Scope & decisions (approved during brainstorming)

1. **Codegen vehicle = trace → C++ source → compile.** The traced computation is emitted as one straight-line C++ translation unit calling the **existing** `tax` operators, compiled at `-O3` as a single TU. "Fusion" is whatever the C++ optimizer does given the whole expression in one TU: inline the recurrences, promote `std::array`s to registers/stack (SROA/mem2reg), drop intermediates. **No kernel is reimplemented.** (Chosen over hand-emitting LLVM IR or finishing the MLIR dialect, both of which reimplement every recurrence and add an LLVM/MLIR runtime dependency.)
2. **Two modes, one mechanism.** Eager and `tax.jit` are the *same* trace→emit-C++→compile→cache pipeline at different granularities — **per-operation** (eager) vs **per-function** (jit). Eager is the default DACE-like experience and needs no `jit`; `tax.jit` is an opt-in accelerator that removes per-op FFI crossings and intermediate buffers for hot loops.
3. **Single Python type pair.** `Expansion` (one truncated series) and `Array` (a vector of expansions over a shared scheme). Both are thin handles over a coefficient buffer + a runtime **scheme descriptor**; there is no per-`(N,M)` Python type.
4. **Schemes in scope from v1:** isotropic `(order, size)`, **named** axes (`name="x"`), **integer-indexed** sized vectors, and **mixed / per-axis order** (`order=` per variable group). These map onto C++ factories that already exist on this branch — `tax::variable<"x",N>`, `tax::variables<N,M>(point)` (commit `05aa412`), and the `MixedScheme` / `MixedTaylorExpansion` named per-axis-order layer.
5. **Static storage is preserved end to end.** Codegen instantiates a *fixed* scheme, so every kernel computes in `std::array`-backed `TaylorExpansion`; the FFI boundary is raw `double*` coefficient buffers. No dynamic-order C++ type ever exists.
6. **Vectors are first-class.** `tax.concatenate`, indexing/slicing, elementwise math, `dot`/`matmul`, and `jacobian`/`hessian` of a vector — eager **and** inside `tax.jit` (multi-input / multi-output maps).
7. **Thin, compiler-only deployment.** The calling path uses `ctypes`/`cffi` — **no compiled extension module** on the Python side. The base package is pure-Python plus vendored headers; the only runtime requirement is a **C++23 compiler** discoverable at runtime.

### Non-goals (v1)

- **ODE integration / time-stepping.** We JIT the *map* (e.g. a flow-map RHS); integrating it is the companion plugin's job or the user's loop. The two-body examples below define and evaluate the RHS map, not the integrator.
- **Sparse storage** through the JIT (dense only; the eager engine could fall back to the existing sparse C++ later).
- **GPU / SIMD-intrinsic codegen.** `batch=K` reuses the existing `Batch<double,K>` for lock-step multi-point evaluation; explicit vectorization is left to the C++ optimizer.
- **Autodiff *through* the JIT boundary** beyond what the Taylor coefficients already give (the coefficients *are* the derivatives).
- **Windows-first support.** Designed portably (POSIX `dlopen`/`ctypes` and MSVC are both reachable) but Linux/macOS are the v1 gate.

## Target examples (the north star)

Both must work; they drive the whole design.

**(1) Unnamed planar two-body RHS (state = [r₂, v₂], M=4), jit-compiled map:**

```python
import numpy as np, tax

x0 = np.array([1.0, 0.0, 0.0, 1.0])          # rx, ry, vx, vy
x  = tax.variables(x0, order=4, size=4)        # Array of 4 coordinate-variable Expansions

@tax.jit                                        # traced on first call, fused, cached
def rhs(t, x, mu):
    r3 = (x[0]*x[0] + x[1]*x[1]) ** 1.5
    return tax.concatenate([x[2], x[3],
                            -mu * x[0] / r3,
                            -mu * x[1] / r3])

dx = rhs(0.0, x, 398600.4418)                   # Array of 4 expansions (the RHS map)
dx.value(); dx.jacobian()                        # constants and state-transition block
```

**(2) Named two-body RHS, with a named parameter axis:**

```python
x  = tax.variables(x0, order=4, size=4, name="x")   # 4-dim named axis "x"
mu = tax.variable(398600.4418, order=4, name="mu")  # 1-dim named axis "mu"

@tax.jit
def rhs(t, x, mu):
    r3 = (x[0]*x[0] + x[1]*x[1]) ** 1.5
    return tax.concatenate([x[2], x[3], -mu*x[0]/r3, -mu*x[1]/r3])

dx = rhs(0.0, x, mu)        # expansion over the union of axes {mu, x} (M=5), order 4
dx.jacobian("x")           # ∂(rhs)/∂x by axis name; dx.jacobian("mu") for parameter sensitivity
```

The same `rhs` source works eagerly (drop `@tax.jit`) — each op dispatches to a cached per-scheme kernel.

## Architecture overview — one pipeline, two granularities

```
 Python op  or  @tax.jit fn
        │  trace
        ▼
   Graph IR  ──────────────┐  (nodes = ops, leaves = variable/const, edges = operands;
   (canonical, hashable)   │   carries the result Scheme; CSE-friendly)
        │  lower           │
        ▼                  │  cache key = sha256( ABI_VER ‖ lib_VER ‖ compiler_id
   one C++ TU  ────────────┤        ‖ flags ‖ scalar ‖ scheme_descriptor ‖ graph_canonical )
   (#include <tax/tax.hpp>,│
    straight-line, -O3)    │  hit ─────────────► load from on-disk cache
        │  compile         │
        ▼                  │
   .so  ──dlopen/ctypes──► call with numpy coefficient buffers ──► Expansion / Array
```

- **Eager** traces a *single op* (a trivial graph): operand schemes → result scheme → a kernel that embeds operands into the result scheme and applies the op. Compiled once per `(op, operand-schemes)`, reused forever.
- **`tax.jit`** traces a *whole function* (a large graph). One TU, one compile, one call; the optimizer fuses across ops and eliminates intermediates.
- The **Graph IR** is the hub: it is hashed for the cache and lowered to C++. The same lowering serves both modes — eager is just the degenerate single-node case.

## Components & interfaces

Each component has one purpose, a narrow interface, and is testable in isolation.

### 6.1 Python surface types (`tax/_frontend/`)

- **`Expansion`** — handle over `(coeffs: np.ndarray[float64], scheme: Scheme)`. Methods mirror the C++ surface: `value()`, `coeff(...)`/`coeff(x=…, y=…)`, `derivative(...)`, `eval(dx)`, `deriv(name|idx)`, `integ(...)`, `truncate(...)`, `gradient()`, `hessian()`, `__add__`/`__mul__`/… `numpy()`.
- **`Array`** — handle over `(coeffs: np.ndarray[float64] shape (K, nCoeff), scheme: Scheme)` (one shared scheme). `__getitem__` (→ `Expansion` or `Array`), elementwise arithmetic/math (broadcast a scalar `Expansion` over the vector), `dot`/`matmul`, `value()` (→ `(K,)` array), `eval(dx)`, `jacobian(axis=None)`, `hessian(...)`, `numpy()`.
- **Factories** (mirror the C++ factories one-to-one):
  - `tax.variable(x0, order=, name=None)` — one variable; `name=` → named axis, else an isotropic univariate.
  - `tax.variables(x0_array, order=, size=None, name=None)` — vector of coordinate variables; `name=` → one *n-dim named axis*, else `size` integer-indexed isotropic axes. `size` defaults to `len(x0_array)`.
  - Per-variable `order=` is allowed (→ mixed scheme); a scalar `order=` applies to all.
  - `tax.constant(v, order=, ...)`, `tax.zeros(...)`.
- **Free functions** — the full math surface (`sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, + inverses, exp, log, sqrt, cbrt, square, cube, erf, pow, atan2, reciprocal`) plus `tax.concatenate`, `tax.stack`, `tax.dot`. Each is overloaded to (a) record a node when given a tracer, (b) dispatch an eager kernel when given a concrete handle.

**Interface contract:** the surface types never touch C++ directly — they emit graph nodes (jit/tracing) or call the **Eager engine** (6.7), which is the only component that knows about kernels.

### 6.2 Scheme descriptor (`tax/_frontend/scheme.py`)

A small, hashable, canonical Python value describing the C++ scheme to instantiate. One of:

- `Isotropic(order:int, vars:int)` → C++ `IsotropicScheme<order, vars>`.
- `Mixed(groups: tuple[Group(dim, order), ...], joint_cap:int|None)` → `MixedScheme<Group<dim,order>…>`.
- `Named(axes: tuple[Axis(name, dim, order), ...])` → the `MixedTaylorExpansion` named layer; **canonicalized sorted-by-name and unique** so `x*p` and `p*x` produce the same descriptor (matching the C++ canonical type).

Responsibilities: `nCoeff`, `vars`, `flat_layout` (for `coeff(...)` lookups and `numpy()` ordering), `union(other)` (axis-union + max-order-per-shared-axis promotion — the Python mirror of `embedMixed`), `cpp_type_string()`, and a stable `descriptor_hash()` for the cache key. This is the single source of truth for "what C++ type does this map to," shared by eager and jit.

### 6.3 Tracer & Graph IR (`tax/_frontend/ir.py`, `trace.py`)

- **IR nodes:** `Var(slot, kind, scheme)`, `Const(value, scheme)`, `Op(opcode, operands, result_scheme)`, `Output(operands)`. `kind ∈ {coordinate(i), general}` lets a `Var` be either a coordinate variable seeded from a scalar point (kernel builds it internally) or an arbitrary input passed as a full coefficient buffer (enables composition / feeding pre-built expansions).
- **Tracer object:** a lightweight value that overloads the operator/math surface to append `Op` nodes. `tax.jit` runs the user function once with tracer arguments (trace-on-first-call, JAX-style); eager builds a one-node graph per op.
- **Canonicalization:** topological order with de Bruijn-style numbering, opcode + operand slots + constant bit-patterns; structurally identical graphs hash equal (drives cache hits and free CSE). Multi-output handled by an `Output` node listing the returned expansions/arrays.

**Interface contract:** the tracer emits IR only; it knows nothing about C++ or files.

### 6.4 C++ code generator (`tax/_codegen/emit_cpp.py`)

Lowers an IR graph + scheme to a single TU:

```cpp
#include <tax/tax.hpp>
using namespace tax;
extern "C" int tax_kernel(const double* const* ins, double* const* outs) noexcept {
    using E = TaylorExpansion<double, /* scheme_descriptor.cpp_type_string() */>;
    // seed inputs: coordinate Vars from a scalar point, general Vars copied from ins[k]
    E v0 = E::variable<0>({ ins[0][0], ins[0][1], ins[0][2], ins[0][3] });
    // ... one `auto tN = <op>(args);` line per IR Op (straight-line; -O3 fuses) ...
    auto r0 = sin( t7 ) * t3;            // example
    std::copy_n(r0.coefficients().data(), E::nCoefficients, outs[0]);
    return 0;                            // nonzero reserved for trapped domain errors
}
```

- **One C++ statement per IR Op**, in topological order; the C++ compiler inlines/fuses. No hand-written recurrence.
- **Named/mixed schemes** emit the corresponding `MixedTaylorExpansion<…>` / `MixedScheme<…>` type and the matching `tax::variable<"name",Dim>` / `tax::variables` seeding; cross-axis operands are embedded into the union scheme by the same `embedMixed`-backed promotion the C++ binary operators already perform — so the generator emits the natural expression and the library handles promotion.
- **Output marshaling:** `std::copy_n` each returned expansion's `coefficients()` into the matching `outs[k]` buffer (graded-lex order, the scheme's canonical layout). `Array` outputs write `K` consecutive rows.
- Emits a **sidecar JSON** (scheme, ABI version, input/output shapes, opcode list) next to the TU for debugging and `dump=True`.

### 6.5 Compile & cache manager (`tax/_codegen/build.py`)

- **Compiler discovery:** `TAX_CXX` → `CXX` → first of `c++`/`clang++`/`g++` on `PATH`; cache the resolved id+version. Require `-std=c++23`; assemble flags `[-O3, -shared, -fPIC, -I<vendored include>, <opt flags>]`.
- **Cache key** (see §9): `sha256(ABI_VERSION ‖ tax_version ‖ compiler_id ‖ flags ‖ scalar ‖ scheme.descriptor_hash() ‖ graph_canonical)`. Cache dir defaults to `platformdirs.user_cache_dir("tax")`, overridable (`TAX_CACHE_DIR`, `cache_dir=`).
- **Build = miss path:** write TU to a temp dir, compile to `.<key>.so`, **atomic-rename** into the cache (concurrency-safe), record a small manifest. A **file lock** per key serializes concurrent first-builds (multiprocessing-safe).
- **Warm-build optimizations:** ship/auto-build a **precompiled header** for `<tax/tax.hpp>` to cut per-kernel compile latency; minimize includes when `la` features (gradient/jacobian) are unused.

### 6.6 Loader / FFI (`tax/_codegen/load.py`)

`ctypes.CDLL(so_path)`, resolve `tax_kernel`, set `argtypes`/`restype`. Call passes arrays of `double*` (from contiguous, C-order `float64` numpy buffers) for `ins`/`outs`; the wrapper allocates output buffers sized `K*nCoeff` from the scheme and rebuilds `Expansion`/`Array`. No per-op binding code — the only native code is the JIT'd kernel.

### 6.7 Eager engine (`tax/_frontend/eager.py`)

Given an op and concrete operand handles: compute the **result scheme** (`scheme.union(...)` for mixed/named operands; promotion for differing orders), build the one-node IR graph, obtain the cached kernel via 6.4–6.6, call it, wrap the result. A small in-process LRU memoizes `(opcode, operand-schemes) → loaded kernel` so steady-state eager is a dict lookup + one FFI call.

### 6.8 `tax.jit` decorator (`tax/_frontend/jit.py`)

- **Trace-on-first-call:** on first call, read each argument's kind — `Expansion`/`Array` → traced input `Var` (kind `general` or `coordinate`); plain Python scalar → runtime scalar input (or compile-time constant if listed in `static_argnums`). Build the graph, infer the result scheme(s), compile+cache, and memoize keyed by the **input signature** (per-arg scheme + staticness). Subsequent calls with a matching signature skip tracing.
- **Options:** `opt` (`"O2"|"O3"|"fast"`, `march_native: bool`, `fastmath: bool`), `cache`/`cache_dir`/`force_recompile`, `compiler`, `scalar` (`"float64"|"float32"`), `batch` (`K` → coefficients become `Batch<double,K>` for lock-step multi-point evaluation), `static_argnums`, `dump` (emit generated C++ + `.so` path). All optional with library-sensible defaults.
- **Multi-output maps:** a function returning a tuple/`Array` lowers to multiple `outs[k]`; this is the general buffer-in/buffer-out ABI, so composition and Jacobian-of-map need no signature change.

## 7. Kernel ABI

A single, versioned C ABI, shapes baked at codegen:

```c
int tax_kernel(const double* const* ins, double* const* outs) noexcept;
```

- `ins[k]` — the k-th input's coefficient buffer (`nCoeff` of its scheme), **or**, for a `coordinate(i)` input, a length-`vars` point used to seed `variable<i>` internally. Input kinds are fixed in the graph, so the wrapper knows each buffer's length.
- `outs[k]` — pre-allocated output buffer (`nCoeff`, or `K*nCoeff` for an `Array`), written in graded-lex order.
- Return value `0` on success; nonzero reserved for trapped domain errors (e.g. `log` of a non-positive constant) so the Python layer can raise rather than read NaNs silently.
- **`ABI_VERSION`** is part of the cache key; bumping it invalidates all cached `.so`s.

This ABI is intentionally the *general* (multi-in/multi-out, buffer-based) one, with `coordinate` inputs as an optimization — so "coordinate variables at a point" and "compose pre-built expansions / vector maps" are the same ABI.

## 8. End-to-end data flow (example 2, named)

1. `tax.variables(x0, order=4, size=4, name="x")` → `Array` over `Named(axes=[Axis("x",4,4)])`; `tax.variable(mu0, order=4, name="mu")` → `Expansion` over `Named(axes=[Axis("mu",1,4)])`.
2. `rhs(0.0, x, mu)` under `@tax.jit`: `t=0.0` is a runtime scalar input; `x`,`mu` become traced `Var`s. Tracing runs the body, recording `Op`s; cross-axis ops promote to the **union** scheme `Named([Axis("mu",1,4), Axis("x",4,4)])` (canonical sorted), `M=5`, `order 4`.
3. Graph canonicalized + hashed; cache key formed with the union scheme descriptor.
4. **Miss:** emit one TU instantiating `MixedTaylorExpansion<double, OrderedAxis<"mu",1,4>, OrderedAxis<"x",4,4>>` (or the `MixedScheme` equivalent), one statement per op, `std::copy_n` the 4 outputs; compile `-O3 -shared`; atomic-rename into cache.
5. Load via `ctypes`; call with `ins = [point_for_x, buffer_for_mu]`, `outs = [4 buffers]`.
6. Wrap the 4 output buffers as an `Array` over the union scheme. `dx.jacobian("x")` slices the linear coefficients of axis "x".
7. **Next call** with the same arg schemes → memoized signature → cache hit → just the FFI call.

## 9. Caching & cache keys

`key = sha256( ABI_VERSION ‖ tax_version ‖ compiler_id+version ‖ canonical(flags) ‖ scalar_type ‖ scheme.descriptor_hash() ‖ graph_canonical_form )`.

- **`tax_version`** ties cached `.so`s to the header library they were built against (read from `CMakeLists.txt` `project(... VERSION 0.1.0)` / a generated `_version.py`); a header change invalidates stale kernels.
- **`graph_canonical_form`** makes structurally identical computations (including post-CSE) share a `.so`.
- **Scheme** + **scalar** + **flags** capture every other axis of specialization.
- On-disk layout: `<cache>/<key>.so` + `<key>.json` (sidecar). A corrupt/missing entry is treated as a miss and rebuilt.

## 10. Static-storage guarantee

Because every emitted TU names a *fixed* scheme, the kernel's locals are `TaylorExpansion<double, Scheme>` with `std::array<double, Scheme::nCoeff>` storage — identical to hand-written C++. The dynamic part lives **only** in Python (which scheme to compile) and in the buffers crossing the ABI (`double*`). There is no runtime-order C++ type, no heap expansion, and no change to the dense core. This is the property that distinguishes the design from the nanobind/sparse approach.

## 11. Error handling

- **No compiler found / wrong standard:** a clear `tax.CompilerNotFound` at first build, naming the tried candidates and the `TAX_CXX` override.
- **Compile failure:** surface the generated TU path + compiler stderr in a `tax.JitCompileError` (and always on `dump=True`); never cache a failed build.
- **Unsupported op / non-traceable control flow:** the tracer raises `tax.TraceError` naming the op (e.g. data-dependent `if` on an `Expansion`).
- **Scheme/axis errors:** incompatible axis caps or order mismatches raise at promotion time with the offending axes (mirrors the C++ `static_assert` messages at runtime).
- **Domain errors at runtime:** kernels trap (return nonzero) on ill-defined constant terms (`log`/`sqrt` of non-positive, `pow` with a zero base and negative exponent) → `tax.DomainError`.
- **Buffer-shape mismatches:** validated in the Python wrapper before the FFI call.

## 12. Build, packaging & runtime-compiler requirement

- **Pure-Python base package** (`tax/`): tracer, IR, codegen, cache, loader — no compiled extension on the import or call path. The wheel is `py3-none-any` plus **vendored headers**: the `tax` headers and the needed **Eigen** headers (`<tax/tax.hpp>` pulls in `tax::la`, which needs Eigen). Eigen is MPL2 and redistributable; vendoring keeps the JIT self-contained except for the compiler.
- **Optional** `scikit-build-core` only if/when a native fast-path (e.g. a prebuilt PCH) is shipped; not required for the calling path.
- **Runtime requirement:** a **C++23 compiler** on the user's machine (documented prerequisite; normal in scientific/HPC/dev environments). Discovery via `TAX_CXX`/`CXX`/`PATH`. A future enhancement may bundle a compiler for pip-install-anywhere, but that is out of v1 scope.
- **Header path** for codegen points at the vendored include dir inside the installed package.

## 13. Performance considerations

- **Eager** pays one Python→C crossing per op; for tiny schemes the `ctypes` call can rival the math. Acceptable for authoring/exploration and removed by `tax.jit`. A steady-state eager op is `dict lookup + one FFI call`.
- **`tax.jit`** removes intermediate buffers and per-op crossings; the win is largest for deep expressions and vector maps (the two-body RHS).
- **First-touch compile latency** (`<tax/tax.hpp>` + Eigen at `-O3`) is seconds per *new* `(graph, scheme)`; amortized by the on-disk cache and cut by a shipped PCH.
- **Fusion is an assumption to validate, not a guarantee.** Milestone M0 includes a spike that inspects the generated assembly / benchmarks a known case (`sin(x)*exp(x)`) to confirm `-O3` actually fuses and promotes the `std::array`s, and quantifies eager-vs-jit-vs-hand-written-C++.

## 14. Testing strategy

- **Unit (pure-Python, no compiler):** scheme descriptor algebra (union/promotion, `nCoeff`, layout, canonical hashing); tracer/IR construction + canonicalization + CSE; generated-C++ **text** golden tests; cache-key composition.
- **Integration (with a compiler):** eager numerics vs a C++ oracle (build the same expression in C++, compare coefficients to `1e-12`); **`jit` numerics == eager numerics** for the same expression; cache hit/miss + atomic-build + concurrent-build (multiprocessing) behavior; `force_recompile`, `dump`, option plumbing.
- **End-to-end:** both two-body examples as tests — unnamed and named, eager and jit, checking `value()`, `jacobian()`/`jacobian("x")`, and parameter sensitivity (`jacobian("mu")`).
- **Regression:** reuse the DACE comparison suite where applicable to validate coefficient accuracy of the JIT path.
- **Cross-compiler:** matrix over `clang++` and `g++` in CI.

## 15. Milestones

- **M0 — Spike / de-risk.** Hand-emit a TU for `sin(x)*exp(x)` at a fixed scheme; compile, `ctypes`-load, verify numerics + that `-O3` fuses; settle compiler discovery + cache plumbing + ABI v0. *Gate: the core assumption holds.*
- **M1 — Eager scalar, isotropic.** `Expansion`, `variable/variables` (indexed), full math surface, eager engine + compile/cache/load. *Gate: eager numerics == C++ oracle.*
- **M2 — Vectors.** `Array`, `concatenate`/`stack`/indexing/elementwise/`dot`, `value`/`eval`/`jacobian`/`hessian`. *Gate: vector eager numerics.*
- **M3 — Named + mixed schemes.** Named/indexed/mixed scheme descriptors, union promotion, codegen for `MixedTaylorExpansion`/`MixedScheme`. *Gate: named eager two-body RHS.*
- **M4 — `tax.jit` fusion.** Trace-on-first-call, whole-function graph, fused TU, options (`opt/cache/compiler/scalar/batch/static_argnums/dump`), multi-output maps. *Gate: jit numerics == eager; both two-body RHS maps run under jit.*
- **M5 — Targets + regression + perf.** Both target examples as e2e tests; DACE/C++ regression; eager-vs-jit-vs-C++ benchmarks. *Gate: targets pass, perf characterized.*
- **M6 — Packaging & docs.** Pure-Python wheel + vendored `tax`/Eigen headers, PCH warm-build, compiler discovery docs, examples, API reference. *Gate: `pip install` + a C++23 compiler runs the examples.*

## 16. Open questions & risks

- **Fusion effectiveness at large `(N,M)`.** For big schemes the `std::array`s stay in memory; the loops still fuse but register promotion won't. M0 quantifies where the crossover is. *Risk: medium; mitigated by measuring early.*
- **Compile latency UX.** Even cached, the *first* run of a fresh program is seconds. PCH + a small shipped grid of the most common `(order,size)` could give zero first-touch for the hot cases. *Decide after M0 numbers.*
- **Eager kernel cache cardinality.** `(op, operand-schemes)` can be large with many distinct mixed schemes; bounded in practice by the schemes a program actually uses, and each entry is small. *Risk: low.*
- **Tracer limitations.** Data-dependent Python control flow over `Expansion` values can't be traced (standard JAX caveat); documented, with a clear `TraceError`.
- **Eigen vendoring & licensing.** MPL2 redistribution is fine but must be documented in the wheel. *Risk: low.*
- **`t` (time) semantics in non-autonomous RHS.** A plain-float `t` is a runtime scalar input; making `t` an axis (for ∂/∂t) is just passing a named/indexed `Expansion` — both supported by the input-kind mechanism. *No blocker.*
- **Thread-safety.** In-process kernel-cache LRU guarded by a lock; on-disk builds guarded by a per-key file lock. *Resolved in design.*

## 17. Relationship to prior branches

- **Supersedes** the nanobind/sparse and pybind11/grid bindings as the dense, static-storage, unbounded-scheme path — but **borrows** their packaging learnings (scikit-build-core config, test layout) and may lift the `tax::la`-based gradient/jacobian marshaling.
- **Does not** depend on or merge the MLIR branch; it deliberately stays in the header-library world. If, later, cross-op *mathematical* fusion (sin/cos pairing, degree scheduling) proves worth more than the C++ optimizer delivers, the MLIR pipeline becomes an *alternative codegen backend* behind the same Graph IR — a clean future extension, not a v1 commitment.
