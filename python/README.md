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

The project's mamba **`tax` conda environment** provides both the C++23 compiler
(via conda-forge `cxx-compiler`) and Eigen (conda-forge `eigen`); running inside
that environment automatically makes them discoverable.

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
