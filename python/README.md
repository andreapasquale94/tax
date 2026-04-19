# tax — Python bindings

Runtime-dimensioned Python bindings for the **tax** C++ library (Truncated
Algebraic eXpansions). Order `N` and variable count `M` are selected at runtime
and operations dispatch to a precompiled grid of `(N, M)` instantiations.

## Installation

### From source

```bash
pip install ./python
```

(The parent `CMakeLists.txt` is used via `scikit-build-core`; Eigen3 must be
discoverable — install via your system package manager or point
`CMAKE_PREFIX_PATH` at a custom prefix.)

### Development build

```bash
cmake -S . -B build -DTAX_BUILD_PYTHON=ON -DTAX_BUILD_TEST=OFF
cmake --build build -j
PYTHONPATH=build/python python -c "import tax; print(tax.__version__)"
```

## Supported (N, M) grid

At build time, the bindings compile every `(N, M)` with
`1 <= N <= TAX_PY_MAX_N`, `1 <= M <= TAX_PY_MAX_M`, and
`C(N+M, M) <= TAX_PY_MAX_COEFFS`. Defaults: `N<=20`, `M<=10`, `<=10_000`
coefficients. Requesting an unsupported pair raises `ValueError`.

## Quick example

```python
import tax

x, y = tax.variables(order=5, nvars=2, x0=[1.0, 2.0])
f = tax.sin(x) * tax.exp(y)

f.value()            # constant term
f.eval([0.1, -0.05]) # evaluate at a displacement
f.coeff([2, 1])      # raw monomial coefficient for x^2 y
f.derivative([1, 0]) # partial ∂/∂x at expansion point
```

See `python/tests/test_te.py` for a broader set of examples.
