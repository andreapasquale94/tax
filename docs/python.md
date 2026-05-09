# Python

The `tax` Python package wraps the C++ library through
[nanobind](https://nanobind.readthedocs.io). Per the architectural
brief, **only `DynamicTaylorExpansion<double>` is exposed** (as
`tax.DynTE`); the static-extent C++ path stays C++-only — no
`std::variant` over an `(Order, Vars)` grid, no JIT.

## Build & install

```sh
pip install nanobind pytest
cmake -S . -B build -DTAX_BUILD_PYTHON=ON
cmake --build build -j
PYTHONPATH=build/python python -c "import tax; print(tax.DynTE)"
```

The compiled module lives in `build/python/tax/_tax.so`; the Python
package wrapper at `build/python/tax/__init__.py` re-exports the
useful surface.

## Surface

```python
import tax

# Constructors / factories
tax.DynTE(order, nvars)                   # zero-initialised
tax.DynTE.zero(order, nvars)
tax.DynTE.one(order, nvars)
tax.DynTE.constant(value, order, nvars)
tax.DynTE.variable(value, order, nvars, var_idx)
tax.DynTE.variables([x0, x1, ...], order)  # returns a Python list

# Properties / accessors
x.order
x.nvars
x.value()
x.coeff([1, 0])                            # multi-index as a list of ints
x.derivative([1, 1])
x.eval([0.1, 0.05])
x.coeffs_norm_inf()
x.coeffs_norm_1()
x.coeffs_norm_2()

# Arithmetic — return fresh DynTEs (no lazy ETs in Python)
a + b      a - b      a * b      a / b      -a
a + 1.5    1.5 + a    a - 1.5    1.5 - a    a * 2.0    2.0 * a    a / 2.0

# Math
tax.sin(a)   tax.cos(a)   tax.tan(a)
tax.sinh(a)  tax.cosh(a)  tax.tanh(a)
tax.exp(a)   tax.log(a)   tax.sqrt(a)
tax.square(a)  tax.cube(a)
```

## Why eager evaluation?

In C++, operator chains build lazy ET trees that the `<<=` driver
consumes within a single full expression. Python statements don't
have that lifetime: an intermediate `MulExpr` referenced through
`x = a * b` would have to outlive the statement.

The Python bindings therefore evaluate every operator into a fresh
`DynTE` immediately:

```cpp
.def("__mul__",
     [](const DynTE& a, const DynTE& b) {
         DynTE out(a.order(), a.nvars());
         out <<= a * b;        // ET runs once into `out`
         return out;
     })
```

Each Python operator pays one streaming sweep. The C++ ET
infrastructure is still doing the work under the hood, just consumed
at every step instead of fused across the whole expression.

## Running the tests

```sh
PYTHONPATH=build/python pytest python/tests
```

CTest also picks up the suite as the `python_bindings` test when
`pytest` is on `PATH` and `TAX_BUILD_PYTHON=ON`.

## End-to-end example

```python
import math
import tax

u, v = tax.DynTE.variables([1.0, 2.0], 3)
f = u * tax.sin(v) + u * v

print(f.value())            # ≈ sin(2) + 2
print(f.derivative([1, 0])) # = sin(2) + 2
print(f.derivative([0, 1])) # = cos(2) + 1
```

## Gotchas

- **Multi-indices are Python lists of ints.** A length mismatch with
  `nvars` raises a TypeError at conversion time (nanobind's
  `std::vector` caster).
- **Order and nvars must match across operands.** Mixed-shape
  arithmetic fires the same compile-time `SameKindExpression` check
  on the C++ side; from Python you'll see a runtime exception out of
  the binding wrapper.
- **No NumPy zero-copy yet.** `coeff` / `derivative` /
  `coeffs_norm_*` return Python floats. If you need bulk access to
  the coefficient buffer, that's a future-work item — file an issue.
