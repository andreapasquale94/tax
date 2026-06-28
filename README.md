# tax

[![Tests](https://github.com/andreapasquale94/tax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax/actions/workflows/tests.yml)
[![Sanitizers](https://github.com/andreapasquale94/tax/actions/workflows/sanitizers.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax/actions/workflows/sanitizers.yml)
[![Docs](https://github.com/andreapasquale94/tax/actions/workflows/docs.yml/badge.svg?branch=main)](https://andreapasquale94.github.io/tax/)
[![codecov](https://codecov.io/gh/andreapasquale94/tax/graph/badge.svg?token=XwO5JOoaz6)](https://codecov.io/gh/andreapasquale94/tax)

**tax** is a header-only C++23 library for **Truncated Algebraic eXpansions** —
truncated multivariate Taylor polynomials as first-class numerical objects.
Write a natural mathematical expression and tax propagates the full Taylor
series through it, yielding the function value **and** every partial derivative
up to order \(N\) in a single evaluation pass.

> :books: **Full documentation:** <https://andreapasquale94.github.io/tax/>

> :warning: **Active development.** APIs and behavior may change between minor
> versions until 1.0.

---

## Features

- **Compile-time fixed shape** — `TaylorExpansion<T, Scheme, Storage>`; the
  `Scheme` fixes the truncation order(s) and variable count at compile time, and
  `Storage` is `Dense` (stack `std::array`) or `Sparse` (sorted-index map). Both
  storages share the kernel layer and agree numerically.
- **Convenience aliases** — `TE<N, M=1>` / `TEn<N, M>` (dense), `STE<N, M=1>`
  (sparse), `NE<N, Axes...>` (named), `MTE<Axes...>` (mixed-order named).
- **Comprehensive math** — arithmetic, trigonometric, hyperbolic,
  transcendental, square/cubic root, reciprocal, integer & real powers,
  `atan2`, `erf`.
- **Direct derivative access** — coefficients, partial derivatives at the
  expansion point, full gradient / Hessian / Jacobian.
- **Eigen integration** — `NumTraits` specialisation plus helpers for
  variables, value extraction, evaluation, gradient, Jacobian, Hessian, and
  formal map inversion.
- **Named expansions** — `NamedTaylorExpansion<T, N, Axes...>` attaches
  compile-time *named axes* to an expansion; values over different axis sets
  compose in their union, and `slice`/`deriv`/`integ` are addressed by name.
  `MixedTaylorExpansion<T, Axes...>` gives each axis its own truncation order.
  The whole API is re-exported under `tax` (`tax::NE`, `tax::MTE`,
  `tax::variable(s)`).
- **Human-readable output** — `std::cout << f` prints the polynomial series;
  `tax::series(...)` adds tabular / per-element (Eigen) rendering.

> :electric_plug: **Plugin:** adaptive ODE integration and Automatic Domain
> Splitting are available as the optional
> [**tax-flow**](https://github.com/andreapasquale94/tax-flow) project, built on
> top of `tax`.

---

## How it works

A `TaylorExpansion<T, N, M>` stores the coefficients of the order-$N$ truncated
Taylor polynomial of a function of $M$ variables about an expansion point
$\mathbf{x}_0$:

$$
f(\mathbf{x}_0 + \delta\mathbf{x})
  = \sum_{|\alpha| \le N} f_\alpha \, \delta\mathbf{x}^\alpha,
\qquad
f_\alpha = \frac{1}{\alpha!}\,
  \frac{\partial^{|\alpha|} f}{\partial x_1^{\alpha_1}\cdots\partial x_M^{\alpha_M}}
  \bigg|_{\mathbf{x}_0}.
$$

The $\binom{N+M}{M}$ coefficients are laid out in graded-lexicographic order in a
`std::array` (dense) or a sorted index/value map (sparse). Every operation is a
**degree-by-degree recurrence relation** that writes the result coefficients
directly, so one evaluation pass yields the value *and* all derivatives up to
order $N$. For example, multiplication is the truncated Cauchy product, and the
exponential follows from $g' = f'g$ (univariate forms shown):

$$
(f \cdot g)_d = \sum_{k=0}^{d} f_k\, g_{d-k},
\qquad
g_0 = e^{f_0},\quad
g_d = \frac{1}{d} \sum_{k=0}^{d-1} (d-k)\, f_{d-k}\, g_k \;\; (d \ge 1).
$$

The same pattern covers `/`, `sqrt`, `log`, `sin`/`cos`, `pow`, `atan2`, `erf`, …
`coeff(α)` returns the raw $f_\alpha$; `derivative(α)` applies the $\alpha!$ scaling.

The full per-operation recurrence catalog (univariate and multivariate forms)
lives under
[Internals / Recurrence Relations](https://andreapasquale94.github.io/tax/internals/recurrences/),
and the convergence behaviour under
[Concepts](https://andreapasquale94.github.io/tax/concepts/).

---

## Requirements

- C++23 compiler — GCC 13+, Clang 17+, Apple Clang 16+
- CMake 3.28+
- Eigen 3.4+

---

## Quick start

### Univariate

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    auto x = tax::TE<9>::variable(0.0);     // x at x₀ = 0, order 9
    tax::TE<9> f = tax::sin(x);

    std::cout << f.value()           << "\n";   // sin(0)  = 0
    std::cout << f.derivative<1>()   << "\n";   // cos(0)  = 1
    std::cout << f.derivative<3>()   << "\n";   // -cos(0) = -1
    std::cout << f.eval({0.3})       << "\n";   // ≈ sin(0.3)
}
```

### Multivariate

```cpp
#include <tax/tax.hpp>

int main() {
    using TE2 = tax::TE<3, 2>;
    auto x = TE2::variable<0>({1.0, 2.0});
    auto y = TE2::variable<1>({1.0, 2.0});
    TE2 f = tax::sin(x + y);

    f.value();              // sin(3)
    f.derivative<1, 0>();   // ∂f/∂x   = cos(3)
    f.derivative<1, 1>();   // ∂²f/∂x∂y = -sin(3)
}
```

### Sparse storage

```cpp
auto x = tax::STE<5>::variable(1.0);     // same API, sparse storage
tax::STE<5> f = x * x + 2.0 * x - 1.0;   // arithmetic merges sorted nonzeros
auto g = tax::exp(x.dense());            // transcendental math runs on the dense form
```

### Eigen integration

```cpp
using TE2 = tax::TE<3, 2>;
auto x = TE2::variable<0>({1.0, 2.0});
auto y = TE2::variable<1>({1.0, 2.0});
Eigen::Vector2<TE2> F = { tax::sin(x), tax::cos(y) };

auto vals = tax::la::value(F);          // Eigen::Vector2d of constant terms
auto J    = tax::la::jacobian(F);       // 2×2 Jacobian at expansion point
```

### Named axes

```cpp
auto x = tax::variable<"x", 4>(1.0);    // order-4 axis "x"
auto p = tax::variable<"p", 4>(2.0);    // order-4 axis "p"
auto f = tax::sin(x) + x * p;           // composes in the union of axes {p, x}

auto dfdx = f.deriv<"x">();             // partial derivative by axis name
auto fx   = f.slice<"x">();             // project onto a single axis
```

`tax::jacobian<"x">(F)` likewise takes the Jacobian of an Eigen vector of named
expansions with respect to a named axis.

---

## Build, test, install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### CMake options

| Option | Default | Description |
|---|:-:|---|
| `TAX_BUILD_UNITTESTS` | `ON`  | Build the unit-test suite |
| `TAX_BUILD_REGRESSIONS` | `OFF` | Build DACE-based regression tests |

The fast Cauchy kernel paths (`TAX_USE_UNROLL` for M=1, `TAX_USE_STENCIL`
for M≥2) are enabled by default in `<tax/kernels/cauchy.hpp>` itself — no
build-system configuration needed. To opt out, pre-define the macro to `0`
identically in **every** translation unit (differing values are an ODR
violation).

### Consume from another project

```bash
cmake --install build --prefix /your/install/prefix
```

```cmake
find_package(tax CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tax::tax)
```

---

## Documentation

Hosted at <https://andreapasquale94.github.io/tax/>.

| Section | Topic |
|---|---|
| [Getting Started](https://andreapasquale94.github.io/tax/getting_started/) | Install, build, write your first Taylor expansion |
| [Guide](https://andreapasquale94.github.io/tax/guide/) | How-to: variables & expressions, extracting results, storage, named expansions, Eigen |
| [Reference](https://andreapasquale94.github.io/tax/reference/) | `TaylorExpansion` API, `tax::la`, `tax::named`, and the per-operation recurrence catalog |
| [Concepts](https://andreapasquale94.github.io/tax/concepts/) | The math: truncated Taylor polynomials, graded-lex ordering, convergence |
| [Internals](https://andreapasquale94.github.io/tax/internals/) | Architecture, kernels, recurrences |

Source for the docs lives in `docs/` and is built with MkDocs Material:

```bash
pip install mkdocs-material pymdown-extensions
mkdocs serve --strict
```

---

## License

[BSD 3-Clause](LICENSE).
