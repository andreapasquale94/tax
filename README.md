# tax

[![Tests](https://github.com/andreapasquale94/tax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax/actions/workflows/tests.yml)
[![Sanitizers](https://github.com/andreapasquale94/tax/actions/workflows/sanitizers.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax/actions/workflows/sanitizers.yml)
[![codecov](https://codecov.io/gh/andreapasquale94/tax/graph/badge.svg?token=XwO5JOoaz6)](https://codecov.io/gh/andreapasquale94/tax)

**tax** is a header-only C++23 library for **Truncated Algebraic eXpansions** -- a framework for computing truncated multivariate Taylor polynomials.

Write natural mathematical expressions and tax automatically propagates the full Taylor series, giving you the function value and all partial derivatives up to order $N$ in a single evaluation pass.

    DISCLAIMER: this repository is under active development. APIs and behavior may change; use with care.

## Features

- **Compile-time fixed** order $N$ and variable count $M$ via `TruncatedTaylorExpansionT<T, N, M>`
- **Lazy expression templates** with automatic sum/product flattening and leaf fast-paths
- **Comprehensive math**: arithmetic, trigonometric, hyperbolic, transcendental, power, and special functions
- **Direct derivative access**: coefficients, partial derivatives, gradient, Jacobian, and higher-order derivative tensors
- **Eigen integration**: adapters for Eigen vectors, matrices, and tensors

## Requirements

- C++23 compiler
- CMake 4.2+
- Eigen 3.4+

## Quick Start

### Univariate

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    using tax::TE;

    // sin(x) expanded at x₀ = 0, up to order 9
    auto x = TE<9>::variable(0.0);
    TE<9> f = tax::sin(x);

    std::cout << f.value()          << "\n";   // sin(0) = 0
    std::cout << f.derivative({1})  << "\n";   // cos(0) = 1
    std::cout << f.derivative({2})  << "\n";   // -sin(0) = 0
    std::cout << f.eval(0.3)        << "\n";   // ≈ sin(0.3)
}
```

### Multivariate

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    using tax::TEn;

    // f(x, y) = sin(x + y) expanded at (1, 2)
    auto [x, y] = TEn<3, 2>::variables({1.0, 2.0});
    TEn<3, 2> f = tax::sin(x + y);

    std::cout << f.value()              << "\n";   // sin(3)
    std::cout << f.derivative({1, 0})   << "\n";   // ∂f/∂x = cos(3)
    std::cout << f.derivative({1, 1})   << "\n";   // ∂²f/∂x∂y = -sin(3)
}
```

## Build and Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

| Option             | Default | Description                       |
|--------------------|---------|-----------------------------------|
| `TAX_BUILD_TEST`   | `ON`    | Build the test suite              |
| `TAX_BUILD_BENCHMARK` | `OFF` | Build Google Benchmark suite      |
| `TAX_USE_DACE` | `OFF`  | Fetch/enable DACE for tests and benchmarks |

### Benchmarks (TAX vs DACE)

```bash
cmake -S . -B build-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DTAX_BUILD_TEST=OFF \
  -DTAX_BUILD_BENCHMARK=ON \
  -DTAX_USE_DACE=ON
cmake --build build-bench --target univariate -j

# Run all univariate benchmarks
./build-bench/benchmarks/nivariat

# Example: compare only sin and exp
./build-bench/benchmarks/nivariat --benchmark_filter='(Tax|Dace)/(Sin|Exp)/.*'
```

## Install

```bash
cmake --install build --prefix /your/install/prefix
```

From another CMake project:

```cmake
find_package(tax CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tax::tax)
```

If installed to a non-standard prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/your/install/prefix
```

## API at a Glance

```cpp
#include <tax/tax.hpp>
```

### Types

| Type            | Description                                       |
|-----------------|---------------------------------------------------|
| `TE<N>`         | `TruncatedTaylorExpansionT<double, N, 1>`                               |
| `TEn<N, M>`     | `TruncatedTaylorExpansionT<double, N, M>`                               |

### Factories

```cpp
TE<N>::variable(x0)              // univariate variable at x₀
TEn<N,M>::variable<I>(x0)       // I-th variable at expansion point
TEn<N,M>::variables(x0)         // all variables (structured bindings)
TruncatedTaylorExpansionT::constant(v) / zero() / one()
```

### Accessors

```cpp
f.value()            // f(x₀)
f.coeff({2, 1})      // coefficient of δx²·δy
f.derivative({2, 1}) // ∂³f/∂x²∂y at x₀
f.derivatives()      // all partial derivatives
f.eval(dx)           // polynomial evaluated at x₀ + δx
```

### Operations

**Arithmetic**: `+`, `-`, `*`, `/` between DA expressions and scalars

**Unary math**: `abs`, `square`, `cube`, `sqrt`, `cbrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `log`, `log10`, `erf`

**Binary math**: `pow` (integer, real, DA exponents), `atan2`, `hypot` (2- and 3-argument)

## Documentation

| Document                                       | Description                                  |
|------------------------------------------------|----------------------------------------------|
| [Getting Started](docs/getting_started.md)     | Installation, core concepts, first examples  |
| [API Reference](docs/api_reference.md)         | Types, factories, accessors, operations      |
| [Eigen Integration](docs/eigen_integration.md) | Vectors, matrices, tensors, Jacobians        |
| [Math Operations](docs/math_operations.md)     | Recurrence formulas for every operation      |

## License

See [LICENSE](LICENSE) for details.
