# tax

**tax** is a header-only C++23 library for **Truncated Algebraic eXpansions** — a framework for computing truncated multivariate Taylor polynomials as first-class objects.

Write natural mathematical expressions and tax automatically propagates the full Taylor series, giving you the function value **and** all partial derivatives up to order \(N\) in a single evaluation pass.

## Features

- **Compile-time fixed** order \(N\) and variable count \(M\) via `TruncatedTaylorExpansionT<T, N, M>`
- **Lazy expression templates** with automatic sum/product flattening and leaf fast-paths
- **Comprehensive math**: arithmetic, trigonometric, hyperbolic, transcendental, power, and special functions
- **Direct derivative access**: coefficients, partial derivatives, gradient, Jacobian, and higher-order derivative tensors
- **Eigen integration**: adapters for Eigen vectors, matrices, and tensors
- **Taylor ODE integrator**: adaptive high-order integration for scalar and vector ODEs
- **Automatic Domain Splitting**: adaptive polynomial approximation over large domains

## Quick Example

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    // sin(x) expanded at x₀ = 0, up to order 9
    auto x = tax::TE<9>::variable(0.0);
    tax::TE<9> f = tax::sin(x);

    std::cout << f.value()         << "\n";   // sin(0) = 0
    std::cout << f.derivative({1}) << "\n";   // cos(0) = 1
    std::cout << f.eval(0.3)       << "\n";   // ≈ sin(0.3)
}
```

## Modules

| Module | Description |
|--------|-------------|
| [Core](core/index.md) | Truncated Taylor polynomials, expression templates, and mathematical functions |
| [Vector (Eigen)](vector/index.md) | Eigen integration for vectors, matrices, Jacobians, and higher-order tensors |
| [Automatic Domain Splitting](ads/index.md) | Adaptive polynomial approximation over large domains |
| [Taylor Integrator](taylor/index.md) | High-order adaptive ODE integration with DA-based flow maps |

## Requirements

- C++23 compiler (GCC 13+, Clang 17+, Apple Clang 16+)
- CMake 4.2+
- Eigen 3.4+

## License

BSD 3-Clause. See [LICENSE](https://github.com/andreapasquale94/tax/blob/main/LICENSE) for details.
