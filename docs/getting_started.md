# Getting Started

## Installation

### Building from Source

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `TAX_BUILD_TEST` | `ON` | Build Google Test suite |
| `TAX_BUILD_BENCHMARK` | `OFF` | Build Google Benchmark suite |
| `TAX_USE_DACE` | `OFF` | Enable DACE comparison tests |

### Using tax in Your Project

```bash
cmake --install build --prefix /your/install/prefix
```

```cmake
find_package(tax CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tax::tax)
```

## Include

A single umbrella header pulls in everything:

```cpp
#include <tax/tax.hpp>
```

## Core Type

The central type is `TruncatedTaylorExpansionT<T, N, M>`:

| Parameter | Meaning |
|-----------|---------|
| `T` | Scalar coefficient type (e.g. `double`) |
| `N` | Maximum total polynomial order |
| `M` | Number of independent variables (default `1`) |

Two convenience aliases:

```cpp
tax::TE<N>        // univariate (M = 1)
tax::TEn<N, M>    // multivariate
```

## Creating Variables

=== "Univariate"

    ```cpp
    auto x = tax::TE<5>::variable(1.0);   // x = 1 + δx
    ```

=== "Multivariate"

    ```cpp
    auto [x, y] = tax::TEn<3, 2>::variables({1.0, 2.0});
    ```

=== "Single indexed"

    ```cpp
    auto z = tax::TEn<2, 3>::variable<2>({1.0, 2.0, 3.0});
    ```

## Building Expressions

Arithmetic and math functions work naturally:

```cpp
auto x = tax::TE<6>::variable(0.0);
tax::TE<6> f = tax::sin(x) + tax::square(x) / 2.0;
```

The right-hand side builds a lazy expression tree. Evaluation happens only on assignment to a `TruncatedTaylorExpansionT` object.

## Extracting Results

```cpp
f.value();              // f(x₀) — the constant term
f.coeff({k});           // coefficient of δxᵏ
f.derivative({k});      // k-th derivative (= k! · coeff)
f.eval(0.3);            // evaluate polynomial at x₀ + 0.3
```

For multivariate DA objects:

```cpp
auto [x, y] = tax::TEn<3, 2>::variables({0.0, 0.0});
tax::TEn<3, 2> g = x*x + x*y + y*y;

g.derivative({2, 0});   // ∂²g/∂x²   = 2
g.derivative({1, 1});   // ∂²g/∂x∂y  = 1
g.coeff<1, 1>();         // compile-time access
```

## Coefficients vs Derivatives

A DA polynomial stores **coefficients** of the monomial basis:

\[
f(\mathbf{x}_0 + \delta\mathbf{x}) = \sum_{|\alpha| \le N} f_\alpha \, \delta\mathbf{x}^\alpha
\]

The relationship to partial derivatives is:

\[
f_\alpha = \frac{1}{\alpha!} \frac{\partial^{|\alpha|} f}{\partial x_1^{\alpha_1} \cdots \partial x_M^{\alpha_M}} \bigg|_{\mathbf{x}_0}
\]

The `derivative()` method returns the partial derivative (multiplies the coefficient by \(\alpha!\)), while `coeff()` returns the raw coefficient.

## Next Steps

- [Core Module](core/index.md) — full treatment of the TTE type and expression templates
- [Vector (Eigen)](vector/index.md) — Eigen integration, Jacobians, Hessians
- [Automatic Domain Splitting](ads/index.md) — adaptive polynomial approximation
- [Taylor Integrator](taylor/index.md) — high-order ODE integration
