# Getting Started

## What is TAX?

TAX is a header-only C++23 library for **Truncated Algebraic eXpansions**. It lets you compute with truncated multivariate Taylor polynomials as first-class objects: instead of evaluating a function at a single point, TAX propagates an entire truncated Taylor series through your computation, giving you the function value *and* all partial derivatives up to order $N$ in one pass.

A DA variable $x$ expanded around a point $x_0$ represents the truncated series:

$$x = x_0 + \delta x$$

When you compose DA variables through arbitrary expressions (arithmetic, trigonometric, transcendental, ...), the library automatically propagates the series through every operation, keeping all cross-terms up to total degree $N$. The result is a polynomial whose coefficients encode all partial derivatives at the expansion point.

## Building

TAX requires a C++23 compiler. Eigen support is optional.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

To enable Eigen integration:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_ENABLE_EIGEN=ON
cmake --build build
```

### CMake Options

| Option             | Default | Description                       |
|--------------------|---------|-----------------------------------|
| `TAX_BUILD_TEST`   | `ON`    | Build the test suite              |
| `TAX_ENABLE_EIGEN` | `OFF`   | Enable Eigen adapters and tensors |

### Installation

```bash
cmake --install build --prefix /your/install/prefix
```

Then from another CMake project:

```cmake
find_package(tax CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tax::tax)
```

## Include

A single umbrella header pulls in everything:

```cpp
#include <tax/tax.hpp>
```

If Eigen headers are available on the include path, the Eigen integration is auto-detected even without the CMake flag.

## Core Type

The central type is `TruncatedTaylorExpansionT<T, N, M>`:

| Parameter | Meaning                                      |
|-----------|----------------------------------------------|
| `T`       | Scalar coefficient type (e.g. `double`)      |
| `N`       | Maximum total polynomial order               |
| `M`       | Number of independent variables (default `1`) |

Two convenience aliases are provided:

```cpp
template <int N>    using TE  = TruncatedTaylorExpansionT<double, N, 1>;   // univariate
template <int N, int M> using TEn = TruncatedTaylorExpansionT<double, N, M>;   // multivariate
```

A `TruncatedTaylorExpansionT` stores $\binom{N+M}{M}$ coefficients in graded lexicographic (grlex) order.

## Creating Variables

**Univariate** --- expand around $x_0$:

```cpp
auto x = TE<5>::variable(1.0);   // x = 1 + δx
```

The constant term is $x_0 = 1$ and the first-order coefficient is $1$ (the identity perturbation). All higher coefficients are zero.

**Multivariate** --- structured bindings:

```cpp
auto [x, y] = TEn<3, 2>::variables({1.0, 2.0});   // expand at (1, 2)
```

Each variable carries a unit perturbation in its own direction.

**Single indexed variable:**

```cpp
auto z = TEn<2, 3>::variable<2>({1.0, 2.0, 3.0});   // only variable 2
```

## Building Expressions

Arithmetic and math functions work naturally:

```cpp
auto x = TE<6>::variable(0.0);
TE<6> f = tax::sin(x) + tax::square(x) / 2.0;
```

The right-hand side builds a lazy expression tree. Evaluation happens only on assignment to a `TruncatedTaylorExpansionT` object.

## Extracting Results

```cpp
f.value();              // f(x₀)  --- the constant term
f.coeff({k});           // coefficient of δxᵏ
f.derivative({k});      // k-th derivative at x₀ (= k! · coeff({k}))
f.eval(0.3);            // evaluate the polynomial at x₀ + 0.3
```

For multivariate DA objects, multi-indices are passed as initializer lists:

```cpp
auto [x, y] = TEn<3, 2>::variables({0.0, 0.0});
TEn<3, 2> g = x*x + x*y + y*y;

g.derivative({2, 0});   // ∂²g/∂x²   = 2
g.derivative({1, 1});   // ∂²g/∂x∂y  = 1
g.derivative({0, 2});   // ∂²g/∂y²   = 2
```

Compile-time multi-index access is also available via template arguments:

```cpp
g.coeff<1, 1>();        // coefficient of δx·δy
g.derivative<2, 0>();   // ∂²g/∂x²
```

## Coefficients vs Derivatives

A DA polynomial stores **coefficients** of the monomial basis:

$$f(\mathbf{x}_0 + \delta\mathbf{x}) = \sum_{|\alpha| \le N} f_\alpha \, \delta\mathbf{x}^\alpha$$

The relationship to partial derivatives is:

$$f_\alpha = \frac{1}{\alpha!} \frac{\partial^{|\alpha|} f}{\partial x_1^{\alpha_1} \cdots \partial x_M^{\alpha_M}} \bigg|_{\mathbf{x}_0}$$

where $\alpha! = \alpha_1! \cdots \alpha_M!$. The `derivative()` method returns the partial derivative (i.e. it multiplies the stored coefficient by $\alpha!$), while `coeff()` returns the raw coefficient.

## Polynomial Evaluation

Use `eval()` to evaluate the stored polynomial at a displacement from the expansion point:

```cpp
auto x = TE<9>::variable(0.0);
TE<9> f = tax::sin(x);

// Taylor polynomial of sin(x) around x₀ = 0, evaluated at δx = 0.3
double result = f.eval(0.3);   // ≈ sin(0.3)
```

For multivariate:

```cpp
auto [x, y] = TEn<4, 2>::variables({1.0, 2.0});
TEn<4, 2> f = tax::exp(x + y);

double result = f.eval({0.1, -0.1});   // ≈ exp(1.1 + 1.9)
```

## In-place Operations

All compound assignment operators are supported:

```cpp
TE<4> f = TE<4>::variable(1.0);
f += TE<4>{2.0};
f -= TE<4>{1.0};
f *= 3.0;           // scalar multiply
f /= 2.0;           // scalar divide
f *= other_da;      // Cauchy product
f /= other_da;      // division via reciprocal
```

## Comparison Operators

Comparisons act on the **constant term** (value at expansion point):

```cpp
TE<3> a = TE<3>::variable(2.0);
TE<3> b = TE<3>::variable(3.0);
a < b;    // true, because 2.0 < 3.0
a == 2.0; // true
```

## Streaming

DA objects can be printed directly:

```cpp
auto x = TE<3>::variable(1.0);
TE<3> f = x * x + 2.0 * x;
std::cout << f << "\n";
// Output: 3 + 4·δx + δx² + O(δx⁴)
```

## A Complete Example

Compute $\sin(x)$ and its first nine derivatives at $x = 0$, then evaluate the resulting polynomial:

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    using tax::TE;

    auto x = TE<9>::variable(0.0);
    TE<9> f = tax::sin(x);

    for (int k = 0; k <= 9; ++k)
        std::cout << "d^" << k << " sin/dx^" << k << " (0) = "
                  << f.derivative({k}) << "\n";

    std::cout << "sin(0.3) ≈ " << f.eval(0.3) << "\n";
    std::cout << "sin(0.5) ≈ " << f.eval(0.5) << "\n";
}
```

## Expression Template Optimisations

TAX automatically optimises expression trees behind the scenes:

- **Sum flattening**: `(a + b) + c` becomes a single flat `SumExpr<A, B, C>` --- one accumulation pass instead of nested binary nodes.
- **Product flattening**: `a * b * c` becomes `ProductExpr<A, B, C>` --- computed with a rolling accumulator using a single intermediate buffer.
- **Leaf fast-paths**: when both operands of a binary operation are already materialised `TruncatedTaylorExpansionT` objects, the kernel is called directly without wrapping in an expression node.

These optimisations are transparent --- write natural mathematical expressions and the compiler handles the rest.

## Next Steps

- [API Reference](api_reference.md) --- full listing of types, factories, accessors, and operations.
- [Eigen Integration](eigen_integration.md) --- working with Eigen vectors, matrices, and tensors.
- [Math Operations](math_operations.md) --- detailed recurrence formulas for every supported operation.
