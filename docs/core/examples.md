# Examples

Worked examples demonstrating the core features of the **tax** library.

---

## Creating Variables

### Univariate

```cpp
#include <tax/tax.hpp>

// Order-5 expansion of x around x₀ = 1.0
auto x = tax::TE<5>::variable(1.0);
// Coefficients: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
//                 ↑     ↑
//              value  δx coefficient

// A pure constant (no δx dependence)
auto c = tax::TE<5>::constant(3.14);
// Coefficients: [3.14, 0, 0, 0, 0, 0]
```

### Multivariate with Structured Bindings

```cpp
#include <tax/tax.hpp>

// Order-3 expansion in two variables around (1.0, 2.0)
auto [x, y] = tax::TEn<3, 2>::variables(1.0, 2.0);
// x has a unit coefficient for δx₁, y has a unit coefficient for δx₂

// Three variables
auto [a, b, c] = tax::TEn<4, 3>::variables(0.0, 1.0, -1.0);
```

---

## Accessing Coefficients and Derivatives

### Univariate

```cpp
auto x = tax::TE<5>::variable(0.0);
auto f = sin(x);

// Constant term (the function value)
double val = f.value();          // sin(0) = 0

// Taylor coefficients: f_k = f^(k)(x₀) / k!
double c0 = f.coeff(0);         // 0
double c1 = f.coeff(1);         // 1
double c2 = f.coeff(2);         // 0
double c3 = f.coeff(3);         // -1/6

// Derivatives (with k! scaling applied)
double d1 = f.derivative(1);    // f'(0)   = cos(0) = 1
double d2 = f.derivative(2);    // f''(0)  = -sin(0) = 0
double d3 = f.derivative(3);    // f'''(0) = -cos(0) = -1
```

### Multivariate

```cpp
auto [x, y] = tax::TEn<3, 2>::variables(1.0, 2.0);
auto g = x * x * y;

// Access by multi-index: coeff({α₁, α₂})
double c_200 = g.coeff({2, 0});  // coefficient of δx²
double c_010 = g.coeff({0, 1});  // coefficient of δy
double c_110 = g.coeff({1, 1});  // coefficient of δx·δy
```

---

## Arithmetic and Composition

The library builds lightweight expression trees that are evaluated only on
assignment. Sums and products are automatically flattened.

```cpp
auto x = tax::TE<5>::variable(1.0);

// Arithmetic builds expression trees, materialised on assignment
tax::TE<5> f = (x + 2.0) * (x - 3.0);   // x² - x - 6 at x₀ = 1

// Chain of additions → single flattened SumExpr (one pass)
tax::TE<5> g = x + x * x + x * x * x;

// Division
tax::TE<5> h = 1.0 / (1.0 + x);         // geometric series at x₀ = 1
```

### Multivariate Products

```cpp
auto [x, y] = tax::TEn<3, 2>::variables(1.0, 2.0);

// Mixed-variable expression
tax::TEn<3, 2> f = x * x + 2.0 * x * y + y * y;  // (x + y)²
```

---

## Differentiation and Integration

Every TTE object supports symbolic differentiation and integration with respect
to any variable.

### Compile-Time Variable Index

```cpp
auto [x, y] = tax::TEn<4, 2>::variables(1.0, 2.0);
auto f = x * x * y + y * y;

// Partial derivatives (compile-time index)
auto df_dx = f.deriv<0>();    // ∂f/∂x = 2xy
auto df_dy = f.deriv<1>();    // ∂f/∂y = x² + 2y

// Integration
auto F_x = f.integ<0>();     // ∫f dx
auto F_y = f.integ<1>();     // ∫f dy
```

### Runtime Variable Index

```cpp
auto x = tax::TE<5>::variable(1.0);
auto f = exp(x);

// Runtime index (throws if out of range)
auto df = f.deriv(0);        // d/dx exp(x) = exp(x)
auto F  = f.integ(0);        // ∫exp(x) dx
```

### Verifying Derivative Identities

```cpp
auto x = tax::TE<6>::variable(0.5);
auto f = sin(x);

// sin'(x) = cos(x)
auto df = f.deriv<0>();
tax::TE<6> expected = cos(tax::TE<6>::variable(0.5));

// The coefficients of df and expected match to machine precision
```

---

## Transcendental Functions

All standard mathematical functions are supported. They are implemented as
degree-by-degree recurrence relations, computing all coefficients in a single
forward pass.

```cpp
auto x = tax::TE<8>::variable(0.0);

// Trigonometric
tax::TE<8> s = sin(x);     // [0, 1, 0, -1/6, 0, 1/120, ...]
tax::TE<8> c = cos(x);     // [1, 0, -1/2, 0, 1/24, ...]

// Exponential and logarithm
auto y = tax::TE<8>::variable(1.0);
tax::TE<8> e = exp(y);     // exp(1+δx) = e · [1, 1, 1/2, 1/6, ...]
tax::TE<8> l = log(y);     // log(1+δx) = [0, 1, -1/2, 1/3, ...]

// Hyperbolic
tax::TE<8> sh = sinh(x);
tax::TE<8> ch = cosh(x);

// Power and roots
tax::TE<8> sq = sqrt(tax::TE<8>::variable(4.0));  // √(4+δx)
tax::TE<8> cb = cbrt(tax::TE<8>::variable(8.0));  // ∛(8+δx)
```

### Composed Expressions

```cpp
auto x = tax::TE<10>::variable(0.0);

// Compound expressions work naturally
tax::TE<10> f = sin(x) * exp(x);
tax::TE<10> g = log(1.0 + x * x);
tax::TE<10> h = atan(x) / (1.0 + x * x);
```

---

## Polynomial Evaluation

Use `.eval()` to evaluate the Taylor polynomial at a displacement
\(\delta x\) from the expansion point.

```cpp
auto x = tax::TE<15>::variable(0.0);
tax::TE<15> f = sin(x);

// Evaluate at x = 0.3 (i.e., x₀ + δx = 0 + 0.3)
double approx = f.eval(0.3);
// approx ≈ sin(0.3) = 0.29552020666...
// The order-15 approximation matches to ~16 digits
```

### Multivariate Evaluation

```cpp
auto [x, y] = tax::TEn<5, 2>::variables(0.0, 0.0);
tax::TEn<5, 2> f = sin(x) * cos(y);

// Evaluate at (x, y) = (0.3, 0.5)
double approx = f.eval({0.3, 0.5});
// approx ≈ sin(0.3) * cos(0.5)
```

---

## Norms and Convergence

TTE objects provide norms useful for convergence monitoring and step-size
control.

```cpp
auto x = tax::TE<10>::variable(0.0);
tax::TE<10> f = sin(x);

// Sum of absolute values of all coefficients
double abs_sum = f.absSum();

// Infinity norm of the coefficient vector
double inf_norm = f.normInf();
```

---

## Nested Expansions (DA over DA)

A TTE whose scalar type is itself a TTE enables **differential algebra** maps:
expand in time around a spatial expansion.

```cpp
#include <tax/tax.hpp>

using SpatialDA = tax::TEn<4, 2>;   // order-4, 2 spatial variables
using TimeDA = tax::TruncatedTaylorExpansionT<SpatialDA, 6, 1>;
// order-6 in time, coefficients are spatial polynomials

// Create a time variable whose constant term is a spatial expansion
SpatialDA x0 = SpatialDA::constant(1.0);
auto t = TimeDA::variable(x0);

// Operations on t propagate through both levels
TimeDA f = sin(t);
// f is a time polynomial whose coefficients are spatial polynomials
```
