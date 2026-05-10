# Math functions

All math functions live in `namespace tax` as templates constrained on
`TaxExpression`. Most return buffered ET nodes that fill their
coefficient buffer degree-by-degree via the streaming `.eval()` driver.

## Catalogue

### Trig & hyperbolic

| Function | Recurrence basis | ET node |
|----------|------------------|---------|
| `tax::sin(x)`     | sin/cos paired Euler-operator recurrence | `SinExpr<E>` |
| `tax::cos(x)`     | sin/cos paired                            | `CosExpr<E>` |
| `tax::tan(x)`     | `sin(x) / cos(x)` (composed)              | `DivExpr<SinExpr<E>, CosExpr<E>>` |
| `tax::sinh(x)`    | sinh/cosh paired                          | `SinhExpr<E>` |
| `tax::cosh(x)`    | sinh/cosh paired                          | `CoshExpr<E>` |
| `tax::tanh(x)`    | `sinh(x) / cosh(x)`                       | `DivExpr<SinhExpr<E>, CoshExpr<E>>` |

### Paired sincos / sinhcosh

```cpp
auto p = tax::sincos(x);   // owns one shared buffered node
TE<5> s, c;
s = (p.sin()).eval();             // streams sin slices from the shared buffers
c = p.cos().eval();               // already populated; the second .eval() is a buffer copy
```

Same pattern for `tax::sinhcosh(x)` exposing `.sinh()` / `.cosh()`.
Use this whenever you need both sides — it saves the second streaming
sweep relative to calling `tax::sin(x)` and `tax::cos(x)` separately
(which build two independent paired evaluators).

The `Pair` owner must outlive the views; consume them in the same
expression that produced the pair, or hold the pair in a named local.

### Inverse trig & hyperbolic

| Function | Auxiliary `G` | F<sub>0</sub> |
|----------|---------------|---------------|
| `tax::atan(x)`   | `1 + x²` (in a buffer)         | `std::atan(x_0)` |
| `tax::atanh(x)`  | `1 - x²`                       | `std::atanh(x_0)` |
| `tax::asin(x)`   | `√(1 - x²)`                    | `std::asin(x_0)` |
| `tax::acos(x)`   | `√(1 - x²)` (sign flipped RHS) | `std::acos(x_0)` |
| `tax::asinh(x)`  | `√(1 + x²)`                    | `std::asinh(x_0)` |
| `tax::acosh(x)`  | `√(x² - 1)`                    | `std::acosh(x_0)` |

All six share the recurrence `G(u) · E[F] = sign · E[u]` (Euler-operator
identity); the only thing that changes is `G` and the sign. Each ET
keeps its own `coeffs_` (for F) plus an auxiliary buffer for G.

### Exp / log

| Function | ET node |
|----------|---------|
| `tax::exp(x)`     | `ExpExpr<E>` |
| `tax::log(x)`     | `LogExpr<E>` |
| `tax::log10(x)`   | `ScalarMulExpr<LogExpr<E>>` (= `log(x) / log(10)`) |

### Roots, powers, square / cube

| Function | ET node |
|----------|---------|
| `tax::sqrt(x)`           | `SqrtExpr<E>` |
| `tax::cbrt(x)`           | `CbrtExpr<E>` (maintains F² alongside F) |
| `tax::square(x)`         | `SquareExpr<E>` |
| `tax::cube(x)`           | `MulExpr<SquareExpr<E>, E>` (composed) |
| `tax::pow<N>(x)`         | composed chain of `MulExpr` / `SquareExpr` / `DivExpr` |
| `tax::pow(x, p)`         | `PowRealExpr<E>` (recurrence `u F' = p u' F`) |

`pow<N>(x)` resolves the integer exponent at compile time and unrolls
into the optimal repeated-squaring chain. For runtime real exponents
use the two-argument `pow(x, p)`.

### atan2, hypot, erf

| Function | ET node |
|----------|---------|
| `tax::atan2(y, x)`           | `Atan2Expr<Y, X>` (maintains `x²+y²` alongside) |
| `tax::hypot(x, y)`           | `SqrtExpr<AddExpr<SquareExpr<X>, SquareExpr<Y>>>` |
| `tax::hypot(x, y, z)`        | 3-argument variant (sum of three squares, then sqrt) |
| `tax::erf(x)`                | `ErfExpr<E>` (maintains `H = exp(-x²)` alongside) |

## Examples

### Pythagorean identity

```cpp
auto x = tax::TE<5>::variable(0.7);
tax::TE<5> identity;
identity = (tax::square(tax::sin(x)) + tax::square(tax::cos(x))).eval();
// identity.value() ≈ 1.0; all higher-degree coefficients ≈ 0.
```

Same with the paired form, sharing buffers:

```cpp
auto p = tax::sincos(x);
tax::TE<5> identity;
identity = (tax::square(p.sin()) + tax::square(p.cos())).eval();
```

### Round-trips

```cpp
auto x = tax::TE<4>::variable(0.5);
tax::TE<4> back;
back = (tax::sin(tax::asin(x))).eval();   // ≈ x
back = (tax::log(tax::exp(x))).eval();    // ≈ x
back = (tax::cube(tax::cbrt(x))).eval();  // ≈ x
```

### Compile-time integer power vs. runtime real

```cpp
auto x = tax::TE<5>::variable(1.5);
tax::TE<5> a, b;
a = (tax::pow<3>(x)).eval();       // unrolled to MulExpr(SquareExpr(x), x)
b = (tax::pow(x, 3.0)).eval();     // PowRealExpr (general recurrence)
// a.coeffs() == b.coeffs() to floating-point precision.
```

### atan2 in all four quadrants

```cpp
auto [y, x] = tax::TEn<3, 2>::variables(std::array{1.0, -1.0});
tax::TEn<3, 2> theta;
theta = (tax::atan2(y, x)).eval();   // theta.value() == 3π/4
```

## Composing freely

Any of these compose, including nesting:

```cpp
auto a = (tax::erf(u + v) * tax::exp(-tax::square(u))).eval();
auto b = (tax::log10(1.0 + tax::hypot(u, v))).eval();
auto c = tax::atan2(tax::sin(u), tax::cos(v)).eval();
```

Each buffered node along the chain owns one `coeffs_` (and any
auxiliary buffer the recurrence requires); view-like nodes between
them allocate nothing.
