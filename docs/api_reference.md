# API Reference

## Core Type

### `TruncatedTaylorExpansionT<T, N, M>`

```cpp
template <typename T, int N, int M = 1>
class TruncatedTaylorExpansionT;
```

Materialized truncated Taylor polynomial in $M$ variables up to total degree $N$ with scalar type $T$. Coefficients are stored in a `std::array<T, nCoefficients>` in graded lexicographic order.

**Template Parameters:**

| Parameter | Constraint | Description                    |
|-----------|------------|--------------------------------|
| `T`       |            | Scalar coefficient type        |
| `N`       | $N \ge 0$  | Maximum total polynomial order |
| `M`       | $M \ge 1$  | Number of independent variables |

**Type Aliases:**

```cpp
template <int N>       using TE  = TruncatedTaylorExpansionT<double, N, 1>;
template <int N, int M> using TEn = TruncatedTaylorExpansionT<double, N, M>;
```

### Constants

| Member                  | Type          | Value                       |
|-------------------------|---------------|-----------------------------|
| `TruncatedTaylorExpansionT::nCoefficients`            | `std::size_t` | $\binom{N+M}{M}$           |
| `TruncatedTaylorExpansionT::Data`      | type alias    | `std::array<T, nCoefficients>`      |
| `TruncatedTaylorExpansionT::Input`       | type alias    | `std::array<T, M>`          |

---

## Constructors

```cpp
constexpr TruncatedTaylorExpansionT();                              // zero polynomial
explicit constexpr TruncatedTaylorExpansionT(Data c);        // from coefficient array
constexpr TruncatedTaylorExpansionT(T val);                         // constant polynomial
constexpr TruncatedTaylorExpansionT(const Expr<Derived, T, N, M>& expr);  // materialize expression
```

| Constructor        | Description                                                |
|--------------------|------------------------------------------------------------|
| Default            | All coefficients zero                                      |
| Coefficient array  | Direct initialization from a `std::array<T, nCoefficients>`       |
| Scalar             | Constant polynomial with `coeff[0] = val`, rest zero       |
| Expression         | Evaluates a lazy expression tree into a materialized `TruncatedTaylorExpansionT` |

---

## Variable Factories

### `variable(T x0)` (univariate)

```cpp
static constexpr TruncatedTaylorExpansionT variable(T x0) noexcept;   // requires M == 1
```

Creates $x_0 + \delta x$. The constant term is $x_0$ and the linear coefficient is $1$.

### `variable<I>(Input x0)` (indexed)

```cpp
template <int I>
static constexpr TruncatedTaylorExpansionT variable(const Input& x0) noexcept;
```

Creates variable $x_I$ expanded around $\mathbf{x}_0$. The constant term is $x_{0,I}$ and the coefficient of $\delta x_I$ is $1$; all other coefficients are zero.

### `variables(Input x0)`

```cpp
static constexpr auto variables(const Input& x0) noexcept;
```

Returns `std::tuple(x_0, ..., x_{M-1})` via structured bindings.

```cpp
auto [x, y, z] = TEn<3, 3>::variables({1.0, 2.0, 3.0});
```

### `constant(T v)` / `zero()` / `one()`

```cpp
static constexpr TruncatedTaylorExpansionT constant(T v) noexcept;
static constexpr TruncatedTaylorExpansionT zero() noexcept;
static constexpr TruncatedTaylorExpansionT one() noexcept;
```

---

## Coefficient Access

### `value()`

```cpp
constexpr T value() const noexcept;
```

Returns $f(\mathbf{x}_0) = f_0$, the constant term.

### `operator[](std::size_t i)`

```cpp
constexpr T  operator[](std::size_t i) const noexcept;
constexpr T& operator[](std::size_t i) noexcept;
```

Direct access to the coefficient at flat index `i` in grlex order.

### `coeffs()`

```cpp
constexpr Data& coeffs() noexcept;
constexpr const Data& coeffs() const noexcept;
```

Returns a reference to the full coefficient array (mutable or const overload).

### `coeff(alpha)` (runtime)

```cpp
constexpr T coeff(const MultiIndex<M>& alpha) const noexcept;
```

Returns $f_\alpha$, the coefficient of $\delta\mathbf{x}^\alpha$.

```cpp
f.coeff({2, 1});   // coefficient of δx²·δy
```

### `coeff<Alpha...>()` (compile-time)

```cpp
template <int... Alpha>
constexpr T coeff() const noexcept;
```

Compile-time version. `sizeof...(Alpha)` must equal $M$ and $\sum \alpha_i \le N$.

```cpp
f.coeff<2, 1>();   // same as f.coeff({2, 1})
```

---

## Derivative Access

### `derivative(alpha)` (runtime)

```cpp
constexpr T derivative(const MultiIndex<M>& alpha) const noexcept;
```

Returns the partial derivative $\frac{\partial^{|\alpha|} f}{\partial x_1^{\alpha_1} \cdots \partial x_M^{\alpha_M}}\big|_{\mathbf{x}_0}$, computed as $\alpha! \cdot f_\alpha$.

```cpp
f.derivative({2, 0});   // ∂²f/∂x²
f.derivative({1, 1});   // ∂²f/∂x∂y
```

### `derivative<Alpha...>()` (compile-time)

```cpp
template <int... Alpha>
constexpr T derivative() const noexcept;
```

Compile-time version.

### `derivatives()`

```cpp
constexpr Data derivatives() const noexcept;
```

Returns all partial derivatives in flat grlex order: entry `i` equals the derivative corresponding to monomial `i`.

---

## Evaluation

### `eval(T dx)` (univariate)

```cpp
constexpr T eval(T dx) const noexcept;   // requires M == 1
```

Evaluates the polynomial at $x_0 + \delta x$ using Horner's method:

$$f(x_0 + \delta x) = \sum_{d=0}^{N} f_d \, (\delta x)^d$$

### `eval(Input dx)` (multivariate)

```cpp
constexpr T eval(const Input& dx) const noexcept;
```

Evaluates the polynomial at $\mathbf{x}_0 + \delta\mathbf{x}$:

$$f(\mathbf{x}_0 + \delta\mathbf{x}) = \sum_{|\alpha| \le N} f_\alpha \, \delta\mathbf{x}^\alpha$$

---

## In-place Operators

```cpp
TruncatedTaylorExpansionT& operator+=(const TruncatedTaylorExpansionT& o);
TruncatedTaylorExpansionT& operator-=(const TruncatedTaylorExpansionT& o);
TruncatedTaylorExpansionT& operator+=(const Expr<Derived, T, N, M>& e);
TruncatedTaylorExpansionT& operator-=(const Expr<Derived, T, N, M>& e);
TruncatedTaylorExpansionT& operator*=(T s);          // scalar multiply
TruncatedTaylorExpansionT& operator/=(T s);          // scalar divide
TruncatedTaylorExpansionT& operator*=(const TruncatedTaylorExpansionT& o); // Cauchy product
TruncatedTaylorExpansionT& operator/=(const TruncatedTaylorExpansionT& o); // division via reciprocal
```

---

## Comparison Operators

All comparisons act on `value()` (the constant term):

```cpp
bool operator==(const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b);
bool operator!=(const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b);
bool operator< (const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b);
bool operator> (const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b);
bool operator<=(const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b);
bool operator>=(const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b);
```

Scalar comparisons (`TruncatedTaylorExpansionT` vs `T` and `T` vs `TruncatedTaylorExpansionT`) are also provided.

---

## Arithmetic Operators

All return lazy expression templates that are evaluated on assignment to a `TruncatedTaylorExpansionT`.

### DA--DA

```cpp
auto operator+(const Expr& a, const Expr& b);   // addition
auto operator-(const Expr& a, const Expr& b);   // subtraction
auto operator*(const Expr& a, const Expr& b);   // Cauchy product
auto operator/(const Expr& a, const Expr& b);   // division
auto operator-(const Expr& a);                     // negation
```

### DA--scalar and scalar--DA

```cpp
auto operator+(const Expr& a, T s);
auto operator+(T s, const Expr& a);
auto operator-(const Expr& a, T s);
auto operator-(T s, const Expr& a);
auto operator*(const Expr& a, T s);
auto operator*(T s, const Expr& a);
auto operator/(const Expr& a, T s);
auto operator/(T s, const Expr& a);
```

---

## Unary Math Functions

All functions live in the `tax` namespace and return lazy expressions.

| Function          | Signature                   | Description                  |
|-------------------|-----------------------------|------------------------------|
| `tax::abs(f)`     | $\lvert f \rvert$           | Absolute value               |
| `tax::square(f)`  | $f^2$                       | Square (Cauchy self-product) |
| `tax::cube(f)`    | $f^3$                       | Cube                         |
| `tax::sqrt(f)`    | $\sqrt{f}$                  | Square root ($f_0 > 0$)     |
| `tax::cbrt(f)`    | $\sqrt[3]{f}$               | Cubic root ($f_0 \ne 0$)    |
| `tax::sin(f)`     | $\sin(f)$                   | Sine                         |
| `tax::cos(f)`     | $\cos(f)$                   | Cosine                       |
| `tax::tan(f)`     | $\tan(f)$                   | Tangent                      |
| `tax::asin(f)`    | $\arcsin(f)$                | Inverse sine                 |
| `tax::acos(f)`    | $\arccos(f)$                | Inverse cosine               |
| `tax::atan(f)`    | $\arctan(f)$                | Inverse tangent              |
| `tax::sinh(f)`    | $\sinh(f)$                  | Hyperbolic sine              |
| `tax::cosh(f)`    | $\cosh(f)$                  | Hyperbolic cosine            |
| `tax::tanh(f)`    | $\tanh(f)$                  | Hyperbolic tangent           |
| `tax::asinh(f)`   | $\operatorname{asinh}(f)$   | Inverse hyperbolic sine      |
| `tax::acosh(f)`   | $\operatorname{acosh}(f)$   | Inverse hyperbolic cosine    |
| `tax::atanh(f)`   | $\operatorname{atanh}(f)$   | Inverse hyperbolic tangent   |
| `tax::exp(f)`     | $e^f$                       | Exponential                  |
| `tax::log(f)`     | $\ln(f)$                    | Natural logarithm ($f_0 > 0$) |
| `tax::log10(f)`   | $\log_{10}(f)$              | Base-10 logarithm            |
| `tax::erf(f)`     | $\operatorname{erf}(f)$     | Error function               |

---

## Binary / Ternary Math Functions

| Function               | Signature                                | Description                          |
|------------------------|------------------------------------------|--------------------------------------|
| `tax::ipow(f, n)`      | $f^n$, $n \in \mathbb{Z}$               | Integer power (binary exponentiation) |
| `tax::dpow(f, c)`      | $f^c$, $c \in \mathbb{R}$               | Real power                           |
| `tax::pow(f, n)`        | $f^n$                                    | Dispatches to `ipow`                |
| `tax::pow(f, c)`        | $f^c$                                    | Dispatches to `dpow`                |
| `tax::pow(f, g)`        | $f^g$                                    | DA power via $\exp(g \ln f)$         |
| `tax::atan2(y, x)`     | $\operatorname{atan2}(y, x)$            | Two-argument arctangent              |
| `tax::hypot(x, y)`     | $\sqrt{x^2 + y^2}$                      | Euclidean norm (2-arg)               |
| `tax::hypot(x, y, z)`  | $\sqrt{x^2 + y^2 + z^2}$                | Euclidean norm (3-arg)               |

---

## Streaming

```cpp
std::ostream& operator<<(std::ostream& os, const TruncatedTaylorExpansionT& a);
```

Prints the polynomial in human-readable form with Unicode superscripts and subscripts:

```
3 + 4·δx + δx² + O(δx⁴)
```
