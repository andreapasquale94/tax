# API Reference

Complete reference for the core **tax** types, constructors, accessors, operators, and mathematical functions.

---

## Core Type

```cpp
namespace tax {

template <typename T, int N, int M = 1>
class TruncatedTaylorExpansionT;

}
```

### Template Parameters

| Parameter | Description |
|-----------|-------------|
| `T` | Scalar coefficient type (must satisfy `std::floating_point`, e.g. `double`, `float`) |
| `N` | Maximum total polynomial order (\(N \ge 0\)) |
| `M` | Number of independent variables (\(M \ge 1\), default 1) |

### Type Aliases

```cpp
template <int N>
using TE = TruncatedTaylorExpansionT<double, N, 1>;    // univariate, double

template <int N, int M>
using TEn = TruncatedTaylorExpansionT<double, N, M>;   // multivariate, double
```

### Constants and Member Types

| Name | Type | Description |
|------|------|-------------|
| `nCoefficients` | `static constexpr std::size_t` | Total number of stored coefficients: \(\binom{N+M}{M}\) |
| `Data` | `std::array<T, nCoefficients>` | Coefficient storage type |
| `Input` | `std::array<T, M>` | Expansion point / displacement vector type |

---

## Constructors

```cpp
// Zero polynomial (all coefficients = 0)
constexpr TruncatedTaylorExpansionT() noexcept;

// From full coefficient array
explicit constexpr TruncatedTaylorExpansionT(Data c) noexcept;

// Constant polynomial (value = val, all other coefficients = 0)
/*implicit*/ constexpr TruncatedTaylorExpansionT(T val) noexcept;

// Materialize an expression template
template <typename Derived>
/*implicit*/ constexpr TruncatedTaylorExpansionT(const Expr<Derived, T, N, M>& expr) noexcept;
```

The expression constructor evaluates the entire lazy expression tree in a single pass, writing the result directly into the coefficient array.

---

## Variable Factories

All factories are `static constexpr` member functions of `TruncatedTaylorExpansionT`.

```cpp
// Univariate variable: x = x0 + 1*dx (requires M == 1)
[[nodiscard]] static constexpr TruncatedTaylorExpansionT
variable(T x0) noexcept;

// Multivariate variable x_I at expansion point x0
template <int I>
[[nodiscard]] static constexpr TruncatedTaylorExpansionT
variable(const Input& x0) noexcept;

// All M variables at expansion point x0, returned as std::tuple
[[nodiscard]] static constexpr auto
variables(const Input& x0) noexcept;

// All M variables from splatted scalar arguments (requires M > 1)
template <typename... X0>
[[nodiscard]] static constexpr auto
variables(X0&&... x0) noexcept;

// Constant polynomial with value v
[[nodiscard]] static constexpr TruncatedTaylorExpansionT
constant(T v) noexcept;

// Zero polynomial
[[nodiscard]] static constexpr TruncatedTaylorExpansionT
zero() noexcept;

// Unit polynomial (constant = 1)
[[nodiscard]] static constexpr TruncatedTaylorExpansionT
one() noexcept;
```

---

## Coefficient Access

```cpp
// Constant term (coefficient of the degree-0 monomial)
[[nodiscard]] constexpr T value() const noexcept;

// Read/write coefficient by flat index
[[nodiscard]] constexpr T  operator[](std::size_t i) const noexcept;
[[nodiscard]] constexpr T& operator[](std::size_t i) noexcept;
[[nodiscard]] constexpr T  operator()(std::size_t i) const noexcept;
[[nodiscard]] constexpr T& operator()(std::size_t i) noexcept;

// Full coefficient array
[[nodiscard]] constexpr const Data& coeffs() const noexcept;
[[nodiscard]] constexpr Data&       coeffs() noexcept;

// Coefficient by runtime multi-index
[[nodiscard]] constexpr T coeff(const MultiIndex<M>& alpha) const noexcept;

// Coefficient by compile-time multi-index
template <int... Alpha>
[[nodiscard]] constexpr T coeff() const noexcept;
```

---

## Derivative Access

Derivatives are related to coefficients by \(\partial^\alpha f = \alpha! \cdot f_\alpha\).

```cpp
// Partial derivative by runtime multi-index
[[nodiscard]] constexpr T derivative(const MultiIndex<M>& alpha) const noexcept;

// Partial derivative by compile-time multi-index
template <int... Alpha>
[[nodiscard]] constexpr T derivative() const noexcept;

// All derivatives in flat monomial order (entry i = coeff[i] * alpha_i!)
[[nodiscard]] constexpr Data derivatives() const noexcept;
```

---

## Differentiation and Integration

These methods return new TTE objects representing the symbolic derivative or integral of the polynomial.

```cpp
// Partial derivative w.r.t. variable I (compile-time index)
template <int I>
[[nodiscard]] constexpr TruncatedTaylorExpansionT deriv() const noexcept;

// Partial derivative w.r.t. variable var (runtime index)
// Throws std::out_of_range if var >= M
[[nodiscard]] constexpr TruncatedTaylorExpansionT deriv(int var) const;

// Indefinite integral w.r.t. variable I (compile-time index)
// Terms of degree N are dropped; constant of integration is zero
template <int I>
[[nodiscard]] constexpr TruncatedTaylorExpansionT integ() const noexcept;

// Indefinite integral w.r.t. variable var (runtime index)
// Throws std::out_of_range if var >= M
[[nodiscard]] constexpr TruncatedTaylorExpansionT integ(int var) const;
```

---

## Norms

```cpp
// Infinity norm: max_i |c_i|
[[nodiscard]] constexpr T coeffsNormInf() const noexcept;

// p-norm (runtime): (sum |c_i|^p)^(1/p), p > 0
// Throws std::invalid_argument if p == 0
[[nodiscard]] T coeffsNorm(unsigned int p) const;

// p-norm (compile-time): P > 0
template <unsigned int P>
[[nodiscard]] T coeffsNorm() const noexcept;

// Grouped coefficient norm estimate with exponential fit
// var: 0 = group by total degree, 1..M = group by variable exponent
// type: 0 = max, 1 = sum, >1 = p-norm
// nc: maximum group index for returned estimates
[[nodiscard]] std::vector<T> coeffsNormEstimate(
    unsigned int var = 0, unsigned int type = 0, unsigned int nc = N) const;

// Convergence radius estimate from norm extrapolation
T radius(T eps, unsigned int type = 1) const;
```

---

## Evaluation

```cpp
// Univariate evaluation at x0 + dx (Horner's method, requires M == 1)
[[nodiscard]] constexpr T eval(T dx) const noexcept;

// Multivariate evaluation at x0 + dx
[[nodiscard]] constexpr T eval(const Input& dx) const noexcept;
```

---

## In-place Operators

```cpp
// TTE-TTE arithmetic
constexpr TruncatedTaylorExpansionT& operator+=(const TruncatedTaylorExpansionT& o) noexcept;
constexpr TruncatedTaylorExpansionT& operator-=(const TruncatedTaylorExpansionT& o) noexcept;
constexpr TruncatedTaylorExpansionT& operator*=(const TruncatedTaylorExpansionT& o) noexcept;
constexpr TruncatedTaylorExpansionT& operator/=(const TruncatedTaylorExpansionT& o) noexcept;

// TTE-expression arithmetic
template <typename Derived>
constexpr TruncatedTaylorExpansionT& operator+=(const Expr<Derived, T, N, M>& e) noexcept;
template <typename Derived>
constexpr TruncatedTaylorExpansionT& operator-=(const Expr<Derived, T, N, M>& e) noexcept;

// TTE-scalar arithmetic
constexpr TruncatedTaylorExpansionT& operator*=(T s) noexcept;
constexpr TruncatedTaylorExpansionT& operator/=(T s) noexcept;
```

Division (`/=`) with a TTE computes the reciprocal of the divisor (via the reciprocal recurrence), then multiplies.

---

## Comparison Operators

All comparison operators act on the **constant term** only. Supported for TTE vs TTE, TTE vs scalar, and scalar vs TTE:

```cpp
==  !=  <  >  <=  >=
```

---

## Arithmetic Operators (Free Functions)

All binary arithmetic operators return **lazy expression objects** that are evaluated only on assignment to a `TruncatedTaylorExpansionT`. Supported operand combinations: TTE-TTE, TTE-scalar, and scalar-TTE.

```cpp
auto operator+(lhs, rhs);   // sum (flattened N-ary for chains)
auto operator-(lhs, rhs);   // difference
auto operator*(lhs, rhs);   // product (flattened N-ary for chains)
auto operator/(lhs, rhs);   // division (via reciprocal)
auto operator-(expr);        // unary negation
auto operator+(expr);        // unary plus (identity)
```

---

## Unary Math Functions

All functions accept expression arguments and return lazy expression objects. Defined in namespace `tax`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `abs` | `abs(f)` | Absolute value (\(f_0 \ne 0\)) |
| `square` | `square(f)` | \(f^2\) via Cauchy self-product |
| `cube` | `cube(f)` | \(f^3\) via two Cauchy products |
| `sqrt` | `sqrt(f)` | Square root (\(f_0 > 0\)) |
| `cbrt` | `cbrt(f)` | Cubic root (\(f_0 \ne 0\)) |
| `sin` | `sin(f)` | Sine (coupled sin/cos recurrence) |
| `cos` | `cos(f)` | Cosine (coupled sin/cos recurrence) |
| `tan` | `tan(f)` | Tangent (solves \(\cos \cdot t = \sin\)) |
| `asin` | `asin(f)` | Arcsine via \(h = \sqrt{1-f^2}\) |
| `acos` | `acos(f)` | Arccosine (\(\pi/2 - \arcsin\)) |
| `atan` | `atan(f)` | Arctangent via \(h = 1 + f^2\) |
| `sinh` | `sinh(f)` | Hyperbolic sine (coupled recurrence) |
| `cosh` | `cosh(f)` | Hyperbolic cosine (coupled recurrence) |
| `tanh` | `tanh(f)` | Hyperbolic tangent (solves \(\cosh \cdot t = \sinh\)) |
| `asinh` | `asinh(f)` | Inverse hyperbolic sine via \(h = \sqrt{1+f^2}\) |
| `acosh` | `acosh(f)` | Inverse hyperbolic cosine via \(h = \sqrt{f^2-1}\) |
| `atanh` | `atanh(f)` | Inverse hyperbolic tangent via \(h = 1-f^2\) |
| `exp` | `exp(f)` | Exponential |
| `log` | `log(f)` | Natural logarithm (\(f_0 > 0\)) |
| `log10` | `log10(f)` | Common logarithm (\(\ln(f)/\ln(10)\)) |
| `erf` | `erf(f)` | Error function via \(h = \frac{2}{\sqrt{\pi}} e^{-f^2}\) |

---

## Binary and Ternary Math Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `ipow` | `ipow(f, int n)` | Integer power via binary exponentiation |
| `dpow` | `dpow(f, T c)` | Real-exponent power (\(f_0 > 0\)) |
| `tpow` | `tpow(f, g)` | DA power: \(\exp(g \cdot \ln f)\) |
| `pow` | `pow(f, int n)` | Dispatches to `ipow` |
| `pow` | `pow(f, T c)` | Dispatches to `dpow` |
| `pow` | `pow(f, g)` | Dispatches to `tpow` |
| `atan2` | `atan2(y, x)` | Two-argument arctangent |
| `hypot` | `hypot(x, y)` | \(\sqrt{x^2 + y^2}\) |
| `hypot` | `hypot(x, y, z)` | \(\sqrt{x^2 + y^2 + z^2}\) |

---

## Streaming

```cpp
friend std::ostream& operator<<(std::ostream& os, const TruncatedTaylorExpansionT& a);
```

Outputs the polynomial in human-readable form with Unicode superscripts for powers and subscripts for variable indices. Zero coefficients are suppressed. The truncation remainder \(\mathcal{O}(\delta\mathbf{x}^{N+1})\) is appended.
