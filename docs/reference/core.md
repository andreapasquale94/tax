# API Reference

Complete reference for the core **tax** types, constructors, accessors,
operators, and mathematical functions.

All names live in `namespace tax` unless noted; the coefficient container
lives in `namespace tax::storage`.

---

## Core Type

```cpp
namespace tax {

template <typename T, typename Scheme>
    requires IndexScheme<Scheme>
class TaylorExpansion;

}  // namespace tax
```

`TaylorExpansion` is parameterized by an `IndexScheme` that encodes the kept
monomial set. Two schemes ship today:

- **`IsotropicScheme<N, M>`** — classic total-degree-$\le N$ graded-lex layout
  over $M$ variables. This is the `TE<N,M>` form.
- **`MixedScheme<Group<Dim,Order>...>`** — anisotropic per-axis order caps
  (a product of per-group simplices). This is the `MixedTE<Group<Dim,Order>...>`
  form. See [Named & Mixed-Order Expansions](../guide/named.md#anisotropic-axes-per-axis-orders).

Per-operator signatures listed below use the `<T, N, M>` shorthand for the
isotropic form; they apply equally to any scheme via template deduction.

### Template parameters

| Parameter | Description |
|---|---|
| `T`       | Scalar coefficient type — must satisfy `tax::Scalar` (`std::floating_point`) |
| `Scheme`  | Index scheme — must satisfy `IndexScheme`; typically `IsotropicScheme<N,M>` or `MixedScheme<...>` |

### Convenience aliases

```cpp
template <int N, int M = 1>
using TE  = TaylorExpansion<double, IsotropicScheme<N,M>>;

template <int N, int M>
using TEn = TaylorExpansion<double, IsotropicScheme<N,M>>;

template <typename... Groups>   // each Group = Group<Dim, Order>
using MixedTE = TaylorExpansion<double, MixedScheme<Groups...>>;
```

### Compile-time members

| Member | Type | Description |
|---|---|---|
| `nCoefficients` | `std::size_t` | Total monomials: $\binom{N+M}{M}$ |
| `order_v`       | `int` | Truncation order $N$ |
| `vars_v`        | `int` | Variable count $M$ |
| `scalar_type`   | type alias | `T` |
| `Input`         | type alias | `std::array<T, M>` — expansion-point / displacement vector |

---

## Constructors

```cpp
constexpr TaylorExpansion() noexcept;                       // zero polynomial
/*implicit*/ constexpr TaylorExpansion(T val) noexcept;     // constant polynomial
explicit constexpr TaylorExpansion(Data coeffs) noexcept;   // direct coefficient array
```

The constant-value constructor is implicit so scalars are promoted naturally in
arithmetic expressions.

---

## Variable factories

All factories are `static constexpr` member functions of `TaylorExpansion`.

```cpp
// Zero polynomial
[[nodiscard]] static constexpr TaylorExpansion zero() noexcept;

// Constant polynomial with value v
[[nodiscard]] static constexpr TaylorExpansion constant(T v) noexcept;

// Univariate variable: x = x0 + 1·δx   (requires M == 1)
[[nodiscard]] static constexpr TaylorExpansion variable(T x0) noexcept;

// I-th coordinate variable at expansion point p   (requires 0 ≤ I < M)
template <int I>
[[nodiscard]] static constexpr TaylorExpansion variable(const Input& p) noexcept;
```

For an Eigen column-vector of all $M$ coordinate variables at once, use the
free function `tax::variables<TE>(x0)` from
[Eigen integration](../guide/eigen.md).

---

## Coefficient access

```cpp
// Constant term, f(x0)
[[nodiscard]] constexpr T value() const noexcept;

// Raw coefficient by flat (graded-lex) index
[[nodiscard]] constexpr T  operator[](std::size_t k) const noexcept;
[[nodiscard]] constexpr T& operator[](std::size_t k) noexcept;          // Dense only

// Coefficient by runtime multi-index
[[nodiscard]] constexpr T coeff(const MultiIndex<M>& alpha) const noexcept;

// Coefficient by compile-time multi-index
//   Usage: f.coeff<2, 1>()  → coefficient of δx₁²·δx₂
template <int... Alpha>
[[nodiscard]] constexpr T coeff() const noexcept;
```

---

## Derivative access

Derivatives are related to coefficients by $\partial^\alpha f = \alpha! \cdot f_\alpha$.

```cpp
// Partial derivative at x0 by runtime multi-index
[[nodiscard]] constexpr T derivative(const MultiIndex<M>& alpha) const noexcept;

// Partial derivative at x0 by compile-time multi-index
//   Usage: f.derivative<2>()    → d²f/dx² (univariate)
//          f.derivative<1, 0>() → ∂f/∂x   (multivariate, M==2)
template <int... Alpha>
[[nodiscard]] constexpr T derivative() const noexcept;
```

---

## Differentiation and integration

These return new `TaylorExpansion` objects with the same shape $(N, M)$.

```cpp
// ∂/∂x_I   compile-time index
template <int I>
[[nodiscard]] constexpr TaylorExpansion deriv() const noexcept;

// ∂/∂x_var   runtime index — throws std::out_of_range if var >= M
[[nodiscard]] TaylorExpansion deriv(int var) const;

// Indefinite integral w.r.t. x_I   (constant of integration = 0)
//   Coefficients of degree N are dropped (truncation).
template <int I>
[[nodiscard]] constexpr TaylorExpansion integ() const noexcept;

// Runtime variant — throws std::out_of_range if var >= M
[[nodiscard]] TaylorExpansion integ(int var) const;
```

---

## Evaluation

```cpp
// Evaluate at displacement dx from the expansion point.
//   Univariate (M==1) uses Horner's method.
[[nodiscard]] constexpr T eval(const Input& dx) const noexcept;

// Eigen-vector displacement (size must equal M, statically or at runtime).
template <typename DxDerived>
[[nodiscard]] T eval(const Eigen::MatrixBase<DxDerived>& dx) const noexcept;
```

---

## Gradient and Hessian (member form)

For multivariate dense expansions, on-object convenience helpers are provided:

```cpp
// ∇f at x0 as an Eigen column vector
[[nodiscard]] Eigen::Matrix<T, M, 1> gradient() const noexcept;

// Hf at x0 as a symmetric Eigen matrix
[[nodiscard]] Eigen::Matrix<T, M, M> hessian() const noexcept;
```

Vector-valued counterparts (Jacobian of a vector function) live in
`tax/la.hpp` — see [Eigen / API Reference](eigen.md).

---

## Arithmetic operators

All binary arithmetic operators are free functions defined for any combination
of TE × TE and TE × scalar. They are eager: each operator materialises a
fresh `TaylorExpansion` by a single kernel pass
(there is no lazy expression-template layer).

```cpp
operator+(lhs, rhs);   // sum
operator-(lhs, rhs);   // difference
operator*(lhs, rhs);   // product (Cauchy convolution)
operator/(lhs, rhs);   // division (via reciprocal recurrence when rhs is TE)
operator-(x);          // unary negation
operator+(x);          // unary plus (identity)
```

In-place forms (`+=`, `-=`, `*=`, `/=`) are available for TE × TE and
TE × scalar.

---

## Comparison operators

`==`, `!=`, `<`, `>`, `<=`, `>=` compare the **constant terms** only. Supported
for TE × TE, TE × scalar, and scalar × TE. Useful when threading TE values
through Eigen factorisations and control-flow predicates.

---

## Unary math functions

All accept a `TaylorExpansion` and return a `TaylorExpansion` of the same
shape, using degree-by-degree recurrences (see
[Recurrence Relations](../internals/recurrences.md)). Only the pure-polynomial
functions (`square`, `cube`, `reciprocal`) are `constexpr` and usable in
constant evaluation. The transcendental and root functions seed their
constant term with a plain libm call (`std::exp`, `std::sin`, …) and are
therefore **runtime-only**.

| Function     | Domain restriction | `constexpr` | Recurrence helper |
|---|---|:-:|---|
| `square(f)`  | none | yes | $f \cdot f$ via Cauchy self-product |
| `cube(f)`    | none | yes | two Cauchy products |
| `reciprocal(f)` | $f_0 \ne 0$ | yes | solve $f \cdot g = 1$ |
| `sqrt(f)`    | $f_0 > 0$ | no | solve $g^2 = f$ |
| `cbrt(f)`    | $f_0 \ne 0$ | no | real-power recurrence at $1/3$, `cbrt` seed (handles $f_0 < 0$) |
| `sin(f)`     | none | no | coupled sin/cos |
| `cos(f)`     | none | no | coupled sin/cos |
| `tan(f)`     | $\cos(f_0) \ne 0$ | no | solve $\cos\cdot t = \sin$ |
| `asin(f)`    | $|f_0| < 1$ | no | `seriesDerivQuotient` with $h = \sqrt{1-f^2}$ |
| `acos(f)`    | $|f_0| < 1$ | no | `seriesDerivQuotient` (negative sign) with $h = \sqrt{1-f^2}$ |
| `atan(f)`    | none | no | `seriesDerivQuotient` with $h = 1 + f^2$ |
| `sinh(f)`    | none | no | shared $e^{f}/e^{-f}$ pair |
| `cosh(f)`    | none | no | shared $e^{f}/e^{-f}$ pair |
| `tanh(f)`    | none | no | solve $\cosh\cdot t = \sinh$ |
| `asinh(f)`   | none | no | `seriesDerivQuotient` with $h = \sqrt{1+f^2}$ |
| `acosh(f)`   | $f_0 > 1$ | no | `seriesDerivQuotient` with $h = \sqrt{f^2-1}$ |
| `atanh(f)`   | $|f_0| < 1$ | no | `seriesDerivQuotient` with $h = 1-f^2$ |
| `exp(f)`     | none | no | `seriesDerivProduct` with $h = g$ itself |
| `log(f)`     | $f_0 > 0$ | no | `seriesDerivQuotient` with $h = f$ |
| `erf(f)`     | none | no | `seriesDerivProduct` with $h = \tfrac{2}{\sqrt\pi}\exp(-f^2)$ |

`seriesDerivQuotient` / `seriesDerivProduct` are the two shared recurrence
drivers in `tax/kernels/algebra.hpp` — they are still marked `constexpr`
internally, but the transcendental kernels that use them seed their constant
term with a runtime libm call, so the end-user functions above are runtime-only.
See [Recurrence Relations](../internals/recurrences.md#shared-recurrence-drivers).

---

## Binary math functions

Only integer `pow(f, int n)` is `constexpr`. The real-exponent `pow`, the
Taylor-valued and scalar-base `pow`, `halfPow<K>`, `invSqrtPow<K>`, and
`atan2` seed with a libm call and are **runtime-only**.

```cpp
// Integer power via binary exponentiation — constexpr.
//   Requires f_0 != 0 only when n < 0 (reciprocal path).
[[nodiscard]] constexpr TaylorExpansion<T, N, M> pow(const TaylorExpansion<T, N, M>& f, int n);

// Compile-time integer power x^N — the exponent lives in the type. Forwards to
// the same integer Cauchy chain as pow(f, N); constexpr, no libm. Prefer this
// when the exponent is a constant.
template <int N>
[[nodiscard]] constexpr TaylorExpansion<T, N, M> pow(const TaylorExpansion<T, N, M>& f);

// Compile-time rational power x^(Num/Den). The exponent is reduced by gcd at
// compile time and bound to the cheapest kernel:
//   Den | Num          -> the integer chain    (constexpr; pow<6,3> == pow<2>);
//   reduced denom == 2 -> the sqrt/invsqrt chain (pow<3,2> == halfPow<3>,
//                                                  pow<-3,2> == invSqrtPow<3>);
//   otherwise          -> one real-exponent recurrence (pow<2,5> == x^(2/5)).
// Requires x_0 > 0 unless the reduced exponent is a non-negative integer.
template <int Num, int Den>
[[nodiscard]] constexpr TaylorExpansion<T, N, M> pow(const TaylorExpansion<T, N, M>& f);

// Real-exponent power. Requires f_0 != 0. Runtime-only.
[[nodiscard]] TaylorExpansion<T, N, M> pow(const TaylorExpansion<T, N, M>& f, P p);
                                          // P = any std::floating_point

// Taylor-valued exponent, f^g = exp(g·log(f)). Requires f_0 > 0. Runtime-only.
[[nodiscard]] TaylorExpansion<T, N, M> pow(const TaylorExpansion<T, N, M>& f,
                                           const TaylorExpansion<T, N, M>& g);

// Scalar base, Taylor exponent, s^g = exp(g·log(s)). Requires s > 0. Runtime-only.
[[nodiscard]] TaylorExpansion<T, N, M> pow(T s, const TaylorExpansion<T, N, M>& g);

// Compile-time-K half-integer power x^(K/2). Runtime-only.
//   Even K: integer-power chain — valid for x_0 < 0; requires x_0 != 0 only for K < 0.
//   Odd  K: one real-exponent recurrence — requires x_0 > 0.
template <int K>
[[nodiscard]] TaylorExpansion<T, N, M> halfPow(const TaylorExpansion<T, N, M>& x);

// Compile-time-K inverse square-root power x^(-K/2) = 1/sqrt(x)^K, K >= 1. Runtime-only.
//   Requires x_0 > 0. invSqrtPow<3>(r2) is the classic 1/r^3 of a squared radius.
template <int K>
[[nodiscard]] TaylorExpansion<T, N, M> invSqrtPow(const TaylorExpansion<T, N, M>& x);

// Two-argument arctangent (via r = y/x). Requires x_0 != 0. Runtime-only.
[[nodiscard]] TaylorExpansion<T, N, M> atan2(const TaylorExpansion<T, N, M>& y,
                                             const TaylorExpansion<T, N, M>& x);
// Constant-operand overloads promote the scalar to a flat expansion:
[[nodiscard]] TaylorExpansion<T, N, M> atan2(const TaylorExpansion<T, N, M>& y, T x);
[[nodiscard]] TaylorExpansion<T, N, M> atan2(T y, const TaylorExpansion<T, N, M>& x);
```

`halfPow<K>` is `pow<K, 2>` and `invSqrtPow<K>` is `pow<-K, 2>`; the general
`pow<Num, Den>` reduces to whichever of them (or the integer chain) fits. For a
genuine fractional exponent the recurrence is a single-pass `seriesPow` — which
is already optimal: it measures *faster* than a dedicated root (`cbrt`) or than
`sqrt`-then-integer-power, because `xᶜ` obeys the same degree-by-degree ODE
recurrence whatever `c` is, and only the scalar constant term needs a libm seed.
See [Guide / Fused Operations](../guide/fused.md).

---

## Fused pair functions

Coupled recurrences that produce two results in one pass (see
[Guide / Fused Operations](../guide/fused.md)). All are **runtime-only** (each
seeds its constant term with a libm call). Pair-returning functions order the
`std::pair` **as spelled in the name**.

| Function | Return type | Ordering / result | Domain restriction |
|---|---|---|---|
| `sinCos(x)`       | `std::pair<TE, TE>` | `{sin(x), cos(x)}` | none |
| `sinhCosh(x)`     | `std::pair<TE, TE>` | `{sinh(x), cosh(x)}` | none |
| `sqrtInvSqrt(x)`  | `std::pair<TE, TE>` | `{sqrt(x), 1/sqrt(x)}` | $x_0 > 0$ |
| `expSin(v, u)`    | `TE` | $e^{v}\sin u$ | none |
| `expCos(v, u)`    | `TE` | $e^{v}\cos u$ | none |
| `expSinCos(v, u)` | `std::pair<TE, TE>` | `{exp(v)*sin(u), exp(v)*cos(u)}` | none |

(`TE` above stands for the operand type `TaylorExpansion<T, Scheme>`; named
and mixed-order overloads are listed in the [Named API](named.md).)

`sqrtInvSqrt` is only worth calling when **both** outputs are consumed —
a single-output caller should use `sqrt`/`pow`/`invSqrtPow` instead.

---

## Basis conversion

Free functions in `namespace tax` (header `tax/core/basis/hermite.hpp` /
`chebyshev.hpp`), for dense, isotropic expansions only. See
[Guide / Extracting Results](../guide/results.md#basis-conversion) and
[Internals / Basis Conversion](../internals/basis-conversion.md).

```cpp
// Coefficients expressed in the (probabilists') Hermite / Chebyshev product
// basis instead of the monomial basis — same Coeffs<T,N,M> layout, distinct
// type so they can't be fed into monomial-basis arithmetic by mistake.
template <typename T, int N, int M>
struct HermiteCoefficients   { Coeffs<T,N,M> data; /* coeff(alpha), coeff<Alpha...>() */ };
template <typename T, int N, int M>
struct ChebyshevCoefficients { Coeffs<T,N,M> data; /* coeff(alpha), coeff<Alpha...>() */ };

template <typename T, int N, int M>
[[nodiscard]] HermiteCoefficients<T,N,M> toHermite(const TaylorExpansion<T, IsotropicScheme<N,M>, storage::Dense>& f) noexcept;
template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T, IsotropicScheme<N,M>, storage::Dense> fromHermite(const HermiteCoefficients<T,N,M>& h) noexcept;

template <typename T, int N, int M>
[[nodiscard]] ChebyshevCoefficients<T,N,M> toChebyshev(const TaylorExpansion<T, IsotropicScheme<N,M>, storage::Dense>& f) noexcept;
template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T, IsotropicScheme<N,M>, storage::Dense> fromChebyshev(const ChebyshevCoefficients<T,N,M>& c) noexcept;
```

---

## Streaming

```cpp
friend std::ostream& operator<<(std::ostream& os, const TaylorExpansion& a);
```

Outputs the polynomial in human-readable form: zero coefficients suppressed,
truncation remainder $\mathcal{O}(\delta\mathbf{x}^{N+1})$ appended.

---

## Eigen integration helpers

Free functions in `namespace tax` (see [Eigen / API Reference](eigen.md)):

```cpp
tax::variables<TE>(x0)               // Eigen column vector of M coordinate vars
tax::value(F)                        // extract constant terms
tax::eval(F, dx)                     // evaluate at displacement
tax::derivative<Alpha...>(F)         // compile-time partial of each entry
tax::gradient(f)                     // ∇f as Eigen vector
tax::hessian(f)                      // Hf as Eigen matrix
tax::jacobian(F)                     // Jacobian of a vector function
tax::invert(map)                     // local inverse of a polynomial map
```
