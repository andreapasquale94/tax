# API Reference

Complete reference for the core **tax** types, constructors, accessors,
operators, and mathematical functions.

All names live in `namespace tax` unless noted; storage tags live in
`namespace tax::storage`.

---

## Core Type

```cpp
namespace tax {

template <typename T,
          typename Scheme,
          typename Storage = storage::Dense>
    requires IndexScheme<Scheme>
class TaylorExpansion;

}  // namespace tax
```

`TaylorExpansion` is parameterized by an `IndexScheme` that encodes the kept
monomial set. Two schemes ship today:

- **`IsotropicScheme<N, M>`** — classic total-degree-$\le N$ graded-lex layout
  over $M$ variables. This is the `TE<N,M>` / `STE<N,M>` form.
- **`MixedScheme<Group<Dim,Order>...>`** — anisotropic per-axis order caps
  (a product of per-group simplices). This is the `MixedTE<Group<Dim,Order>...>`
  form. See [Mixed-order expansions](../guide/mixed.md).

Per-operator signatures listed below use the `<T, N, M>` shorthand for the
isotropic form; they apply equally to any scheme via template deduction.

### Template parameters

| Parameter | Description |
|---|---|
| `T`       | Scalar coefficient type — must satisfy `tax::Scalar` (`std::floating_point`) |
| `Scheme`  | Index scheme — must satisfy `IndexScheme`; typically `IsotropicScheme<N,M>` or `MixedScheme<...>` |
| `Storage` | Storage policy: `tax::storage::Dense` (default) or `tax::storage::Sparse` |

### Convenience aliases

```cpp
template <int N, int M = 1>
using TE  = TaylorExpansion<double, IsotropicScheme<N,M>, storage::Dense>;

template <int N, int M = 1>
using STE = TaylorExpansion<double, IsotropicScheme<N,M>, storage::Sparse>;

template <typename... Groups>   // each Group = Group<Dim, Order>
using MixedTE = TaylorExpansion<double, MixedScheme<Groups...>, storage::Dense>;
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

Dense and Sparse expose the same shape. Sparse additionally exposes:

```cpp
[[nodiscard]] std::size_t nnz() const noexcept;                              // # nonzeros
[[nodiscard]] std::span<const storage::flat_index_t> support() const;        // flat indices
[[nodiscard]] std::span<const T>                     values()  const;        // values
[[nodiscard]] auto dense() const noexcept;                                   // → Dense conversion
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
`tax/eigen.hpp` — see [Eigen / API Reference](eigen.md).

---

## Arithmetic operators

All binary arithmetic operators are free functions defined for any combination
of TE × TE and TE × scalar. They are eager for Dense/Sparse and operate on
materialised polynomials (the lazy expression-template layer is invoked
internally via the kernels).

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
[Recurrence Relations](../internals/recurrences.md)).

| Function     | Domain restriction | Recurrence helper |
|---|---|---|
| `square(f)`  | none | $f \cdot f$ via Cauchy self-product |
| `cube(f)`    | none | two Cauchy products |
| `reciprocal(f)` | $f_0 \ne 0$ | solve $f \cdot g = 1$ |
| `sqrt(f)`    | $f_0 > 0$ | solve $g^2 = f$ |
| `cbrt(f)`    | $f_0 \ne 0$ | solve $g^3 = f$ with incremental $g^2$ |
| `sin(f)`     | none | coupled sin/cos |
| `cos(f)`     | none | coupled sin/cos |
| `tan(f)`     | $\cos(f_0) \ne 0$ | solve $\cos\cdot t = \sin$ |
| `asin(f)`    | $|f_0| < 1$ | via $h = \sqrt{1-f^2}$ |
| `acos(f)`    | $|f_0| < 1$ | $\pi/2 - \arcsin$ |
| `atan(f)`    | none | via $h = 1 + f^2$ |
| `sinh(f)`    | none | coupled sinh/cosh |
| `cosh(f)`    | none | coupled sinh/cosh |
| `tanh(f)`    | none | solve $\cosh\cdot t = \sinh$ |
| `asinh(f)`   | none | via $h = \sqrt{1+f^2}$ |
| `acosh(f)`   | $f_0 > 1$ | via $h = \sqrt{f^2-1}$ |
| `atanh(f)`   | $|f_0| < 1$ | via $h = 1-f^2$ |
| `exp(f)`     | none | derivative-driven |
| `log(f)`     | $f_0 > 0$ | derivative-driven |
| `erf(f)`     | none | via $h = \tfrac{2}{\sqrt\pi}\exp(-f^2)$ |

---

## Binary math functions

```cpp
// Integer power via binary exponentiation
[[nodiscard]] TaylorExpansion<T, N, M> pow(const TaylorExpansion<T, N, M>& f, int n);

// Real-exponent power, requires f_0 > 0
[[nodiscard]] TaylorExpansion<T, N, M> pow(const TaylorExpansion<T, N, M>& f, T c);

// Two-argument arctangent
[[nodiscard]] TaylorExpansion<T, N, M> atan2(const TaylorExpansion<T, N, M>& y,
                                              const TaylorExpansion<T, N, M>& x);
```

Power overloads exist for both Dense and Sparse storage.

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
