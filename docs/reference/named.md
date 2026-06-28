# Named API (`tax::named`)

Complete reference for the **named-axis** layer: types, factories, member
operations, composition rules, and Eigen helpers.

The names are implemented in `namespace tax::named`, but the entire public API
is **re-exported under `tax`** and that is the spelling you should use:
`tax::NE`, `tax::variable`, `tax::variables`, `tax::Axis`,
`tax::NamedExpansion`, and the Eigen helpers `tax::gradient`,
`tax::hessian`, `tax::jacobian`, `tax::value`, `tax::eval` are all reachable
directly from `<tax/tax.hpp>`.

For the narrative how-to, see the [Named Expansions guide](../guide/named.md).

---

## Named type

```cpp
namespace tax::named {

template <typename T, typename Basis, int N, typename... Axes>
    requires Scalar<T> && BasisPolicy<Basis>
class NamedExpansion;

}  // namespace tax::named
```

A `NamedExpansion` wraps a dense `Expansion<T, Basis, IsotropicScheme<N, M>>` and
attaches a compile-time list of **named axes** to it. It is **basis-generic**: the
underlying polynomial family is the `Basis` parameter (`TaylorBasis` for the terse
`NE` alias). Each axis is a contiguous block of the underlying $M$ variables
identified by a compile-time string.

### Template parameters

| Parameter | Description |
|---|---|
| `T`       | Scalar coefficient type — must satisfy `tax::Scalar` |
| `Basis`   | Polynomial family — must satisfy `tax::BasisPolicy` (e.g. `TaylorBasis`) |
| `N`       | Maximum total polynomial order, $N \ge 0$ |
| `Axes...` | A pack of `Axis<Name, Dim>` types, **canonically ordered** (sorted by name, unique) |

The axis list is kept in canonical order, so the type does **not** depend on the
order you wrote the operands: `x * p` and `p * x` produce the *same* type. Build
named types through the factories or composition rather than spelling the axis
pack by hand (a `static_assert` enforces canonical order).

### Compile-time members

| Member | Type | Description |
|---|---|---|
| `vars_v`       | `int` | Total underlying variables (sum of axis dimensions) |
| `order_v`      | `int` | Truncation order $N$ |
| `scalar_type`  | type alias | `T` |
| `basis`        | type alias | `Basis` |
| `axis_list`    | type alias | Internal `TypeList<Axes...>` |
| `Inner`        | type alias | Underlying `Expansion<T, Basis, IsotropicScheme<N, vars_v>, storage::Dense>` |
| `Input`        | type alias | `Inner::Input` — expansion-point / displacement vector |
| `nCoefficients`| `std::size_t` | `Inner::nCoefficients` |

### Convenience alias

```cpp
template <int N, typename... Axes>
using NE = NamedExpansion<double, TaylorBasis, N, Axes...>;   // double-valued, Taylor basis
```

A `NamedTaylorExpansion<T, N, Axes...>` alias (`= NamedExpansion<T, TaylorBasis,
N, Axes...>`) is also provided for the explicit-`T` Taylor case.

e.g. `tax::NE<4, tax::Axis<"x", 3>>`.

---

## Axis

```cpp
template <FixedString Name, int Dim>
struct Axis {
    static constexpr auto name = Name;
    static constexpr int  dim  = Dim;     // Dim >= 1
};
```

A named axis: the compile-time string `Name` labels a block of `Dim ≥ 1`
consecutive variables of the underlying expansion.

```cpp
using PosX = tax::Axis<"x", 3>;   // a 3-D axis called "x"
using Time = tax::Axis<"t", 1>;   // a scalar axis called "t"
```

---

## FixedString

```cpp
template <std::size_t K>
struct FixedString {
    char data[K]{};
    constexpr FixedString(const char (&s)[K]) noexcept;     // implicit
    [[nodiscard]] static constexpr std::size_t size() noexcept;   // K - 1
    [[nodiscard]] constexpr char operator[](std::size_t i) const noexcept;
};
```

A null-terminated, structural compile-time string usable as a non-type template
parameter — this is what lets `Axis<"x", 3>` and `variable<"x", N>(...)` take a
string literal as a template argument.

---

## Variable factories

Free functions in `namespace tax::named` (re-exported as `tax::variable` /
`tax::variables`).

```cpp
// Single coordinate of a 1-D named axis: returns one NamedExpansion
//   over Axis<Name, 1>, expanded about x0. Basis defaults to TaylorBasis.
template <FixedString Name, int N, typename Basis = TaylorBasis, typename T>
    requires Scalar<T>
[[nodiscard]] constexpr auto variable(T x0) noexcept;

// The D coordinate variables of a single named axis Name:
//   returns std::array<NamedExpansion<T, Basis, N, Axis<Name, D>>, D>.
template <FixedString Name, int N, typename Basis = TaylorBasis, typename T, std::size_t D>
[[nodiscard]] constexpr auto variables(const std::array<T, D>& x0) noexcept;
```

```cpp
auto t = tax::variable<"t", 6>(0.0);             // NE<6, Axis<"t",1>>

std::array<double, 3> x0{1.0, 2.0, 3.0};
auto x = tax::variables<"x", 6>(x0);             // std::array<NE<6,Axis<"x",3>>, 3>
```

### Eigen overload

Declared in `<tax/la/named.hpp>` (pulled in by `<tax/tax.hpp>`), reachable as
`tax::variables`:

```cpp
// Build the D coordinate variables of axis Name from an Eigen vector
//   expansion point; returns Eigen::Matrix<NamedExpansion<T,Basis,N,Axis<Name,D>>, D, 1>.
template <FixedString Name, int N, typename Basis = TaylorBasis, typename Derived>
[[nodiscard]] auto variables(const Eigen::MatrixBase<Derived>& x0);
```

```cpp
Eigen::Vector3d v0{1.0, 2.0, 3.0};
auto xv = tax::variables<"x", 6>(v0);            // Eigen vector of named variables
```

The expansion point must have a compile-time size.

---

## Coordinate-variable member factory

```cpp
// I-th coordinate variable of the joint variable space at p   (0 <= I < vars_v)
template <int I>
[[nodiscard]] static constexpr NamedExpansion variable(const Input& p) noexcept;
```

Equivalent to `Inner::variable<I>(p)` lifted into the named type — used
internally by the free `variable`/`variables` factories.

---

## Access

```cpp
[[nodiscard]] constexpr T value() const noexcept;            // constant term

[[nodiscard]] constexpr const Inner& inner() const noexcept; // underlying expansion
[[nodiscard]] constexpr       Inner& inner()       noexcept;
```

---

## Embedding and slicing

```cpp
// Embed into a target named type R whose axes are a superset of these axes.
//   Each monomial is remapped; absent axes get zero exponents. Value-preserving.
template <typename R>
[[nodiscard]] constexpr R embed() const noexcept;

// Project onto the subset of axes named by Names...:
//   keeps only monomials whose exponents on the dropped axes are all zero
//   (i.e. restricts each dropped axis to its expansion point).
//   The result type carries exactly the requested axes (canonicalised).
template <FixedString... Names>
[[nodiscard]] constexpr auto slice() const noexcept;
```

```cpp
auto h = f.slice<"x">();        // drop every axis except "x"
```

`embed<R>()` requires `R`'s axis set to be a superset (otherwise a hard
`static_assert`); `slice<Names...>()` requires every requested name to exist.

---

## Per-axis differentiation and integration

```cpp
// Partial derivative w.r.t. coordinate Local of named axis Name (axis set preserved).
template <FixedString Name, int Local = 0>
[[nodiscard]] constexpr NamedExpansion deriv() const noexcept;

// Indefinite integral w.r.t. coordinate Local of named axis Name (axis set preserved,
//   order stays N; degree-N terms are dropped, matching Expansion::integ).
template <FixedString Name, int Local = 0>
[[nodiscard]] constexpr NamedExpansion integ() const noexcept;
```

Composing `deriv` with `slice` yields the "sub-derivative" projection:
`f.deriv<"p">().slice<"x">()`.

---

## Implicit promotion

```cpp
// Promote from an expansion over a *subset* of these axes (value-preserving).
template <typename... B>
    requires(/* TypeList<B...> is a proper subset of axis_list */)
/*implicit*/ constexpr NamedExpansion(const NamedExpansion<T, Basis, N, B...>& other) noexcept;
```

A value depending on **fewer** axes promotes implicitly into a wider axis set
(absent axes get zero derivatives), so a narrow expansion can be passed where a
wider one is expected — no manual padding.

---

## Composition operators

Binary arithmetic between expansions over **different** axis sets runs in the
**union** of the two sets: both operands are embedded into the union first, then
the dense kernels do the work. The result type carries the union of axes.

```cpp
template <typename T, int N, typename... A, typename... B>
[[nodiscard]] constexpr auto operator+(const NamedTaylorExpansion<T, N, A...>& a,
                                       const NamedTaylorExpansion<T, N, B...>& b) noexcept;
// likewise operator-, operator*, operator/
```

```cpp
auto x = tax::variable<"x", 4>(1.0);   // axes {x}
auto y = tax::variable<"y", 4>(2.0);   // axes {y}
auto g = x * x + x * y + y * y;        // axes {x, y}
```

Scalar combinations (`+`, `-`, `*`, `/` with a `T`, either side) and unary
negation are provided and leave the axis set unchanged. When the same name
appears on both operands, the dimensions must match (a `static_assert`
otherwise).

---

## Math functions

All accept a `NamedTaylorExpansion` and return one with the **same** axis set
(forwarded to the corresponding `tax::` series function on the inner expansion):

```
square  cube  sqrt  cbrt  reciprocal  exp  log
sin  cos  tan  asin  acos  atan  sinh  cosh  tanh  asinh  acosh  atanh  erf
```

### Binary math functions

```cpp
// x^n, integer exponent (axis set preserved)
template <typename T, int N, typename... A>
[[nodiscard]] NamedTaylorExpansion<T, N, A...> pow(const NamedTaylorExpansion<T, N, A...>& x, int n) noexcept;

// x^p, real exponent (axis set preserved; requires x.value() != 0)
template <typename T, int N, typename... A>
[[nodiscard]] NamedTaylorExpansion<T, N, A...> pow(const NamedTaylorExpansion<T, N, A...>& x, T p) noexcept;

// atan2(y, x) over the union of the two operands' axis sets
template <typename T, int N, typename... A, typename... B>
[[nodiscard]] auto atan2(const NamedTaylorExpansion<T, N, A...>& y,
                         const NamedTaylorExpansion<T, N, B...>& x) noexcept;
```

---

## Eigen integration helpers

Declared in `<tax/la/named.hpp>`; reachable in `namespace tax::named` and
re-exported under `tax`. A `NumTraits` specialisation lets named expansions act
as first-class Eigen scalars (so `Eigen::Matrix<NE<...>, D, 1>` works and can be
integrated as an ODE state).

```cpp
// Gradient w.r.t. one named axis → Eigen::Matrix<T, dim, 1>
template <FixedString Name, typename T, int N, typename... Axes>
[[nodiscard]] auto gradient(const NamedTaylorExpansion<T, N, Axes...>& f) noexcept;

// Hessian restricted to one named axis → Eigen::Matrix<T, dim, dim>
template <FixedString Name, typename T, int N, typename... Axes>
[[nodiscard]] auto hessian(const NamedTaylorExpansion<T, N, Axes...>& f) noexcept;

// Jacobian of an Eigen vector of named expansions w.r.t. one named axis
//   → Eigen::Matrix<T, K, dim>, J(i, j) = dF_i / dx_j
template <FixedString Name, typename Derived>
[[nodiscard]] auto jacobian(const Eigen::MatrixBase<Derived>& F);
```

```cpp
auto gx = tax::gradient<"x">(f);   // gradient w.r.t. axis "x"
auto Hx = tax::hessian<"x">(f);    // Hessian  w.r.t. axis "x"
auto Jx = tax::jacobian<"x">(F);   // Jacobian of an Eigen vector F w.r.t. "x"
```

`value` and `eval` overloads mirror `tax::la` for named scalars and Eigen
vectors of named expansions:

```cpp
template <typename T, int N, typename... Axes>
[[nodiscard]] T value(const NamedTaylorExpansion<T, N, Axes...>& f) noexcept;

template <typename Derived>
[[nodiscard]] auto value(const Eigen::MatrixBase<Derived>& F);   // requires named scalar

template <typename T, int N, typename... Axes, typename DxDerived>
[[nodiscard]] T eval(const NamedTaylorExpansion<T, N, Axes...>& f,
                     const Eigen::MatrixBase<DxDerived>& dx);

template <typename Derived, typename DxDerived>
[[nodiscard]] auto eval(const Eigen::MatrixBase<Derived>& F,
                        const Eigen::MatrixBase<DxDerived>& dx);   // requires named scalar
```

---

## Headers

| Header | Contents |
|---|---|
| `tax/expansion/named.hpp` | `NamedExpansion`, `Axis`, `FixedString`, `NE`, `variable`/`variables`, embed/slice/deriv/integ, composition + math |
| `tax/la/named.hpp`   | `NumTraits` for named expansions, Eigen `variables` overload, `gradient`/`hessian`/`jacobian`/`value`/`eval` by axis name |

Both are pulled in by the umbrella `<tax/tax.hpp>`.
