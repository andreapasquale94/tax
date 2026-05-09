# API reference

Public symbols, listed by component. All live in `namespace tax`.

## Storage types

### `TruncatedTaylorExpansionT<T, Order, Vars>`

Static-extent TTE. `T` is a floating-point scalar; `Order >= 0`,
`Vars >= 1`.

```cpp
template <class T, std::size_t Order, std::size_t Vars>
class TruncatedTaylorExpansionT;
```

| Constexpr member | Value |
|---|---|
| `kStatic` | `true` |
| `kOrder`  | `Order` |
| `kVars`   | `Vars` |
| `kSize`   | `monomialCount(Order, Vars) = C(Order+Vars, Vars)` |

| Type alias | Definition |
|---|---|
| `Scalar`   | `T` |
| `Coeffs`   | `Eigen::Matrix<T, kSize, 1>` |

| Static factory | Returns |
|---|---|
| `zero()`              | TTE with all coefficients zero |
| `one()`               | TTE with constant 1, rest zero |
| `constant(T c)`       | TTE with constant c, rest zero |
| `variable(T x0)` *(M=1 only)*           | `x = x0 + dx` |
| `variables(const Vec& x0)`              | `std::tuple<TTE, …>` of M variables |
| `variables(args...)` *(M args)*         | same, accepting M scalars |

| Instance method | Returns |
|---|---|
| `order()` / `nvars()`               | constexpr sizes |
| `value()`                            | `coeffs(0)` |
| `coeff(std::span<const size_t>)`     | raw Taylor coefficient |
| `coeff(const std::array&)`           | array overload |
| `coeff<size_t... Alpha>()`           | compile-time index |
| `derivative(span)` / `(array)` / `<>()`     | `α! · coeff(α)` |
| `eval(const Vec&)` / `(const array&)`       | truncated polynomial at `dx` |
| `coeffsNormInf()`, `coeffsNorm<P>()`        | buffer norms |
| `data()`, `coeffs()`, `rawCoeff(i)`, `setRawCoeff(i, v)` | raw access |
| `slice(d)`                           | Eigen `VectorBlock` for degree d |
| `advanceTo(d)`                       | no-op (storage is fully populated) |
| `operator<<=(Expr&&)`                | streaming assignment from any `StreamingExpression` |

| Alias | Expansion |
|---|---|
| `tax::TE<N>`        | `TruncatedTaylorExpansionT<double, N, 1>` |
| `tax::TEn<N, M>`    | `TruncatedTaylorExpansionT<double, N, M>` |

### `DynamicTaylorExpansion<T>`

Runtime-sized TTE. Coefficients live in `Eigen::VectorX<T>`.

```cpp
template <class T>
class DynamicTaylorExpansion;
```

| Constexpr member | Value |
|---|---|
| `kStatic` | `false` |
| `kOrder`, `kVars` | `0` (placeholders; runtime `order_` / `nvars_` are authoritative) |

| Static factory | Returns |
|---|---|
| `zero(order, nvars)` |  |
| `one(order, nvars)` |  |
| `constant(c, order, nvars)` |  |
| `variable(x0, order, nvars, var_idx)` | seeds dx at index `var_idx` |
| `variables(const std::vector<T>& x0, order)` | `std::vector<DynTE>` of `len(x0)` variables |

The instance methods mirror the static path. `coeff` / `derivative`
accept `std::span<const size_t>` directly; the array and
template-parameter forms are static-only (the multi-index has to be
known at compile time).

| Alias | Expansion |
|---|---|
| `tax::DynTE<T = double>` | `DynamicTaylorExpansion<T>` |

## Concepts

```cpp
template <class T>
concept Scalar = std::floating_point<T>;

template <class E>
concept TaxExpression = /* see expr/base.hpp */;

template <class L, class R>
concept SameKindExpression =
    TaxExpression<L> && TaxExpression<R>
    && L::kStatic == R::kStatic
    && std::is_same_v<typename L::Scalar, typename R::Scalar>;

template <class E>
concept TaylorExpansion = /* runtime-side TTE concept; see concepts.hpp */;

template <class E>
concept StreamingExpression = /* requires advanceTo(d) and slice(d) */;
```

## Operators

All operators are constrained on `TaxExpression` /
`SameKindExpression`. They return ET nodes; results are materialised
on `<<=`.

```cpp
namespace tax {

// arithmetic
auto operator+(const L&, const R&);
auto operator-(const L&, const R&);
auto operator*(const L&, const R&);
auto operator/(const L&, const R&);
auto operator-(const E&);

// scalar variants
auto operator+(const E&, Scalar c);
auto operator+(Scalar c, const E&);
auto operator-(const E&, Scalar c);
auto operator-(Scalar c, const E&);
auto operator*(const E&, Scalar c);
auto operator*(Scalar c, const E&);
auto operator/(const E&, Scalar c);   // = e * (1/c)

// math
auto sin(const E&);
auto cos(const E&);
auto tan(const E&);     // sin / cos
auto sinh(const E&);
auto cosh(const E&);
auto tanh(const E&);    // sinh / cosh
auto exp(const E&);
auto log(const E&);
auto sqrt(const E&);
auto square(const E&);
auto cube(const E&);    // square(e) * e

}
```

## ET nodes

User code rarely names these directly — the operators above produce
them. They are spelled out here because they appear in error messages.

| Node (in `tax::expr`) | Kind | Buffer |
|---|---|---|
| `AddExpr<L, R>`        | view | none |
| `SubExpr<L, R>`        | view | none |
| `NegExpr<E>`           | view | none |
| `ScalarMulExpr<E>`     | view | none |
| `ScalarAddExpr<E>`     | view | none |
| `MulExpr<L, R>`        | buffered | `coeffs_` |
| `DivExpr<L, R>`        | buffered | `coeffs_` |
| `SquareExpr<E>`        | buffered | `coeffs_` |
| `SqrtExpr<E>`          | buffered | `coeffs_` |
| `ExpExpr<E>`           | buffered | `coeffs_` |
| `LogExpr<E>`           | buffered | `coeffs_` |
| `SinCosExpr<E, ReturnSin>` | buffered | `sin_`, `cos_` (paired) |
| `SinhCoshExpr<E, ReturnSinh>` | buffered | `sinh_`, `cosh_` (paired) |

Aliases `SinExpr`, `CosExpr`, `SinhExpr`, `CoshExpr` instantiate the
paired classes with the right `ReturnSin` / `ReturnSinh`.

## Multi-index utilities (`tax::util`)

```cpp
constexpr std::size_t monomialCount(std::size_t order, std::size_t nvars) noexcept;
constexpr std::size_t degreeSize(std::size_t d, std::size_t nvars) noexcept;
constexpr std::size_t degreeOffset(std::size_t d, std::size_t nvars) noexcept;
constexpr std::size_t totalDegree(std::span<const std::size_t> alpha) noexcept;
constexpr std::size_t factorial(std::span<const std::size_t> alpha) noexcept;
constexpr std::size_t flatIndex(std::span<const std::size_t> alpha) noexcept;
constexpr std::size_t flatIndexWithinDegree(std::span<const std::size_t> alpha) noexcept;
constexpr void unflatIndex(std::size_t idx, std::span<std::size_t> out) noexcept;

template <class F>
inline void forEachMultiIndexOfDegree(std::size_t degree, std::size_t nvars, F&& f);
```

## Coefficient kernels (`tax::kernels`)

Kernels are slice-aware: they take operand objects (anything with
`slice(d)`) and write into an `Eigen::VectorBlock` or a slice-providing
output object.

| Kernel | Mathematical operation |
|---|---|
| `cauchyAccumulateSlice`   | `out[α] += scale · a[β] · b[γ]`, β+γ=α |
| `cauchyMulComputeDegree`  | `(a · b)_d` |
| `squareComputeDegree`     | `(a²)_d` |
| `divComputeDegree`        | `(a / b)_d` |
| `reciprocalComputeDegree` | `(1 / b)_d` |
| `sqrtComputeDegree`       | `sqrt(u)_d` |
| `expComputeDegree`        | `exp(u)_d` |
| `logComputeDegree`        | `log(u)_d` |
| `sinCosComputeDegree`     | `(sin(u))_d` and `(cos(u))_d` |
| `sinhCoshComputeDegree`   | `(sinh(u))_d` and `(cosh(u))_d` |

See [Architecture](architecture.md) for the recurrence relations.

## Headers

```cpp
#include <tax/tax.hpp>           // umbrella — pulls in everything below
```

Sub-headers exist for fine-grained inclusion:

```cpp
#include <tax/concepts.hpp>
#include <tax/fwd.hpp>
#include <tax/util/binomial.hpp>
#include <tax/util/multi_index.hpp>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/elementary.hpp>
#include <tax/kernels/exp_log.hpp>
#include <tax/kernels/trig.hpp>
#include <tax/storage/static_tte.hpp>
#include <tax/storage/dynamic_tte.hpp>
#include <tax/expr/base.hpp>
#include <tax/expr/view_nodes.hpp>
#include <tax/expr/buffered_nodes.hpp>
#include <tax/ops/arithmetic.hpp>
#include <tax/ops/assign.hpp>
#include <tax/ops/math.hpp>
```

Day-to-day, `<tax/tax.hpp>` is enough.
