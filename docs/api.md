# API reference

Public symbols, listed by component. All live in `namespace tax`.

## Storage type

### `TaylorExpansionT<T, Order, Vars>`

Single template that handles both compile-time and runtime sizes —
modelled on `Eigen::Matrix<T, Rows, Cols>` with `Eigen::Dynamic` (= -1)
as the sentinel.

```cpp
template <class T, int Order, int Vars>
class TaylorExpansionT;
```

Valid configurations: both `Order` and `Vars` are non-negative
integers (static) or both equal `Eigen::Dynamic` (dynamic). Mixed
dynamism is rejected with a `static_assert`.

| Constexpr member | Value |
|---|---|
| `OrderAtCompileTime` | `Order` (template arg, possibly `Eigen::Dynamic`) |
| `VarsAtCompileTime`  | `Vars`  (template arg, possibly `Eigen::Dynamic`) |
| `IsStatic`           | `(Order != Eigen::Dynamic) && (Vars != Eigen::Dynamic)` |
| `IsDynamic`          | `!IsStatic` |

| Type alias | Definition |
|---|---|
| `Scalar`   | `T` |
| `Coeffs`   | `Eigen::Matrix<T, monomialCount(Order, Vars), 1>` (static) or `Eigen::Matrix<T, Eigen::Dynamic, 1>` (dynamic) |

| Aliases (all in `namespace tax`) | Expansion |
|---|---|
| `TE<N>`        | `TaylorExpansionT<double, N, 1>` |
| `TEn<N, M>`    | `TaylorExpansionT<double, N, M>` |
| `DynTE<T = double>` | `TaylorExpansionT<T, Eigen::Dynamic, Eigen::Dynamic>` |

#### Static-only factories *(require `IsStatic`)*

| Factory | Returns |
|---|---|
| `zero()`           | TTE with all coefficients zero |
| `one()`            | TTE with constant 1, rest zero |
| `constant(T c)`    | TTE with constant c, rest zero |
| `variable(T x0)` *(requires `Vars == 1`)* | `x = x0 + dx` |
| `variables(const Vec& x0)`         | `std::tuple<TTE, …>` of `Vars` variables |
| `variables(args...)` *(`Vars` args)*| same, accepting `Vars` scalars |

#### Dynamic-only factories *(require `IsDynamic`)*

| Factory | Returns |
|---|---|
| `zero(order, nvars)` |  |
| `one(order, nvars)`  |  |
| `constant(c, order, nvars)` |  |
| `variable(x0, order, nvars, var_idx)` | seeds dx at index `var_idx` |
| `variables(const std::vector<T>& x0, order)` | `std::vector<DynTE>` of `len(x0)` variables |

#### Instance methods (both paths)

| Method | Returns |
|---|---|
| `order()` / `nvars()`                | sizes (constexpr in static path, runtime in dynamic) |
| `value()`                            | `coeffs(0)` |
| `coeff(std::span<const size_t>)`     | raw Taylor coefficient |
| `coeff(const std::array&)` *(static)*| array overload |
| `coeff<size_t... Alpha>()` *(static)*| compile-time index |
| `derivative(span)` / `(array)` / `<>()` | `α! · coeff(α)` |
| `eval(const Vec&)` / `(const array&)` | truncated polynomial at `dx` |
| `coeffsNormInf()`, `coeffsNorm<P>()` | buffer norms |
| `data()`, `coeffs()`, `rawCoeff(i)`, `setRawCoeff(i, v)` | raw access |
| `slice(d)`                           | Eigen `VectorBlock` for degree d |
| `advanceTo(d)`                       | no-op (storage is fully populated) |
| `auto eval() const`                | streaming assignment from any `StreamingExpression` |

The array and template-parameter `coeff` / `derivative` overloads
require compile-time multi-indices and are therefore disabled on the
dynamic path. `eval` accepts `std::array<T, Vars>` only on the static
path; the templated `Vec` form covers both.

## Concepts

```cpp
template <class T>
concept Scalar = std::floating_point<T>;

template <class E>
concept TaxExpression = /* see expr/base.hpp */;

template <class L, class R>
concept SameKindExpression =
    TaxExpression<L> && TaxExpression<R>
    && L::IsStatic == R::IsStatic
    && std::is_same_v<typename L::Scalar, typename R::Scalar>;

template <class E>
concept TaylorExpansion = /* runtime-side TTE concept; see concepts.hpp */;

template <class E>
concept StreamingExpression = /* requires advanceTo(d) and slice(d) */;
```

## Operators

All operators are constrained on `TaxExpression` /
`SameKindExpression`. They return ET nodes; results are materialised
on `.eval()`.

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

// trig + hyperbolic
auto sin(const E&);
auto cos(const E&);
auto tan(const E&);     // sin / cos (composed)
auto sinh(const E&);
auto cosh(const E&);
auto tanh(const E&);    // sinh / cosh (composed)

// paired trig / hyperbolic — share buffers across the two branches
auto sincos(const E&);     // returns SinCosPair<E> with .sin() / .cos()
auto sinhcosh(const E&);   // returns SinhCoshPair<E> with .sinh() / .cosh()

// inverse trig + hyperbolic
auto asin(const E&);
auto acos(const E&);
auto atan(const E&);
auto asinh(const E&);
auto acosh(const E&);
auto atanh(const E&);
auto atan2(const Y&, const X&);   // requires SameKindExpression<Y, X>

// exp / log
auto exp(const E&);
auto log(const E&);
auto log10(const E&);   // log(e) / log(10) (scaled view)

// roots, powers
auto sqrt(const E&);
auto cbrt(const E&);
auto square(const E&);
auto cube(const E&);                      // square(e) * e (composed)
template <int N> auto pow(const E&);      // compile-time integer exponent
auto pow(const E&, Scalar p);             // runtime real exponent

// hypot
auto hypot(const X&, const Y&);
auto hypot(const X&, const Y&, const Z&);

// erf
auto erf(const E&);

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
| `CbrtExpr<E>`          | buffered | `coeffs_`, F² aux |
| `ExpExpr<E>`           | buffered | `coeffs_` |
| `LogExpr<E>`           | buffered | `coeffs_` |
| `SinCosExpr<E, ReturnSin>`         | buffered | `sin_`, `cos_` (paired) |
| `SinhCoshExpr<E, ReturnSinh>`       | buffered | `sinh_`, `cosh_` (paired) |
| `SinCosNodeExpr<E>` + `SinCosPairView<Node, ReturnSin>` | shared buffered + view | sin/cos sharing |
| `SinhCoshNodeExpr<E>` + `SinhCoshPairView<Node, ReturnSinh>` | shared buffered + view | sinh/cosh sharing |
| `InverseFunctionExpr<E, FunKind, GMode, Sign>` | buffered | `coeffs_`, G aux |
| `Atan2Expr<Y, X>`      | buffered | `coeffs_`, x²+y² aux |
| `ErfExpr<E>`           | buffered | `coeffs_`, exp(-u²) aux |
| `PowRealExpr<E>`       | buffered | `coeffs_` |

Aliases `SinExpr`, `CosExpr`, `SinhExpr`, `CoshExpr` instantiate the
non-shared paired classes with the right `ReturnSin` / `ReturnSinh`.
Aliases `AtanExpr`, `AtanhExpr`, `AsinExpr`, `AcosExpr`, `AsinhExpr`,
`AcoshExpr` instantiate `InverseFunctionExpr` with the right
`(FunKind, GMode, Sign)` triple.

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
#include <tax/kernels/inverse_trig.hpp>
#include <tax/kernels/trig.hpp>
#include <tax/storage/tte.hpp>
#include <tax/expr/base.hpp>
#include <tax/expr/view_nodes.hpp>
#include <tax/expr/buffered_nodes.hpp>
#include <tax/ops/arithmetic.hpp>
#include <tax/ops/assign.hpp>
#include <tax/ops/math.hpp>
```

Day-to-day, `<tax/tax.hpp>` is enough.
