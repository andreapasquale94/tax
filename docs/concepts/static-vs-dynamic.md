# Static vs. dynamic

tax exposes a single storage template. Modelled on
`Eigen::Matrix<T, Rows, Cols>`, the size template parameters use
`Eigen::Dynamic` (= -1) as the runtime-size sentinel:

```cpp
template <class T, int Order, int Vars>
class TaylorExpansionT;
```

Two configurations are publicly supported:

| Configuration | Template parameters | Storage |
|--------------|----------------------|---------|
| **Static**  | `Order, Vars >= 0` (compile-time integers) | `Eigen::Matrix<T, monomialCount(N, M), 1>`, stack-allocated |
| **Dynamic** | `Order = Vars = Eigen::Dynamic`            | `Eigen::VectorX<T>`, heap-allocated |

Mixed dynamism — static `Order` with dynamic `Vars` or vice versa —
is rejected with a `static_assert`.

## The aliases that hide the sentinel

User code rarely spells out `TaylorExpansionT` directly. Three aliases
cover every case:

```cpp
template <int N>           using TE  = TaylorExpansionT<double, N, 1>;
template <int N, int M>    using TEn = TaylorExpansionT<double, N, M>;
template <class T = double> using DynTE = TaylorExpansionT<T, Eigen::Dynamic, Eigen::Dynamic>;
```

So:

```cpp
auto x        = tax::TE<5>::variable(1.0);                       // static, M=1
auto [u, v]   = tax::TEn<3, 2>::variables(std::array{1.0, 2.0});  // static, M=2
auto x_const  = tax::TE<5>::constant(3.0);
auto z        = tax::TE<5>::zero();

auto dx = tax::DynTE<>::variable(1.0, /*order=*/3, /*nvars=*/1, /*var_idx=*/0);
auto dvars = tax::DynTE<>::variables({1.0, 2.0}, /*order=*/3);    // std::vector<DynTE>
```

The static aliases pin both sizes; the dynamic alias wires both
template parameters to the sentinel and leaves the runtime values to
the constructor.

## One template, one math core

Every `TaxExpression` exposes:

```cpp
using Scalar = …
static constexpr int  OrderAtCompileTime;   // template arg, possibly Eigen::Dynamic
static constexpr int  VarsAtCompileTime;    //          "
static constexpr bool IsStatic;             // (Order != Dynamic) && (Vars != Dynamic)
static constexpr bool IsDynamic;            // !IsStatic
std::size_t order() const;                  // runtime value (matches OrderAtCompileTime when static)
std::size_t nvars() const;                  // runtime value
auto slice(std::size_t d) const;            // Eigen-compatible slice view
void advanceTo(std::size_t d) const;
```

The constants mirror Eigen's `RowsAtCompileTime` /
`ColsAtCompileTime` convention: they always carry the template
argument, with `Eigen::Dynamic` standing in for unknown sizes. The
runtime methods `order()` / `nvars()` return real values — for the
static path they're `constexpr` and equal the template arguments;
for the dynamic path they return runtime members.

Internal code branches with `if constexpr (IsStatic)` / `if constexpr
(IsDynamic)` to select compile-time vs. runtime paths. The
mathematical kernels (Cauchy convolution, transcendental recurrences)
read sizes through `order()` / `nvars()` and therefore work for either
case unchanged.

## How the static path stays allocation-free

Internally `TaylorExpansionT` privately inherits from a tiny
`detail::ShapeBase<Order, Vars>` helper:

- `ConstexprShape<Order, Vars>` — empty struct exposing
  `static constexpr` `order()` / `nvars()` returning the template values.
- `DynamicShape` — holds runtime `order_` / `nvars_` members.

`std::conditional_t` picks the right one by template argument; empty
base optimisation drops `ConstexprShape` to zero size in the static
case. A static `TE<3>` is therefore byte-for-byte the same as
`Eigen::Matrix<double, 4, 1>` — a static_assert in
`tests/test_storage.cpp` enforces this.

## Mixed expressions are rejected

```cpp
auto sx = tax::TE<3>::variable(1.0);
auto dx = tax::DynTE<>::variable(1.0, 3, 1, 0);

auto bad = sx + dx;     // ❌ does not compile:
                        //    SameKindExpression<TE, DynTE> fails
```

The `SameKindExpression` concept checks `L::IsStatic == R::IsStatic`
(and, when both are static, that orders and variable counts match).
Mixed dimensions fail at the operator's `requires` clause; the error
points at the assertion text rather than a deep template backtrace.

If you genuinely need to bridge the two paths, copy coefficients
between buffers in user code. There is no implicit adapter,
intentionally.

## When to use which

| Situation | Use |
|-----------|-----|
| C++ code with truncation order known at compile time | `TE<N>` / `TEn<N, M>` |
| Variable order or variable count chosen at runtime | `DynTE<T>` |
| Python | `DynTE` (`-DTAX_BUILD_PYTHON=ON`) |
| Storing many TTEs in an `Eigen::Matrix` of TTEs | static path |
| Iterating with sizes from a config file | dynamic path |

The static path is faster — fully inlinable, no heap, no size
checks — and is the recommended default for C++. The dynamic path
exists for the long tail of cases where compile-time sizes don't
work.
