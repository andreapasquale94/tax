# Static vs. dynamic

tax exposes two storage types. Both satisfy a single
`tax::TaxExpression` concept and feed the same coefficient kernels;
only the size resolution differs.

## Two storage types

### `TruncatedTaylorExpansionT<T, N, M>`

Compile-time-fixed `T`, order `N`, and number of variables `M`.
Coefficients live in `Eigen::Matrix<T, monomialCount(N, M), 1>` —
stack-allocated, no heap.

```cpp
auto x        = tax::TE<5>::variable(1.0);
auto [u, v]   = tax::TEn<3, 2>::variables(std::array{1.0, 2.0});
auto x_const  = tax::TE<5>::constant(3.0);
auto z        = tax::TE<5>::zero();
```

This is the C++ hot path. Inlining and constant propagation are
maximal; the compiler sees every size at compile time.

Aliases:

```cpp
template <std::size_t N>            using TE  = TruncatedTaylorExpansionT<double, N, 1>;
template <std::size_t N, std::size_t M> using TEn = TruncatedTaylorExpansionT<double, N, M>;
```

### `DynamicTaylorExpansion<T>`

Runtime-fixed `order_` and `nvars_` carried as members; coefficients
live in `Eigen::VectorX<T>`.

```cpp
auto x = tax::DynTE<>::variable(1.0, /*order=*/3, /*nvars=*/1, /*var_idx=*/0);
auto vars = tax::DynTE<>::variables({1.0, 2.0}, /*order=*/3);  // returns std::vector<DynTE>
```

This is the type Python sees. There is no `std::variant` over an
`(N, M)` grid and no JIT instantiation: Python users get exactly
`DynTE<double>`, full stop.

Alias:

```cpp
template <class T = double> using DynTE = DynamicTaylorExpansion<T>;
```

## Both satisfy `TaxExpression`

```cpp
template <class E>
concept TaxExpression =
    std::is_base_of_v<expr::ExprTag, std::remove_cvref_t<E>>
    || requires { typename std::remove_cvref_t<E>::Scalar; }
       && requires {
           { std::remove_cvref_t<E>::kStatic } -> std::convertible_to<bool>;
       };
```

Every `TaxExpression` exposes:

- `using Scalar = …`
- `static constexpr bool kStatic`
- `std::size_t order() const`
- `std::size_t nvars() const`
- `void advanceTo(std::size_t d)` (const; mutable internally for buffered nodes)
- `auto slice(std::size_t d) const`

The kernels never branch on `kStatic`; they take any operand that
honours the contract. The dynamic path's `monomialCount(order, nvars)`
runs at runtime where the static path's runs at compile time, but the
code is identical.

## Mixed expressions are rejected

```cpp
auto sx = tax::TE<3>::variable(1.0);
auto dx = tax::DynTE<>::variable(1.0, 3, 1, 0);

auto bad = sx + dx;     // ❌ does not compile:
                        //    SameKindExpression<TE, DynTE> fails
```

The `SameKindExpression` concept checks `L::kStatic == R::kStatic`
(and, when both are static, that orders and variable counts match).
Mixed dimensions fail at the operator's `requires` clause; the error
points at the static_assert text rather than a deep template
backtrace.

If you genuinely need to bridge the two paths, convert explicitly —
copy coefficients between buffers in user code. There is no implicit
adapter, intentionally.

## When to use which

| Situation | Use |
|-----------|-----|
| C++ code with a fixed truncation order known at compile time | `TE<N>` / `TEn<N, M>` |
| Variable order or variable count chosen at runtime | `DynTE<T>` |
| Python | `DynTE` (`-DTAX_BUILD_PYTHON=ON`) |
| Storing many TTEs in an `Eigen::Matrix` of TTEs | static path |
| Iterating with sizes from a config file | dynamic path |

The static path is faster — fully inlinable, no heap, no size
checks — and is the recommended default for C++. The dynamic path
exists for the long tail of cases where compile-time sizes don't
work.
