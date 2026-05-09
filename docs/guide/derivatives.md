# Derivatives & evaluation

After streaming an expression into a `TEn` or `DynTE`, you query it
through a small set of accessors. None of them allocate; all of them
are `[[nodiscard]]`.

## Accessor matrix

| Accessor | Returns | Index forms (static path) |
|----------|---------|----------------------------|
| `value()` | scalar at expansion centre | — |
| `coeff(α)` | raw Taylor coefficient at α | span / array / template |
| `derivative(α)` | `α! · coeff(α)` | span / array / template |
| `eval(dx)` | truncated polynomial evaluated at dx | array / templated `Vec` |
| `coeffsNormInf()` | `‖c‖_∞` over the whole buffer | — |
| `coeffsNorm<P>()` | `‖c‖_p` over the whole buffer | template parameter |

The dynamic path supports the span and `std::vector` forms; the
template-parameter form only exists on the static path (the
multi-index has to be known at compile time).

## Three index forms (static path)

```cpp
auto [u, v] = tax::TEn<3, 2>::variables(std::array{1.0, 2.0});
tax::TEn<3, 2> p;
p <<= u * u * v;

// 1. std::span<const std::size_t> — canonical, used by the kernel layer.
std::array<std::size_t, 2> a{2, 1};
double c1 = p.coeff(std::span<const std::size_t>(a));

// 2. const std::array<std::size_t, Vars>& — braced-init friendly.
double c2 = p.coeff({2, 1});

// 3. template <std::size_t... Alpha> — compile-time index.
double c3 = p.coeff<2, 1>();
```

All three return the same value. Prefer:

- the **template form** when the index is known at compile time;
- the **array form** in everyday code where `{2, 1}` is more
  readable;
- the **span form** in generic kernel-like code that already has a
  `std::span<const std::size_t>` to forward.

## `derivative` vs. `coeff`

The Taylor coefficient `c_α` and the partial derivative `∂^|α| f /
∂x^α` differ by a factor of `α! = ∏ αᵢ!`. tax keeps the raw `c_α` in
the buffer so kernels can reuse the natural form, and exposes the
factorial scaling through `derivative`:

```cpp
auto x = tax::TE<3>::variable(0.0);
tax::TE<3> r;
r <<= tax::exp(x);

r.coeff({2});        // = 1/2 (Taylor coefficient)
r.derivative({2});   // = 1   (= 2! · 1/2)
r.derivative<2>();   // = 1   (compile-time index)
```

## `eval(dx)`

`eval(dx)` evaluates the truncated polynomial at displacement `dx`
from the expansion centre.

```cpp
auto x = tax::TE<8>::variable(0.0);
tax::TE<8> r;
r <<= tax::exp(x);

r.eval({0.3});       // ≈ exp(0.3) within the truncation error of order 8
```

For multivariate, `dx` is an `M`-element range:

```cpp
auto [u, v] = tax::TEn<5, 2>::variables(std::array{0.0, 0.0});
tax::TEn<5, 2> r;
r <<= tax::sin(u * v);
r.eval({0.1, 0.2});   // truncation of sin(0.02)
```

The static path's `eval` overloads to `std::array<T, Vars>` for
braced-init; the underlying template form accepts any range-like
indexable type (e.g. `Eigen::Vector`).

## Coefficient norms

```cpp
r.coeffsNormInf();      // L^∞ over the flat coefficient buffer
r.coeffsNorm<1>();      // L^1
r.coeffsNorm<2>();      // L^2
```

These are buffer norms, not function norms — useful for residual
checks and adaptive truncation, but they don't directly bound
`|f − f̃|` on a domain. (The standard sup-norm bound on a box involves
multiplying by box half-widths raised to each multi-index; see the
relevant DA literature.)
