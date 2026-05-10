# Multivariate

For more than one variable, use `TEn<N, M>` (static) or
`DynTE<T>` (dynamic). The two paths share the same coefficient
layout and the same multi-index conventions.

## Creating M independent variables

### Static path — structured bindings

```cpp
auto [u, v] = tax::TEn<3, 2>::variables(std::array{1.0, 2.0});
// u.value() == 1.0, v.value() == 2.0
// u has dx-seed at multi-index (1, 0); v at (0, 1).
```

`variables` returns a `std::tuple<TEn, TEn>` of size `M` with each
element seeded against the matching variable.

There's also a parameter-pack overload for plain scalars:

```cpp
auto [u, v, w] = tax::TEn<3, 3>::variables(1.0, 2.0, 3.0);
```

### Dynamic path — vector of TTEs

```cpp
auto vars = tax::DynTE<>::variables({1.0, 2.0}, /*order=*/3);
auto& u = vars[0];
auto& v = vars[1];
```

The dynamic factory returns a `std::vector<DynTE<T>>` with one entry
per element of `x0`.

## Multi-index layout

Coefficients are laid out **graded reverse-lex**. For `M = 2`, `N = 2`:

| Flat index | Multi-index | Degree |
|-----------:|:-----------:|:------:|
| 0 | (0, 0) | 0 |
| 1 | (1, 0) | 1 |
| 2 | (0, 1) | 1 |
| 3 | (2, 0) | 2 |
| 4 | (1, 1) | 2 |
| 5 | (0, 2) | 2 |

Lower total degree first; within a degree, smaller alpha at the
**rightmost** axis comes first. This makes the degree-d slice a
contiguous block of length `degreeSize(d, M) = C(d+M-1, M-1)`.

The exact layout is rarely user-visible — `coeff` /
`derivative` accept the multi-index directly. But it's worth knowing
when you reach for `rawCoeff(i)`.

## Reading a coefficient

```cpp
auto [u, v] = tax::TEn<3, 2>::variables(std::array{1.0, 2.0});
tax::TEn<3, 2> p;
p = (u * v).eval();

// Three equivalent index forms — all return the (1, 1) coefficient.
double a = p.coeff({1, 1});                     // std::array overload
double b = p.coeff<1, 1>();                     // template-param form
double c = p.coeff(std::span<const std::size_t>(
              std::array<std::size_t, 2>{1, 1}));  // canonical span
```

The template-param form `coeff<1, 1>()` resolves the flat index at
**compile time**; the runtime cost is a single Eigen access. It's the
form to reach for when the multi-index is known statically.

## Partial derivatives

`derivative({α})` returns `α! · coeff({α})` — the actual
`∂^|α| f / ∂x^α` evaluated at the expansion centre.

```cpp
tax::TEn<3, 2> p;
p = (u * u * v).eval();

p.derivative({2, 0});   // 2! · 1! · coeff({2,0}) — value at the centre
p.derivative<2, 1>();   // = 2! · 1! · coeff<2,1>() = 2 · 1 = 2
```

For `f = u² · v` at `(u₀, v₀) = (1, 2)`, the second-derivative-in-u,
first-in-v is `∂³f / (∂u² ∂v) = 2`.

## Polynomial evaluation

`eval(dx)` evaluates the truncated polynomial at displacement `dx`
from the expansion centre.

```cpp
tax::TEn<8, 2> r;
r = (tax::exp(u + v)).eval();
r.eval({0.1, 0.05});   // ≈ exp(1.0 + 2.0 + 0.1 + 0.05)
```

The dynamic path's `eval` takes any range-like (e.g. `std::vector`,
`Eigen::Vector`) of length `nvars()`.

## Two static-path conventions worth pinning

1. **`coeff` doesn't apply factorials**, `derivative` does.
   `coeff({α})` returns the raw Taylor coefficient; `derivative({α})`
   multiplies by `α!`.

2. **The variable seeding convention.** `TE<N>::variable(x₀)` produces
   `x = x₀ + dx` (a degree-1 seed). For multivariate variables, the
   `i`-th call seeds against the `i`-th `dx_i`.

If you want a TTE whose constant term is `x₀` and whose linear part is
zero (a "constant" centred at `x₀`), use `constant(x₀)` instead.
