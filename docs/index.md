---
hide:
  - navigation
---

# tax

**Truncated Algebraic eXpansions** — a header-only C++23 library for computing
truncated multivariate Taylor polynomials as first-class numerical objects.

Write a natural mathematical expression and tax propagates the full Taylor
series through it, yielding the function value **and** every partial derivative
up to order $N$ in a single evaluation pass.

```cpp
#include <tax/tax.hpp>

auto x = tax::TE<9>::variable(0.0);    // 9-th order univariate variable at x₀ = 0
tax::TE<9> f = tax::sin(x);            // one recurrence pass: value + all derivatives

f.value();          // sin(0)     = 0
f.derivative<1>();  // cos(0)     = 1
f.derivative<3>();  // −cos(0)    = −1
f.eval(0.3);        // sin(0.3) within machine precision
```

!!! warning "Active development"
    APIs and behavior may change between minor versions until 1.0.

---

## Why tax?

<div class="grid cards" markdown>

- :material-function-variant:{ .lg .middle } **Fixed-shape, allocation-free**

    Coefficient storage is `std::array<T, C(N+M, M)>` on the stack. Both `N` and
    `M` are compile-time integers, so the optimizer sees through every loop.

- :material-lightning-bolt:{ .lg .middle } **Fused kernels**

    Coupled pairs — `sinCos`, `sinhCosh`, `sqrtInvSqrt`, `expSinCos`, the
    `invSqrtPow<3>` gravity kernel — run in a single recurrence pass. The
    pure-polynomial surface (arithmetic, `square`, `cube`, `reciprocal`,
    integer `pow`) is `constexpr` and works in constant evaluation.

- :material-tune-vertical:{ .lg .middle } **Dense or sparse storage**

    `TE<N, M>` (dense, `std::array`) for the hot path; `STE<N, M>` (sparse,
    sorted-index map) when only a handful of monomials are non-zero. The two
    share the kernel layer and agree numerically.

- :material-orbit:{ .lg .middle } **Eigen-native**

    A `NumTraits` specialisation lets `TaylorExpansion` live inside Eigen
    vectors and matrices; helpers extract values, gradients, Jacobians, and
    Hessians.

- :material-tag-text:{ .lg .middle } **Named expansions**

    `NamedTaylorExpansion<T, N, Axes...>` attaches compile-time *named axes* to
    an expansion; values over different axis sets compose in their union, and
    `slice`/`deriv`/`integ` are addressed by name.

</div>

---

## At a glance

| What you write | What you get |
|---|---|
| `tax::TE<N>::variable(x0)` | univariate TE at $x_0$, order $N$ |
| `tax::TE<N, M>::variable<I>(x0)` | $I$-th coordinate variable, others as parameters |
| `tax::variables<TE<N,M>>(x0)` | Eigen column vector of all $M$ coordinate variables |
| `tax::sin(x) * tax::exp(y)` | full Taylor series of the product, one kernel pass per op |
| `tax::expSinCos(v, u)` | `{exp(v)·sin(u), exp(v)·cos(u)}` fused in one coupled pass |
| `constexpr auto g = tax::square(x) + 2.0*x + 1.0;` | polynomial pipeline evaluated at compile time |
| `f.derivative<2, 1>()` | $\partial^3 f / \partial x^2 \partial y$ at $x_0$ |
| `f.eval(dx)` | Horner evaluation of the polynomial at $x_0 + \delta x$ |
| `tax::jacobian(F, M)` | Eigen Jacobian of a vector function |
| `tax::toHermite(f)` / `tax::toChebyshev(f)` | monomial-basis coefficients converted to the Hermite / Chebyshev basis |
| `tax::la::mean(F)` / `covariance(F)` / `skewnessTensor(F)` / `kurtosisTensor(F)` | statistical moments of a polynomial map, `x ~ N(0, I)` |
| `tax::MixedTE<Group<Dim,Order>...>` | anisotropic per-axis order caps (see [Named & Mixed-Order expansions](guide/named.md#anisotropic-axes-per-axis-orders)) |
| `tax::NamedTaylorExpansion<T, N, Axes...>` | TE with named, type-level variables |

---

## Navigating the docs

<div class="grid cards" markdown>

- [:material-rocket-launch: __Getting Started__](getting_started.md)

    Install, build, and write your first Taylor expansion.

- [:material-school: __Guide__](guide/index.md)

    How-to walkthroughs: variables and expressions, fused operations,
    extracting results, storage, named and mixed-order expansions, Eigen
    integration.

- [:material-book-open-variant: __Reference__](reference/index.md)

    Exact signatures — the `TaylorExpansion` API, `tax::la`, `tax::named`,
    and the per-operation recurrence catalog.

- [:material-cog: __Internals__](internals/index.md)

    Architecture, kernels, recurrence relations.

</div>

---

## Requirements

- C++23 compiler — GCC 13+, Clang 17+, Apple Clang 16+
- CMake 3.28+
- Eigen 3.4+ (linked via `find_package(Eigen3)`)

## License

[BSD 3-Clause](https://github.com/andreapasquale94/tax/blob/main/LICENSE).
