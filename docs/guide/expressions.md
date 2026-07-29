# Variables & Expressions

This guide shows how to introduce independent variables and build expressions
over them. Every snippet assumes `#include <tax/tax.hpp>`.

---

## Creating variables

A variable is a Taylor expansion whose constant term is the expansion point and
whose first-order term is the identity perturbation. Everything you build from
there propagates the full Taylor series in one pass.

=== "Univariate (dense)"

    ```cpp
    // Order-5 expansion of x around x₀ = 1.0
    auto x = tax::TE<5>::variable(1.0);   // x = 1 + δx
    // Coefficients (Dense): [1.0, 1.0, 0, 0, 0, 0]
    //                         ↑    ↑
    //                       value  δx coefficient

    // A pure constant (no δx dependence)
    auto c = tax::TE<5>::constant(3.14);  // [3.14, 0, 0, 0, 0, 0]
    ```

=== "Multivariate (per-coordinate)"

    ```cpp
    using TE2 = tax::TE<3, 2>;
    const std::array<double, 2> p{1.0, 2.0};

    auto x = TE2::variable<0>(p);   // δx coordinate around (1, 2)
    auto y = TE2::variable<1>(p);   // δy coordinate around (1, 2)
    ```

=== "Eigen vector form"

    ```cpp
    // The whole coordinate vector at once, as Eigen::Vector<TE2, 2>
    auto v = tax::variables<tax::TE<3, 2>>(Eigen::Vector2d{1.0, 2.0});
    auto& x = v(0);
    auto& y = v(1);
    ```

The per-coordinate index can also be chosen at runtime when it is not known at
compile time:

```cpp
auto z = tax::TE<3, 2>::variable(1.0, /*var_idx=*/0);
```

!!! note "Picking a factory"
    Use the compile-time `variable<I>(p)` form whenever the coordinate index is
    a constant — it is fully checked at compile time. The runtime
    `variable(value, idx)` form exists for the cases where the index is computed.
    The Eigen `tax::variables<TE>(x0)` helper is the most convenient way to seed
    a full coordinate vector at once; see
    [Eigen Integration](eigen.md) for what you can do with the result.

---

## Building expressions

Arithmetic and math functions work naturally; every operator materialises its
result eagerly by running a single kernel pass. You just write the math.

```cpp
auto x = tax::TE<6>::variable(0.0);
tax::TE<6> f = tax::sin(x) + tax::square(x) / 2.0;
```

!!! note "Eager evaluation"
    Each operator and math function returns a fresh, fully evaluated
    `TaylorExpansion` — there is no lazy expression-template layer. The
    fixed-shape `std::array` payload and RVO keep the eager path free of heap
    allocations and intermediate copies; see
    [Internals / Architecture](../internals/architecture.md) for why the
    library deliberately avoids expression templates.

### Arithmetic and composition

```cpp
auto x = tax::TE<5>::variable(1.0);

tax::TE<5> f = (x + 2.0) * (x - 3.0);   // x² - x - 6 at x₀ = 1
tax::TE<5> g = x + x*x + x*x*x;         // chained sums
tax::TE<5> h = 1.0 / (1.0 + x);         // reciprocal recurrence
```

Multivariate expressions compose the same way:

```cpp
using TE2 = tax::TE<3, 2>;
const std::array<double, 2> p{1.0, 2.0};
auto x = TE2::variable<0>(p);
auto y = TE2::variable<1>(p);

TE2 f = x*x + 2.0*x*y + y*y;   // (x + y)² at (1, 2)
```

### Math functions

All standard mathematical functions — `sin`, `cos`, `tan`, `exp`, `log`,
`sqrt`, `cbrt`, `pow`, the hyperbolic and inverse families, `erf`, `atan2`, … —
are propagated via degree-by-degree recurrences in a single forward pass.

```cpp
auto x = tax::TE<8>::variable(0.0);

tax::TE<8> s  = tax::sin(x);     // [0, 1, 0, -1/6, 0, 1/120, ...]
tax::TE<8> c  = tax::cos(x);     // [1, 0, -1/2, 0, 1/24, 0, ...]

auto y = tax::TE<8>::variable(1.0);
tax::TE<8> e  = tax::exp(y);     // exp(1+δx) = e · [1, 1, 1/2, 1/6, ...]
tax::TE<8> l  = tax::log(y);     // log(1+δx) = [0, 1, -1/2, 1/3, ...]

tax::TE<8> sq = tax::sqrt(tax::TE<8>::variable(4.0));  // √(4+δx)
tax::TE<8> cb = tax::cbrt(tax::TE<8>::variable(8.0));  // ∛(8+δx)

tax::TE<10> h = tax::atan(x) / (1.0 + x*x);
```

A worked composition mixing arithmetic and transcendentals:

```cpp
using TE2 = tax::TE<5, 2>;
const std::array<double, 2> p{0.0, 0.0};
auto x = TE2::variable<0>(p);
auto y = TE2::variable<1>(p);

TE2 f = tax::sin(x) * tax::cos(y);   // full bivariate Taylor series in one pass
```

!!! tip "Fused pair operations"
    When an expression needs *both* halves of a natural pair — `sin` **and**
    `cos` of the same argument, `sqrt` **and** `1/sqrt`, or the damped
    oscillation `exp(v)·sin(u)` / `exp(v)·cos(u)` — the fused surface
    (`sinCos`, `sinhCosh`, `sqrtInvSqrt`, `expSin`/`expCos`/`expSinCos`,
    `halfPow<K>`, `invSqrtPow<K>`) computes them in a single coupled
    recurrence pass. See [Fused Operations](fused.md).

---

## Compile-time evaluation

The **polynomial surface** is `constexpr`, so a fixed local Taylor model built
from it can be baked into the binary at compile time. This covers the
arithmetic operators (`+`, `-`, `*`, `/`), `square`, `cube`, `reciprocal`,
integer `pow(x, n)`, and the differential/evaluation accessors
(`deriv`, `integ`, `eval`, `truncate`, `coeff`, `derivative`):

```cpp
constexpr auto x = tax::TE<8>::variable(0.5);
constexpr auto g = tax::square(x) + 2.0 * x + 1.0;   // (x + 1)²

static_assert(g.value() == 2.25);                // evaluated by the compiler
static_assert(g.coeff<1>() != 0.0);
constexpr double d2 = g.derivative<2>();         // usable as a constant expression

constexpr auto p = tax::pow(x, 3);               // integer power — constexpr
```

Multivariate, named, and mixed-order polynomial pipelines work the same way
(wrap the variable setup in an immediately-invoked `constexpr` lambda when it
needs more than one statement).

!!! note "Transcendentals are runtime-only"
    The transcendental, root, and real-exponent functions — `exp`, `log`,
    `sin`, `cos`, `sqrt`, `cbrt`, `pow(x, p)` for real `p`, `atan2`, `erf`, the
    hyperbolic and inverse families, and the fused pair operations — seed their
    recurrence with a plain libm call (`std::exp`, `std::sin`, …) at the
    constant term. They are therefore **not** `constexpr` and cannot run in
    constant evaluation; call them at runtime.

---

**Next:** once you have built an expression, see
[Extracting Results](results.md) for pulling out values, coefficients, and
derivatives. The complete operator and function surface is listed in the
[Core API Reference](../reference/core.md).
