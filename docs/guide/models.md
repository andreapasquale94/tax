# Taylor Models

A **Taylor model** is a truncated Taylor expansion that carries a rigorous
error bound: a pair $(P, I)$ of a polynomial and a remainder interval, tied to
an expansion point $\mathbf{x}_0$ and a domain box $[\mathbf{a}, \mathbf{b}]$,
such that

$$
f(\mathbf{x}) \;\in\; P(\mathbf{x} - \mathbf{x}_0) + I
\qquad \text{for all } \mathbf{x} \in [\mathbf{a}, \mathbf{b}].
$$

Where a plain `TE<N, M>` gives you a *floating-point approximation* of the
Taylor polynomial, a `TM<N, M>` gives you a *mathematically guaranteed
enclosure* of the function itself — the foundation of validated numerics:
verified range bounds, verified quadrature, and (downstream) verified ODE
flows. The implementation follows chapter 4 of K. Makino's PhD thesis
(*Rigorous Analysis of Nonlinear Motion in Particle Accelerators*, MSU 1998),
where Taylor models were introduced as "remainder-enhanced DA".

Everything lives in `namespace tax::model`; the types are re-exported as
`tax::Interval`, `tax::TaylorModel`, and the alias `tax::TM<N, M>`.

---

## Why not plain intervals?

Interval arithmetic is rigorous but suffers from the **dependency problem**:
`x - x` evaluates to `[a-b, b-a]` instead of `0`, and widths compound
catastrophically through long computations. Taylor models keep the *functional
dependence* in the polynomial — which cancels exactly — and only the small
truncation error in the interval:

```cpp
using I = tax::Interval<double>;

I xi{1.9, 2.1};
auto blowup = 1.0 / xi + xi;          // width 0.250 — actual range has width 0.150

auto x = tax::TM<3>::variable(2.0, I{1.9, 2.1});
auto f = 1.0 / x + x;                 // remainder width 4.0e-6 (!)
auto b = f.bound();                   // enclosure of width 0.151
```

The remainder width shrinks like $d^{N+1}$ with the domain width $d$, so
higher orders buy accuracy fast (thesis Table 4.2 — reproduced in
`tests/model/test_taylor_model_math.cpp`).

## Creating Taylor models

```cpp
#include <tax/tax.hpp>          // or just <tax/model.hpp>
using I = tax::Interval<double>;

// Univariate identity on a domain: x = 2 + dx on [1.9, 2.1], remainder [0,0]
auto x = tax::TM<3>::variable(2.0, I{1.9, 2.1});

// Multivariate
using TM2 = tax::TM<5, 2>;
TM2::Point  x0 {1.0, -1.0};
TM2::Domain dom{I{0.5, 1.5}, I{-1.5, -0.5}};
auto v  = TM2::variables(x0, dom);          // std::array of the M coordinates
auto xi2 = TM2::variable<0>(x0, dom);       // or one at a time
auto c  = TM2::constant(3.0, x0, dom);      // exact constant

// From parts: (P, I, x0, [a,b]) — e.g. to inject measured uncertainty
auto noisy = tax::TM<3>(x.polynomial(), I{-1e-9, 1e-9}, {2.0}, {I{1.9, 2.1}});
```

The expansion point must lie inside the domain (`std::invalid_argument`
otherwise). Binary operations require both operands to share the *same*
expansion point and domain.

## Arithmetic and intrinsics

The full expression surface mirrors the ordinary expansion types:

```cpp
auto g = exp(sin(x) + log(2.0 + x)) / sqrt(4.0 + x);
```

Available: `+`, `-`, `*`, `/` (Taylor model, scalar, and `Interval` operands),
and the intrinsics `exp`, `log`, `sqrt`, `isqrt` (reciprocal square root),
`reciprocal`, `square`, `pow(f, int)`, `sin`, `cos`, `tan`, `asin`, `acos`,
`atan`, `sinh`, `cosh`, `tanh`. Every operation propagates the remainder
rigorously — polynomial truncation excess, cross terms, and a Lagrange bound
on the series tail all land in the result's interval (see
[Internals / Taylor Models](../internals/taylor-models.md) for the formulas).

Interval operands model *unknown-but-bounded* constants:

```cpp
auto measured = x * I{1.9, 2.1};   // s*x for some s in [1.9, 2.1]
auto shifted  = x + I{-0.1, 0.3};  // x + s, s in [-0.1, 0.3]
```

!!! warning "Domain conditions throw"
    Intrinsics verify their domain conditions on the *enclosure* of the
    argument and throw `std::domain_error` when they cannot be certified
    (`log`/`sqrt`/`isqrt` need a positive enclosure, `reciprocal` and `/`
    need an enclosure excluding 0, `asin`/`acos` need $(-1, 1)$). A validated
    computation must never silently return garbage.

## Reading results out

```cpp
f.polynomial();     // the underlying TE<N, M> polynomial part
f.remainder();      // the remainder interval I
f.value();          // constant coefficient P(0)
f.bound();          // rigorous range enclosure B(P) + I over the whole domain
f.polynomialBound();// B(P) alone
f.orderBound(k);    // range of the degree-k homogeneous part (thesis I^k)
f.eval({0.05});     // enclosure of f(x0 + dx) at a point — an Interval
```

`eval` takes the displacement from the expansion point (like `TE::eval`) and
returns an `Interval`, since a rigorous statement about one point still
carries the remainder. Points outside the domain throw — the containment
guarantee only holds inside.

`bound()` and `polynomialBound()` accept an optional `Bounder` strategy. The
default `Bounder::Quadratic` bounds each variable's quadratic-plus-linear
part exactly (capturing interior minima the order-sum misses); pass
`Bounder::Naive` for the cheaper order-sum. Both are rigorous:

```cpp
auto g = (x - 0.3) * (x - 0.3);      // true range [0, 1.69] on a wide domain
g.bound(tax::Bounder::Naive);        // [-0.51, 1.69] — loose below zero
g.bound();                           // [0, 1.69] — exact, interior vertex found
```

## Verified integration

Antiderivation (thesis eq. 4.12) integrates the polynomial part and absorbs
the freed top-order block into the remainder:

```cpp
auto x = tax::TM<8>::variable(0.0, I{-0.5, 0.5});
auto F = cos(x).integ<0>();                    // TM of  ∫₀^dx cos(t) dt

// Definite integral over [-1/2, 1/2] by evaluating the antiderivative:
auto enclosure = F.eval({0.5}) - F.eval({-0.5});
// contains 2*sin(0.5) = 0.9588…, width ~1e-5
```

Iterating `integ` over several variables gives verified multidimensional
integrals by inclusion–exclusion over the corners — the §5.5.2 double
integral of the thesis is worked exactly this way in
`tests/model/test_thesis_examples.cpp`.

## Interval arithmetic on its own

`tax::Interval<T>` is usable stand-alone and is **outward-rounded**: every
computed endpoint is widened so results remain guaranteed enclosures under
IEEE floating point:

```cpp
I a{1.0, 2.0}, b{3.0, 5.0};
auto s = a + b;                       // ⊇ [4, 7]
auto q = a / b;                       // ⊇ [1/5, 2/3]; throws if 0 ∈ b
auto e = exp(a);                      // monotone enclosure
auto t = sin(I{0.0, 3.2});            // detects the enclosed maximum: hi = 1
auto p = tax::model::sqr(I{-1.0, 2.0}); // [0, 4] — sharp, not [-2, 4]
```

See the [API reference](../reference/models.md) for the complete surface and
the exact rounding contract.

## What is (and isn't) guaranteed

All *interval* computations — range bounds, remainder propagation, Lagrange
remainders, domain checks — are outward-rounded and conservative. The
floating-point rounding of the *polynomial coefficients* themselves (≈1 ulp
per coefficient operation) is **not** swept into the remainder; bounds are
rigorous in exact coefficient arithmetic. This matches the presentation of
thesis chapter 4; the coefficient-error tallying of COSY's `RD` type
(chapter 5) is future work. See
[Internals / Taylor Models](../internals/taylor-models.md#rigor-contract)
for the precise statement.
