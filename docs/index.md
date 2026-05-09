---
hide:
  - navigation
  - toc
---

<div class="tax-hero" markdown>

# Truncated Taylor expansions, in one pass.

A header-only C++23 library for multivariate Differential Algebra.
Each value carries a polynomial in M variables truncated at total
degree N; arithmetic and math operations propagate the full Taylor
jet through any expression in a single evaluation pass — no chain
rule, no tape.

<div class="tax-hero-actions" markdown>

[Get started :material-arrow-right:](getting-started.md){ .md-button .md-button--primary }
[Read the architecture](architecture.md){ .md-button }
[GitHub :fontawesome-brands-github:](https://github.com/andreapasquale94/tax){ .md-button }

</div>
</div>

```cpp
#include <tax/tax.hpp>

auto [u, v] = tax::TEn<3, 2>::variables(std::array{1.0, 2.0});

auto f = u * tax::sin(v) + u * v;     // lazy expression, no allocation yet

tax::TEn<3, 2> result;
result <<= f;                          // streaming degree-by-degree sweep

result.value();              // function value at the centre
result.derivative<1, 0>();   // partial w.r.t. u (compile-time index)
result.eval({0.1, 0.05});    // Taylor polynomial at a displacement
```

<div class="tax-features" markdown>

<div class="tax-feature" markdown>
### Slice-streamed ETs
View-like nodes never allocate; buffered nodes (`*`, `/`,
transcendentals) own only the buffer Cauchy convolution structurally
requires. The `<<=` driver writes straight into the destination, so
the top of the tree allocates nothing either.
</div>

<div class="tax-feature" markdown>
### Static and dynamic, one math core
`TruncatedTaylorExpansionT<T, N, M>` is the C++ hot path with
compile-time sizes. `DynamicTaylorExpansion<T>` is the runtime-sized
path Python sees. They share every coefficient kernel — no JIT, no
variant dispatch.
</div>

<div class="tax-feature" markdown>
### Compile-time indices
`result.coeff<1, 1>()` resolves the multi-index, flat offset, and
factorial scaling at compile time. The runtime cost collapses to a
single Eigen access.
</div>

<div class="tax-feature" markdown>
### Eigen-backed
Coefficients live in `Eigen::Matrix` (static) or `Eigen::VectorX`
(dynamic). Slices are real `VectorBlock` views; kernels operate on
Eigen-compatible expressions throughout.
</div>

<div class="tax-feature" markdown>
### Honest about its limits
Polynomial algebra, not chain-rule-on-tangents AD. No forward / reverse
mode. Buffered nodes structurally must allocate. Mixed static/dynamic
expressions are rejected at compile time.
</div>

<div class="tax-feature" markdown>
### Python via nanobind
`pip install nanobind`, configure with `-DTAX_BUILD_PYTHON=ON`, and
the `tax` module exposes `DynTE` plus the math functions. The brief's
example runs unchanged.
</div>

</div>

## At a glance

| | C++ | Python |
|---|---|---|
| Static-extent storage     | `tax::TE<N>`, `tax::TEn<N, M>` | — |
| Dynamic-extent storage    | `tax::DynTE<T>`                 | `tax.DynTE`                     |
| Streaming ET nodes        | view-like + buffered            | not exposed (eager evaluation)  |
| Build flag                | always on                       | `-DTAX_BUILD_PYTHON=ON`         |

## License

[BSD 3-Clause](https://github.com/andreapasquale94/tax/blob/main/LICENSE).
