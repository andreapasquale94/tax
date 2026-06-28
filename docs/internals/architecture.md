# Architecture

The library is layered so each concern stays focused on one thing: storage
manages bytes, kernels manage math, operators manage syntax, and the Eigen
layer manages linear-algebra interop. Downstream projects build solvers on top
of this surface without touching it.

```
                ┌─────────────────────────────────────────┐
eigen           │ NumTraits + variables/value/eval/       │  Eigen interop
                │ derivative/gradient/jacobian/invert     │
                └────────────────┬────────────────────────┘
                                 │
                ┌────────────────▼────────────────────────┐
ops             │ +, −, ·, /, sin, exp, log, sqrt, …      │  user-facing surface
                └────────────────┬────────────────────────┘
                                 │
                ┌────────────────▼────────────────────────┐
detail/kernels  │ degree-by-degree recurrences            │  computational core
                │ cauchy, algebra, trig, transcendental,  │
                │ sparse_cauchy, sparse_subs              │
                └────────────────┬────────────────────────┘
                                 │
                ┌────────────────▼────────────────────────┐
expansion       │ TaylorExpansion<T,Scheme,Storage>       │  data type
                │ IndexScheme: IsotropicScheme<N,M>       │
                │           / MixedScheme<Groups...>      │
                │ MultiIndex, enumeration, concepts       │
                │ storage::Dense, storage::Sparse         │
                └─────────────────────────────────────────┘
```

---

## Core data type

`tax::TaylorExpansion<T, Scheme, Storage>` is parameterized by an `IndexScheme`
(constrained via `requires IndexScheme<Scheme>`) and partial-specialised on the
storage tag. The scheme encodes the monomial index set: `IsotropicScheme<N,M>`
gives the classic total-degree-$\le N$ graded-lex layout (exposed as `TE<N,M>`),
while `MixedScheme<Groups...>` supports anisotropic per-axis order caps
(exposed as `BoxTE<Group<Dim,Order>...>`). The kernels are scheme-generic.

- `storage::Dense` keeps a `std::array<T, C(N+M, M)>` — stack-resident, no heap,
  `constexpr`-friendly accessors.
- `storage::Sparse` keeps two parallel sorted vectors of flat-index / value
  pairs. `nnz()` returns the current support size.

Both expose the same public API (`value`, `coeff`, `derivative`, `eval`,
`deriv`, `integ`). `MultiIndex<M>` and the graded-lexicographic flat indexing
in `tax/expansion/enumeration.hpp` are storage-agnostic.

---

## Kernels

Every mathematical recurrence lives in `tax/expansion/detail/` (the private
kernel layer, `tax::detail::kernels`). A kernel takes raw
coefficient buffers — `T*` for dense, sorted index/value pairs for sparse —
plus the compile-time shape $(N, M)$ and writes directly into the result.
The kernel layer is the one place where the recurrences of
[Recurrence Relations](recurrences.md) live in code.

Univariate vs multivariate is dispatched by `if constexpr (M == 1)` — the
univariate path runs scalar loops over flat indices, the multivariate path
routes through `forEachSubIndex<M>(alpha, lo, hi, callback)`. Sparse uses its
own `sparse_cauchy` and `sparse_subs` kernels that exploit the sorted-index
representation for two-pointer merges.

See [Kernels & Recurrences](kernels.md) for the file-by-file map.

### Dispatch toggles

The dispatch macros are configured **in-header** in
`tax/expansion/detail/stencil_config.hpp` (`TAX_USE_UNROLL`,
`TAX_USE_STENCIL`, `TAX_STENCIL_MAX_BYTES`) — never injected from the build
system, so every translation unit sees identical inline definitions (ODR).

| Macro | What it changes |
|---|---|
| `TAX_USE_UNROLL`  | Switches the Dense `M == 1` Cauchy kernel to a compile-time-unrolled variant — faster for small $N$. |
| `TAX_USE_STENCIL` | For Dense `M ≥ 2`, precomputes the sub-multi-index stencil at compile time and reuses it across every Cauchy call. |

Both default to `ON`. The non-stencil and non-unroll paths remain in the tree
for cross-validation in `tests/kernels/`.

---

## Operators

`tax/expansion/ops/` is a thin facade that wraps each kernel in a free function
returning a fresh `TaylorExpansion`:

```cpp
template <typename T, IndexScheme Scheme>
[[nodiscard]] constexpr
TaylorExpansion<T, Scheme> square(const TaylorExpansion<T, Scheme>& x) noexcept {
    TaylorExpansion<T, Scheme> r;
    detail::kernels::seriesSquare<T, Scheme>(r.coefficients(), x.coefficients());
    return r;
}
```

No lazy expression-template layer — the return is materialised at every
operator boundary. RVO and the named-return optimisation keep this as cheap as
the in-place form when the compiler can see the kernel.

Operator overloads cover three cases: TE × TE, TE × scalar, and scalar × TE,
for `+`, `-`, `*`, `/`. Comparison operators compare the constant terms only
and are useful when threading TE values through Eigen factorisations or
control-flow predicates that branch on a representative value.

---

## Eigen integration

`tax/la.hpp` (namespace `tax::la`, sources under `tax/la/`) does two things:

1. **NumTraits specialisation** so `Eigen::Matrix<TE, R, C>` is a first-class
   Eigen type. Add/mul cost is set to the monomial count so Eigen's
   cache-aware algorithms pick reasonable strategies.
2. **Vocabulary helpers** — `variables`, `value`, `eval`, `derivative`,
   `gradient`, `hessian`, `jacobian`, `invert`. These wrap the underlying
   `TaylorExpansion` accessors and route through Eigen shape templates so they
   work uniformly with static-size, dynamic-size, and mixed expressions.

The integration is symmetric: any Eigen routine that doesn't require sparse
matrix traits accepts `TE` as a scalar; any user-written generic lambda on
Eigen vectors can be re-instantiated on `TE`-valued state without source
changes. This is exactly what downstream Taylor-based solvers exploit — they
compose a user function on `TE`-valued state to obtain Taylor coefficients via
automatic differentiation.

---

## Building on the core

Higher-level numerics build on `TaylorExpansion` without modifying it —
relying only on the dense type, its Eigen scalar integration, and the operator
surface documented above.

---

## Why not expression templates?

The library deliberately *doesn't* use lazy expression templates today. The
fixed-shape `std::array` payload and the kernels (which write
coefficient-by-coefficient into a destination buffer) make the eager path
already free of intermediate `TaylorExpansion` allocations under
RVO. Expression templates would add compile-time complexity without a measured
runtime win on the workloads driving the design (ODE integration and polynomial
maps in the downstream solvers). The door is open if a profile justifies it.
