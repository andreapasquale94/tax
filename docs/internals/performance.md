# Performance notes & optimization roadmap

`tax` evaluates **eagerly**: every operator materialises a full
`TaylorExpansion` and runs the relevant recurrence over its whole coefficient
array. This page records where the time goes, the optimizations already applied,
and the evidence-backed proposals that remain.

Reproduce any number here with the in-repo micro-benchmarks:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARKS=ON
cmake --build build --target bench_taylor -j
taskset -c 0 ./build/benchmarks/bench_taylor      # pin a core for stable ns/op
```

Measurements below: GCC 13.3, `-O3 -march=native`, single core.

## Where the time goes

The dominant cost in every non-trivial expression is the **multivariate Cauchy
product** (`operator*` for `M >= 2`) and the functions built on it
(`square`, `pow`, the inverse trig/hyperbolic kernels, `composite`). It grows
as `numMonomials(N, 2M)` and dwarfs the elementwise ops:

| op (N6 M4)     | ns/op |
|----------------|------:|
| add-chain (×8) |   600 |
| div            |  2200 |
| exp            |  2350 |
| **mul**        | **5800** |
| **mul-chain (×4)** | **22600** |

The univariate path (`M == 1`) uses a fully-unrolled product kernel and is
already near-optimal (a loop kernel is ~60× slower at N=30); the `O(N^2)`
univariate recurrences (`exp`, `sin`, `div`, `sqrt`) leave little headroom.

## Implemented

### 1. CSR (row-compressed) Cauchy stencil

The `M >= 2` product was driven by a flat table of `(out_idx, a_idx, b_idx)`
entries, accumulated with a scattered read-modify-write `out[out_idx] += …`.
Because the table is generated in graded-lex (flat-index) order, every output's
contributions are already contiguous, so the table is now stored **CSR-style**:
a `pairs[(a_idx, b_idx)]` array plus per-output `offsets`. The kernel
accumulates each output in a register and writes it once.

Benefits: no scattered RMW (no load/store dependency chain on `out`), no
separate zero-fill, and a third less table memory (8 vs 12 bytes/entry → better
cache residency). Results are **bit-identical** (per-output summation order is
preserved). Applied to both `IsotropicScheme` and `MixedScheme`.

Speedups (before → after, same machine):

| product        | M6 N8 | M9 N6 | M9 N8 | mixed (4 vars) |
|----------------|------:|------:|------:|---------------:|
| `operator*`    | 3.3×  | 3.0×  | 3.1×  | 2.8×           |

Multiplication-heavy composites improve 1.4–2.5× in step; the win **grows with
problem size**, which is exactly the many-variable regime.

### 2. Graceful stencil → loop fallback (`TAX_STENCIL_MAX_BYTES`)

Precomputed stencil tables scale as `numMonomials(N, 2M)`. For many variables at
high order they become enormous — e.g. `M=9, N=10` needs ~100 MB, `M=9, N=15`
~8 GB — and previously tripped a **hard `static_assert`: those expansions
would not compile at all.**

The dispatch now gates on a configurable budget (`TAX_STENCIL_MAX_BYTES`,
default 64 MB). When a table would exceed it, the product and recurrence kernels
fall back to the constexpr loop enumeration instead of erroring. `TE<10,9>` and
`TE<15,6>` now compile and run (slower, table-free) rather than failing. The
recurrence loop fallback's scratch buffer is bounded by the true maximum row
length (`maxRecurrenceRow`), independent of `nCoeff`, so it stays small.

> The loop and stencil paths sum each output in different orders, so they agree
> only to round-off (the long-standing loop-vs-stencil contract).

## Proposed (not yet implemented)

### A. Expression templates for elementwise fusion

Eager elementwise chains (`2*a + 3*b - c + 0.5*d`) materialise one full
temporary array per operator. At large `nCoeff` those passes spill to memory; a
single fused loop (what an expression-template layer would emit) recovers it:

| linear combination | M1/M2 | N6 M3 | N6 M4 |
|--------------------|------:|------:|------:|
| eager vs fused     | ~1.0× | 1.8×  | **2.5×** |

No benefit at small `nCoeff` (temporaries stay in registers), so an ET layer
should target `+`, `-`, scalar `*`, and `axpy`-shaped nodes for the
large-`nCoeff` (many-variable) case. The `*` (Cauchy) node stays eager.
Caveat: invasive — it interacts with the Eigen `NumTraits` surface and the
named/mixed wrappers, so it needs a design pass.

### B. Common-subexpression reuse

Eager evaluation cannot reuse a repeated subexpression: `sin(x)*cos(x) + sin(x)`
computes `sin(x)` twice. Binding it manually recovers ~1.3–1.45× consistently:

```cpp
auto s = sin(x);                 // compute once
auto f = s * cos(x) + s;
```

Short term: document this (a single transcendental at high `M` is one of the
most expensive calls in the library). Long term: a lazy expression DAG with
hashing could automate CSE, but that is a large architectural change.

### C. Loop-fallback performance for very large `M`

When the stencil budget is exceeded the loop kernel recomputes `flatIndex`/
sub-index decompositions per call. For the upper end of the many-variable range
this dominates. Options: a smaller "degree-banded" stencil that stores only the
decomposition skeleton, or recommending **sparse storage** (`STE`) when the
underlying functions have sparse high-order expansions.
