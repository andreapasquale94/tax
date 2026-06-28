# Kernels & Recurrences

The kernel layer is where the math lives in code. Each function in
`tax/expansion/detail/` implements one of the degree-by-degree recurrence
relations from [Recurrence Relations](recurrences.md), operating on raw
coefficient buffers rather than the user-facing `TaylorExpansion` type. The
operator layer in `tax/expansion/ops/` is a thin wrapper that calls a kernel
and returns a fresh `TaylorExpansion`.

---

## File map

| Header | Kernels | Operations |
|---|---|---|
| `tax/expansion/detail/cauchy.hpp` | `seriesCauchy`, `seriesCauchyAccumulate`, `seriesSquare`, `seriesCube` | multiplication, integer powers |
| `tax/expansion/detail/cauchy_unroll.hpp` | unrolled `M==1` Dense Cauchy | tight univariate hot path (`TAX_USE_UNROLL`) |
| `tax/expansion/detail/cauchy_stencil.hpp` | precomputed stencil-driven `M≥2` Dense Cauchy | multivariate hot path (`TAX_USE_STENCIL`) |
| `tax/expansion/detail/recurrence_stencil.hpp` | `RecurrenceStencil`, `forEachRecurrenceRow` | shared decomposition table driving every `M≥2` recurrence |
| `tax/expansion/detail/stencil_config.hpp` | `TAX_USE_UNROLL`, `TAX_USE_STENCIL`, `TAX_STENCIL_MAX_BYTES` | in-header kernel-dispatch configuration |
| `tax/expansion/detail/algebra.hpp` | `seriesReciprocal`, `seriesSqrt`, `seriesCbrt`, `seriesSquare`, `seriesCube` | algebraic recurrences |
| `tax/expansion/detail/trigonometric.hpp` | coupled `seriesSinCos`, `seriesTan`, inverse trig | trig and inverse trig |
| `tax/expansion/detail/transcendental.hpp` | `seriesExp`, `seriesLog`, `seriesSinhCosh`, `seriesTanh`, `seriesErf`, `seriesPow` | exp/log/hyperbolic/erf/power |
| `tax/expansion/detail/sparse_cauchy.hpp` | `seriesCauchy` for the sparse storage | multiplication on sparse polynomials |
| `tax/expansion/detail/sparse_subs.hpp` | shared sparse subroutines (add/sub merges, etc.) | sparse arithmetic helpers |

---

## Univariate vs multivariate dispatch

Every recurrence picks one of two paths via `if constexpr (M == 1)`:

- **Univariate** (`M == 1`) — scalar loops over flat indices. Simple,
  branchless, easy to vectorise. The unrolled Dense Cauchy variant
  (`TAX_USE_UNROLL`) is what runs here.
- **Multivariate** (`M ≥ 2`) — routes through
  `forEachRecurrenceRow<N, M>(fn)`, which calls
  `fn(ai, d, row)` for every output flat index `ai` of degree `d ≥ 1` in
  graded-lex order, where `row` is the span of all decompositions
  $(\beta, \gamma)$ with $\beta + \gamma = \alpha$ and $|\beta| \ge 1$ —
  each entry carrying precomputed `flatIndex(beta)`, `flatIndex(gamma)`
  and $|\beta|$. That row shape is exactly what every recurrence in
  [Recurrence Relations](recurrences.md) needs; the weight
  ($1$, $|\beta|$, $|\gamma|$, $c|\beta| - |\gamma|$, …) stays in the kernel.

With `TAX_USE_STENCIL=1` (the in-header default from
`tax/expansion/detail/stencil_config.hpp`) the rows come from a
`RecurrenceStencil<N, M>` table built once at first use — no multi-index
arithmetic remains in the inner loops. The general Cauchy product uses its
own flat `CauchyStencil` table. With the macro off — and always in constant
evaluation — the rows are enumerated on the fly in the same order, so the
two paths are bit-identical; the agreement is pinned by
`tests/kernels/test_cauchy_stencil_diff.cpp`.

---

## Recurrence shape — anatomy of `seriesExp`

For $g = \exp(f)$ the closed-form recurrence is

$$
g_\alpha \;=\; \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} |\beta| \, f_\beta \, g_{\alpha - \beta},
\qquad g_0 = \exp(f_0)
$$

which the kernel implements (in pseudocode) as

```cpp
g[0] = std::exp(f[0]);
forEachRecurrenceRow<N, M>([&](std::size_t ai, int d,
                               std::span<const RecurrenceEntry> row) {
    T sum{};
    for (const RecurrenceEntry& e : row)
        sum += T(e.db) * f[e.b_idx] * g[e.g_idx];
    g[ai] = sum / T(d);
});
```

Every other recurrence has the same skeleton: initialise the
constant-term value with the scalar math, then walk degrees $d = 1, \ldots, N$
applying the appropriate sub-multi-index sum.

For *coupled* recurrences (sin/cos, sinh/cosh) the kernel maintains both
arrays in lockstep — see `seriesSinCos`. For *helper-driven* recurrences
(asin via $h = \sqrt{1 - f^2}$, tan via $\cos \cdot t = \sin$) the helper is
materialised first by an inner kernel call, then the outer recurrence runs.

---

## Symmetric exploitation

A few recurrences appear with explicit factor-of-two savings:

- `cauchySelfProduct` (univariate, and multivariate in constant evaluation)
  enumerates only unordered pairs $(\beta, \alpha - \beta)$, doubling pairs
  and counting diagonals once. At runtime for `M ≥ 2` it routes through the
  stencil-driven general product instead — the table walk's elimination of
  per-pair index arithmetic outweighs the halved multiplication count.
- `seriesCbrt` maintains $q = g^2$ incrementally, dropping the worst-case
  cost from $\mathcal{O}(N^3)$ to $\mathcal{O}(N^2)$.
- `seriesSqrt`'s ordered row walk needs no $|\beta| < d$ filter: the
  $\beta = \alpha$ entry reads `out[ai]`, which is still zero when the row
  is processed.

The symmetries don't change the math — they're computational identities that
follow directly from the closed-form recurrence and the graded-lexicographic
ordering of the flat index.

---

## Sparse Cauchy

`sparse_cauchy` exploits the parallel-sorted-vector representation of
`storage::Sparse`. For inputs with `nnz_a` and `nnz_b` nonzeros, the kernel
enumerates pairs in roughly $\mathcal{O}(\text{nnz}_a \cdot \text{nnz}_b)$
operations (rather than the dense $\mathcal{O}(\binom{N+M}{M}^2)$) and
inserts each contribution into a result buffer keyed by flat index. The result
is materialised into a sorted-index representation at the end.

Sparse add/sub uses the standard two-pointer merge over the sorted index
arrays — $\mathcal{O}(\text{nnz}_a + \text{nnz}_b)$.

---

## Tests as the canonical executable spec

Every kernel has a paired test in `tests/kernels/`:

| Test | What it pins |
|---|---|
| `test_cauchy_dense.cpp`         | Dense `cauchy` against a direct convolution baseline |
| `test_cauchy_unroll_diff.cpp`   | Unrolled vs non-unrolled univariate Cauchy agree |
| `test_cauchy_stencil_diff.cpp`  | Stencil vs non-stencil multivariate Cauchy agree |
| `test_sparse_cauchy.cpp`        | Sparse Cauchy against the Dense Cauchy reference |

Treat these tests as the canonical executable specification: when in doubt
about a recurrence's expected behavior, they show inputs and the exact
expected outputs.
