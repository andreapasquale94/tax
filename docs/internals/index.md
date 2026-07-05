# Internals

This section documents the implementation strategy of **tax** — the layers
above the public API that make the library both fast and ergonomic.

| Page | Topic |
|---|---|
| [Architecture](architecture.md) | Layering of storage, kernels, operators, and Eigen |
| [Kernels & Recurrences](kernels.md) | Where each mathematical operation is implemented and how |
| [Recurrence Relations](recurrences.md) | The univariate and multivariate recurrence math for every operation |
| [Map Inversion](map-inversion.md) | Picard-iteration inversion of polynomial maps (`tax::invert`) |
| [Taylor Models](taylor-models.md) | Outward-rounded intervals, remainder propagation, Lagrange tails, and the ch. 4 vs ch. 5 boundary |

The headline ideas:

1. **Compile-time shape.** $(N, M)$ drive every loop bound. The optimizer
   sees through the storage container, the recurrences, and the Eigen matrix
   wrappers because none of them allocate at runtime.

2. **Storage policy.** A single class template is partial-specialised on
   `storage::Dense` vs `storage::Sparse`. The two share the kernel surface and
   differ only in how coefficients are stored and how the surface is wired.

3. **Eager operators, hot-path kernels.** Free-function operators (`+`, `*`,
   `sin`, …) materialise into a fresh `TaylorExpansion` by calling a single
   kernel. Each kernel writes its result coefficient-by-coefficient into a raw
   buffer, with no temporary `TaylorExpansion` objects between the user
   expression and the final answer.

4. **Kernels are recurrences.** Every transcendental and algebraic function is
   computed by a degree-by-degree recurrence relation derived from the
   classical chain or product rules. Univariate and multivariate share the
   same algorithm via `forEachSubIndex<M>(alpha, lo, hi, callback)`; the
   univariate path is special-cased through `if constexpr (M == 1)` for tight
   scalar loops.
