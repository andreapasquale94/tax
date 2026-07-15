# Internals

This section documents the implementation strategy of **tax** — the layers
above the public API that make the library both fast and ergonomic.

| Page | Topic |
|---|---|
| [Architecture](architecture.md) | Layering of storage, kernels, operators, and Eigen |
| [Kernels & Recurrences](kernels.md) | Where each mathematical operation is implemented and how |
| [Recurrence Relations](recurrences.md) | The univariate and multivariate recurrence math for every operation |
| [Map Inversion](map-inversion.md) | Picard-iteration inversion of polynomial maps (`tax::invert`) |
| [Basis Conversion](basis-conversion.md) | Monomial ↔ Hermite / Chebyshev basis conversion |
| [Statistical Moments](moments.md) | Mean, covariance, skewness/kurtosis tensors via Isserlis' theorem |

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

4. **Kernels are recurrences — mostly two of them.** Every transcendental and
   algebraic function is computed by a degree-by-degree recurrence relation
   derived from the classical chain or product rules; nearly all of them are
   instances of the two shared drivers `seriesDerivProduct` /
   `seriesDerivQuotient` in `tax/kernels/algebra.hpp`. Pair shapes that share
   their recurrence work (`sinCos`, `sinhCosh`, `sqrtInvSqrt`, `expSinCos`)
   are fused into single coupled passes. The univariate path is special-cased
   for tight scalar loops; multivariate recurrences walk one shared
   decomposition table.

5. **The polynomial surface is `constexpr`.** Arithmetic, `square`, `cube`,
   `reciprocal`, integer `pow`, and the differential/evaluation accessors run
   in constant evaluation. The transcendental, root, and real-exponent
   functions seed their constant term with a libm call (`std::exp`, …) and are
   therefore runtime-only (see
   [Constant-term seeding](kernels.md#constant-term-seeding)).
