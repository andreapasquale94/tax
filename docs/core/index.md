# Core Module

The core module of **tax** provides truncated multivariate Taylor polynomials as first-class C++ objects. A single evaluation pass through any supported mathematical expression yields the function value and all partial derivatives up to a compile-time truncation order \(N\). The library uses expression templates for lazy evaluation: arithmetic and transcendental operations build a lightweight expression tree that is materialized only on assignment, eliminating intermediate temporary objects and enabling automatic sum/product flattening.

The central type, `TruncatedTaylorExpansionT<T, N, M>`, stores \(\binom{N+M}{M}\) Taylor coefficients in a fixed-size `std::array` with zero heap allocation. Coefficients follow **graded-lexicographic ordering** -- grouped first by total degree, then lexicographically within each grade. A comprehensive set of mathematical functions (trigonometric, hyperbolic, transcendental, algebraic, and special functions) is implemented via degree-by-degree recurrence relations, supporting both univariate and multivariate expansions.

## Key Headers

| Header | Contents |
|--------|----------|
| `tax/tax.hpp` | Umbrella header -- the only include users need |
| `tax/tte.hpp` | Core `TruncatedTaylorExpansionT<T, N, M>` class with constructors, variable factories, coefficient/derivative access, evaluation, differentiation, integration, norms, and in-place operators |
| `tax/operators.hpp` | Facade that pulls in all free-function operators and math functions |
| `tax/kernels.hpp` | Facade that pulls in all series computation kernels (recurrence relations) |
| `tax/expr/` | Expression template nodes: `Expr<>` CRTP base, binary/unary/sum/product expressions, arithmetic and function call nodes |
| `tax/kernels/` | Degree-by-degree recurrence implementations: `cauchy.hpp` (Cauchy product), `algebra.hpp` (reciprocal, sqrt, cbrt), `trigonometric.hpp` (sin, cos, tan, sinh, cosh, tanh and inverses), `transcendental.hpp` (exp, log, pow, erf, asin, acos, atan, inverse hyperbolics) |
| `tax/utils/` | Combinatorics, multi-index enumeration, flat-index mapping, type traits |
