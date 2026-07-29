# Guide

Task-oriented walkthroughs for getting real work done with `tax`. Each page is
how-to: create something, compute something, read a result out.

| Page | What you'll do |
|---|---|
| [Background](background.md) | The theory: truncated Taylor polynomials, graded-lex ordering, convergence |
| [Variables & Expressions](expressions.md) | Create variables, build expressions with arithmetic and math functions, run pipelines at compile time |
| [Fused Operations](fused.md) | Compute coupled pairs (`sinCos`, `sqrtInvSqrt`, `expSinCos`) and half-integer powers in one pass |
| [Extracting Results](results.md) | Read out values, coefficients, derivatives, and evaluate the polynomial |
| [Coefficient Storage](storage.md) | How coefficients are laid out, and how the shape $\binom{N+M}{M}$ grows |
| [Named & Mixed-Order Expansions](named.md) | Attach named axes, compose across axis sets, slice and differentiate by name; give each axis its own order |
| [Eigen Integration](eigen.md) | Use `TaylorExpansion` inside Eigen vectors and matrices |

New here? Start with [Getting Started](../getting_started.md), then come back.
For exact signatures see [Reference](../reference/index.md).
