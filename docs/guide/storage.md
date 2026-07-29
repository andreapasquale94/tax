# Coefficient Storage

`TaylorExpansion<T, Scheme>` keeps its coefficients in a single dense
`std::array`. The `Scheme` parameter encodes the monomial index set (see
below); `TE<N, M>` is the shorthand for the isotropic case
`TaylorExpansion<double, IsotropicScheme<N,M>>`.

```cpp
template <typename T, typename Scheme>   // Scheme = IsotropicScheme<N,M> for TE<N,M>
class TaylorExpansion;
```

- Storage: `std::array<T, C(N+M, M)>` — contiguous, stack-resident, no heap
  allocation, `constexpr` on every accessor.
- Coefficient order: graded-lexicographic flat index.
- All kernels iterate flat indices directly, including the unrolled
  (`TAX_USE_UNROLL`) and precomputed stencil (`TAX_USE_STENCIL`) Cauchy paths.
- Comparison operators, `+=`/`-=`/`*=`/`/=`, in-place updates: all defined.

---

## The shape

$$
S = \binom{N + M}{M}
$$

is the number of coefficients each expansion carries.

Reference table:

| $N \backslash M$ | 1 | 2 | 3 | 4 | 5 | 6 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 2  | 3  | 4   | 5    | 6    | 7    |
| 3  | 4  | 10 | 20  | 35   | 56   | 84   |
| 5  | 6  | 21 | 56  | 126  | 252  | 462  |
| 8  | 9  | 45 | 165 | 495  | 1287 | 3003 |
| 10 | 11 | 66 | 286 | 1001 | 3003 | 8008 |

$S$ grows quickly in both $N$ and $M$: for $N \ge 8$ and $M \ge 4$ an
expansion starts to push the stack frame budget, so keep the truncation order
and the number of live variables as low as the problem allows. *Mixed-order*
axes are the tool for that — give each axis only the order it needs instead of
paying the isotropic worst case (see
[Named & Mixed-Order Expansions](named.md#anisotropic-axes-per-axis-orders)).

---

## Numerical agreement

The generic loop, unrolled and stencil Cauchy paths implement the same
recurrence. Round-off differences on the order of $10^{-12}$ (double
precision) are expected from ordering effects in the Cauchy sum; the agreement
is tested by the `test_cauchy_unroll_diff` and `test_cauchy_stencil_diff`
suites.
