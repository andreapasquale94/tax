# Dense vs Sparse Storage

`TaylorExpansion<T, Scheme, Storage>` selects its coefficient layout via the
`Storage` policy. The `Scheme` parameter encodes the monomial index set (see
below); `TE<N, M>` is the shorthand for the isotropic case
`TaylorExpansion<double, IsotropicScheme<N,M>>`. Two storage policies ship today:

| Tag | Container | API alias | When it shines |
|---|---|---|---|
| `tax::storage::Dense`  | `std::array<T, C(N+M, M)>` — stack-allocated | `TE<N, M>` | Hot computational paths, low $N$ and $M$, most TE coefficients are nonzero |
| `tax::storage::Sparse` | Two parallel `std::vector`: sorted flat indices + values | `STE<N, M>` | High $N$ or $M$ with most coefficients = 0, structured polynomials |

Both share **identical user-facing API** — `value()`, `coeff()`,
`derivative()`, `eval()`, `deriv()`, `integ()`, arithmetic operators,
mathematical functions, Eigen helpers. Switching storage is a single template
parameter change.

---

## The shape

$$
S = \binom{N + M}{M}
$$

is the *maximum* number of coefficients. Dense always allocates $S$ slots;
sparse allocates only the actual nonzeros (`nnz()`).

Reference table:

| $N \backslash M$ | 1 | 2 | 3 | 4 | 5 | 6 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 2  | 3  | 4   | 5    | 6    | 7    |
| 3  | 4  | 10 | 20  | 35   | 56   | 84   |
| 5  | 6  | 21 | 56  | 126  | 252  | 462  |
| 8  | 9  | 45 | 165 | 495  | 1287 | 3003 |
| 10 | 11 | 66 | 286 | 1001 | 3003 | 8008 |

For $N \ge 8$ and $M \ge 4$, Dense storage starts to push the stack frame
budget; sparse becomes attractive precisely when monomial population stays
low.

---

## Dense

```cpp
template <typename T, typename Scheme>   // Scheme = IsotropicScheme<N,M> for TE<N,M>
class TaylorExpansion<T, Scheme, storage::Dense>;
```

- Storage: `std::array<T, C(N+M, M)>` — contiguous, stack-resident, no heap
  allocation, `constexpr` on every accessor.
- Coefficient order: graded-lexicographic flat index.
- All kernels iterate flat indices directly — the unrolled (`TAX_USE_UNROLL`)
  and precomputed stencil (`TAX_USE_STENCIL`) Cauchy paths only apply to
  Dense.
- Comparison operators, `+=`/`-=`/`*=`/`/=`, in-place updates: all defined.

Use Dense when:

- $N \cdot M$ is small enough that the stack frame fits;
- Almost every coefficient ends up nonzero (typical for general
  transcendental expressions);
- You want `constexpr` evaluation or strictest control over per-step cost.

---

## Sparse

```cpp
template <typename T, typename Scheme>   // Scheme = IsotropicScheme<N,M> for STE<N,M>
class TaylorExpansion<T, Scheme, storage::Sparse>;
```

!!! note "Sparse is isotropic-only"
    The sparse specialisation is constrained to isotropic schemes
    (`IsotropicScheme<N,M>`). The anisotropic `MixedTE` (see
    [Named & Mixed-Order Expansions](named.md#anisotropic-axes-per-axis-orders)) is dense-only.

- Storage: two parallel `std::vector<uint32_t> idx_` and `std::vector<T> val_`
  kept strictly sorted by flat index.
- `nnz()` returns the current support size; `support()` and `values()` expose
  read-only `std::span`s.
- Addition / subtraction is an $O(\text{nnz}_a + \text{nnz}_b)$ two-pointer
  merge; element lookup is $O(\log \text{nnz})$ via binary search.
- Multiplication uses the dedicated sparse Cauchy kernel
  (`tax/kernels/sparse_cauchy.hpp`).
- **Full API parity with Dense, evaluated natively.** Arithmetic (including
  `+=`/`-=`/`*=`/`/=`), the differential/evaluation members (`deriv`, `integ`,
  `eval`, `gradient`, `hessian`), the complete math surface (`exp`, `log`,
  `sin`/`cos`/`tan` and inverses, hyperbolics, `erf`, every `pow`/`halfPow`/
  `atan2` spelling), and the Eigen `tax::la` helpers (`gradient`, `hessian`,
  `jacobian`, `norm`, `dot`, `invert`, …) all accept `STE`. The transcendental
  kernels use forward-substitution drivers (`tax/kernels/sparse_subs.hpp`) that
  walk **only the additive closure of the input's support** — never the full
  dense monomial set — so a function of one axis never populates the others.
  (The `sinCos`/`sqrtInvSqrt`/`expSinCos` fused family is the one exception: it
  materialises through Dense, as it is a convenience wrapper over the same
  results.)
- A conversion helper is provided:

```cpp
tax::STE<5, 2> s = /* ... */;   // TaylorExpansion<double, IsotropicScheme<5,2>, Sparse>
auto d = s.dense();              // → TaylorExpansion<double, IsotropicScheme<5,2>, Dense>
```

Use Sparse when:

- The polynomial is dominated by zeros — e.g. a single monomial $x^k$,
  bilinear forms, or polynomials engineered to stay sparse;
- $N$ or $M$ is large enough that Dense storage exceeds a few kilobytes
  per object;
- You're storing many polynomials in memory and total RAM matters more than
  per-op cost.

---

## Numerical agreement

The Sparse and Dense kernels implement the same recurrences. Round-off
differences on the order of $10^{-12}$ (double precision) are expected from
ordering effects in the Cauchy sum but the agreement is tested via the
`testCauchyStencilDiff` and `test_sparse_*` suites.

---

## Choosing in practice

| You're doing | Use |
|---|---|
| ODE integration (`tax::ode`)              | Dense (`TE<N>`) — kernels are tuned for it |
| One-off symbolic derivatives, low $N$   | Dense |
| Reading polynomials produced elsewhere with structural zeros | Sparse |
| Storing thousands of polynomials in memory at $N \ge 8$ | Sparse |
| Building a polynomial map of a sparse system (e.g. $f = x_1 \cdot x_7$) | Sparse |

When in doubt, write the code template-generic on the storage tag and switch
between `TE` and `STE` to compare wall time on your actual problem.

---

## Sparse storage drop-in

The factories on `STE<N, M>` mirror those on `TE<N, M>`; the storage layout
differs but the API is identical.

```cpp
using STE2 = tax::STE<5, 2>;
const std::array<double, 2> p{0.0, 0.0};
auto x = STE2::variable<0>(p);
auto y = STE2::variable<1>(p);

STE2 f = x*x + y*y;                  // only 2 nonzeros stored
auto nnz = f.nnz();                  // = 2

auto fd = f.dense();                 // → TaylorExpansion<…, Dense> conversion
```
