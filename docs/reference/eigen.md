# Eigen API Reference

All helpers live in `namespace tax` (header `tax/la.hpp`, transitively
included by `tax/tax.hpp`). These helpers are templated over the `IndexScheme`
parameter; where signatures below write `TaylorExpansion<T, N, M, S>` or use
`<T, N, M, S>` shorthands, they denote the isotropic `TE<N,M>` form
(`TaylorExpansion<T, IsotropicScheme<N,M>, S>`). The same helpers work unchanged
for `MixedTE` — the scheme is deduced from the type.

---

## NumTraits

```cpp
namespace Eigen {
    template <typename T, typename Scheme, typename Storage>
    struct NumTraits<tax::TaylorExpansion<T, Scheme, Storage>> : NumTraits<T> {
        using Self = tax::TaylorExpansion<T, Scheme, Storage>;
        using Real = Self;
        using NonInteger = Self;
        using Nested = Self;
        enum {
            IsComplex             = 0,
            IsInteger             = 0,
            IsSigned              = 1,
            RequireInitialization = 1,
            ReadCost              = int(Scheme::nCoeff),
            AddCost               = int(Scheme::nCoeff),
            MulCost               = int(Scheme::nCoeff) * int(Scheme::nCoeff),
        };
    };
}
```

Brings `TaylorExpansion` into Eigen's scalar trait system. With this in place,
`Eigen::Matrix<tax::TE<N, M>, R, C>` is a first-class type.

---

## Variable construction

```cpp
// Build all M coordinate variables as Eigen::Matrix<TE, M, 1>.
//   x0 may be an Eigen vector of size M (or Dynamic with size() == M).
template <typename TE, typename Derived>
[[nodiscard]] auto variables(const Eigen::MatrixBase<Derived>& x0);
```

Returns `Eigen::Matrix<TE, TE::vars_v, 1>` with entry $i$ equal to
`TE::variable<i>(x0)`.

---

## Constant extraction

```cpp
// Extract the constant term of each entry in F.
//   Requires Derived::Scalar to be a TaylorExpansion.
template <typename Derived>
    requires(/* Derived::Scalar is a TaylorExpansion */)
[[nodiscard]] auto value(const Eigen::MatrixBase<Derived>& F);
```

Returns an Eigen matrix of `T` of the same shape as `F`.

---

## Evaluation

```cpp
// Scalar form: f.eval(dx)
template <typename T, int N, int M, typename S, typename DxDerived>
[[nodiscard]] T eval(const TaylorExpansion<T, N, M, S>& f,
                     const Eigen::MatrixBase<DxDerived>& dx) noexcept;

// Vector / matrix form: element-wise eval at a shared dx
template <typename Derived, typename DxDerived>
    requires(/* Derived::Scalar is a TaylorExpansion */)
[[nodiscard]] auto eval(const Eigen::MatrixBase<Derived>& F,
                        const Eigen::MatrixBase<DxDerived>& dx);
```

`dx` must have compile-time size equal to `TE::vars_v` (or `Eigen::Dynamic`
with the matching runtime size).

---

## Element-wise compile-time partial

```cpp
// Extract ∂^|Alpha| F_i / ∂x^Alpha at the expansion point, entry-by-entry.
//   Usage: tax::derivative<1, 0>(F) → matrix of dF_i/dx_0
template <int... Alpha, typename Derived>
    requires(/* Derived::Scalar is a TaylorExpansion */)
[[nodiscard]] auto derivative(const Eigen::MatrixBase<Derived>& F);
```

---

## Gradient, Hessian, Jacobian

```cpp
// Gradient of a scalar TE  →  Eigen::Matrix<T, M, 1>
template <typename T, int N, int M, typename S>
[[nodiscard]] Eigen::Matrix<T, M, 1>
gradient(const TaylorExpansion<T, N, M, S>& f) noexcept;

// Hessian of a scalar TE   →  Eigen::Matrix<T, M, M>
template <typename T, int N, int M, typename S>
[[nodiscard]] Eigen::Matrix<T, M, M>
hessian(const TaylorExpansion<T, N, M, S>& f) noexcept;

// Jacobian of a vector TE  →  Eigen::Matrix<T, K, M>, J(i, j) = ∂F_i/∂x_j
template <typename Derived>
    requires(/* Derived::Scalar is a TaylorExpansion */)
[[nodiscard]] auto jacobian(const Eigen::MatrixBase<Derived>& F);
```

The Jacobian helper supports any compile-time `K = Derived::SizeAtCompileTime`,
including dynamic-size vectors.

---

## Vector norms

```cpp
// ||v||_P^Q for a vector of expansions. Q defaults to 1 (the plain P-norm),
// P to the Euclidean 2-norm. Input is an Eigen column vector or any range of
// dense/named/mixed expansions. Raises the accumulated power-sum ONCE to Q/P.
template <int P = 2, int Q = 1, /* Eigen vector or range of expansions */>
[[nodiscard]] auto norm(const V& v) noexcept;
```

`norm(v)` = `sqrt(Σ vᵢ²)`, `norm<3>(v)` the 3-norm, `norm<2,-3>(v)` the
`1/‖v‖³` gravity kernel (bit-identical to `invSqrtPow<3>(Σ vᵢ²)`). See
[Guide / Fused Operations](../guide/fused.md#vector-norms-normp-q).

---

## Vector algebra

All operate on Eigen vectors/matrices of expansions (dense `TE`, named `NE`,
or mixed `MTE`); results are full Taylor series. Reachable as `tax::…`.

```cpp
// a · b  (vector · vector → scalar expansion)
[[nodiscard]] auto dot(const Eigen::MatrixBase<LA>& a, const Eigen::MatrixBase<LB>& b);

// A · b  (matrix · vector → vector). A may be a matrix of expansions OR a
// constant real matrix (a linear map — the constant case scalar-multiplies,
// skipping the Cauchy product).
[[nodiscard]] auto dot(const Eigen::MatrixBase<MA>& A, const Eigen::MatrixBase<VB>& b);

// a × b  (3-vectors)
[[nodiscard]] auto cross(const Eigen::MatrixBase<LA>& a, const Eigen::MatrixBase<LB>& b);

// angle between a and b = acos((a·b)/(|a||b|)). Requires |cos| < 1 at x0.
[[nodiscard]] auto angle(const Eigen::MatrixBase<LA>& a, const Eigen::MatrixBase<LB>& b);

// v / |v|   (requires dot(v,v).value() > 0)
[[nodiscard]] auto unitvec(const Eigen::MatrixBase<D>& v);

// (a × b) / |a × b|   — the unit normal to the plane of a and b
[[nodiscard]] auto unitcross(const Eigen::MatrixBase<LA>& a, const Eigen::MatrixBase<LB>& b);

// projection of a onto the direction d:  (a·d / d·d) d
[[nodiscard]] auto projvec(const Eigen::MatrixBase<LA>& a, const Eigen::MatrixBase<LD>& d);

// projection of a onto the plane with normal n:  a − projvec(a, n)
[[nodiscard]] auto projplane(const Eigen::MatrixBase<LA>& a, const Eigen::MatrixBase<LN>& n);
```

Fusion: norms enter only as `dot(v, v)`, and where two norms multiply (`angle`,
the unit vectors) the reciprocal square root is taken once over the product —
one recurrence pass, not two.

---

## Map inversion

```cpp
// Formally invert a square polynomial map F : R^M → R^M via Picard iteration.
//
// Returns a map G such that G ∘ F = identity up to order N. The linear part of
// F must be invertible; otherwise throws std::invalid_argument.
//
// The input's constant terms are ignored — the inversion operates on the
// non-constant (perturbation) part. The returned map has zero constant.
template <typename Derived>
    requires(/* Derived::Scalar is a TaylorExpansion */)
[[nodiscard]] auto invert(const Eigen::MatrixBase<Derived>& map);
```

The input vector size must equal `TE::vars_v` (statically or at runtime).

---

## Statistical moments

`namespace tax::la` (header `tax/la/moments.hpp`), for dense, isotropic
expansions only. Assumes the expansion's formal variables are i.i.d. standard
normal — see [Guide / Eigen Integration](../guide/eigen.md#statistical-moments)
and [Internals / Statistical Moments](../internals/moments.md). The map
dimension `D = Derived::SizeAtCompileTime` must be a compile-time constant (a
dynamic-size input is rejected with a `static_assert`); returns are fixed-size.
The rank-3/4 returns use `Eigen::TensorFixedSize` from
`unsupported/Eigen/CXX11/Tensor` and are fully symmetric under any index
permutation.

```cpp
// E[F] — mean of a vector map F(x), x ~ N(0, I) i.i.d.  → fixed-size Eigen vector (D x 1).
template <typename Derived>
    requires(/* Derived::Scalar is a dense, isotropic TaylorExpansion */)
[[nodiscard]] auto mean(const Eigen::MatrixBase<Derived>& F);

// Cov(F_i, F_j) — fixed-size D x D covariance matrix (Eigen::Matrix<T, D, D>).
template <typename Derived>
    requires(/* ... */)
[[nodiscard]] auto covariance(const Eigen::MatrixBase<Derived>& F);

// S_ijk = E[(F_i-mu_i)(F_j-mu_j)(F_k-mu_k)]
//   → Eigen::TensorFixedSize<T, Eigen::Sizes<D, D, D>>.  Access: skewnessTensor(F)(i, j, k).
template <typename Derived>
    requires(/* ... */)
[[nodiscard]] auto skewnessTensor(const Eigen::MatrixBase<Derived>& F);

// K_ijkl = E[(F_i-mu_i)(F_j-mu_j)(F_k-mu_k)(F_l-mu_l)]
//   → Eigen::TensorFixedSize<T, Eigen::Sizes<D, D, D, D>>.  Access: kurtosisTensor(F)(i, j, k, l).
template <typename Derived>
    requires(/* ... */)
[[nodiscard]] auto kurtosisTensor(const Eigen::MatrixBase<Derived>& F);

// kurtosisTensor(F) minus the jointly-Gaussian Isserlis baseline
// Cov_ij*Cov_kl + Cov_ik*Cov_jl + Cov_il*Cov_jk — a non-Gaussianity measure.
//   → Eigen::TensorFixedSize<T, Eigen::Sizes<D, D, D, D>>.
template <typename Derived>
    requires(/* ... */)
[[nodiscard]] auto excessKurtosisTensor(const Eigen::MatrixBase<Derived>& F);
```
