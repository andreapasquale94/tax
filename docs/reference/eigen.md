# Eigen API Reference

All helpers live in `namespace tax` (header `tax/eigen.hpp`, transitively
included by `tax/tax.hpp`). These helpers are templated over the `IndexScheme`
parameter; where signatures below write `TaylorExpansion<T, N, M, S>` or use
`<T, N, M, S>` shorthands, they denote the isotropic `TE<N,M>` form
(`TaylorExpansion<T, IsotropicScheme<N,M>, S>`). The same helpers work unchanged
for `BoxTE` — the scheme is deduced from the type.

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
