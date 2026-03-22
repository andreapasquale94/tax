# API Reference

This page documents all public functions and type aliases in the Vector (Eigen)
integration module.

---

## Type Aliases

**Header:** `tax/eigen/types.hpp`

```cpp
namespace tax {

template <typename T, int N, int M, int Rows, int Cols>
using Mat = Eigen::Matrix<TruncatedTaylorExpansionT<T, N, M>, Rows, Cols>;

template <typename Scalar, int Size>
using VecT = Eigen::Matrix<Scalar, Size, 1>;

template <typename Scalar, int Size>
using RowVecT = Eigen::Matrix<Scalar, 1, Size>;

template <int N, int Size>
using TEVec = VecT<TE<N>, Size>;

template <int N, int Size>
using TERowVec = RowVecT<TE<N>, Size>;

template <int N, int M, int Size = M>
using TEnVec = VecT<TEn<N, M>, Size>;

template <int N, int M, int Size = M>
using TEnRowVec = RowVecT<TEn<N, M>, Size>;

}
```

| Alias | Description |
|-------|-------------|
| `Mat<T, N, M, Rows, Cols>` | Eigen matrix with `TruncatedTaylorExpansionT<T, N, M>` scalar type |
| `VecT<Scalar, Size>` | Generic column vector `Eigen::Matrix<Scalar, Size, 1>` |
| `RowVecT<Scalar, Size>` | Generic row vector `Eigen::Matrix<Scalar, 1, Size>` |
| `TEVec<N, Size>` | Univariate TTE column vector (`TE<N>` elements) |
| `TERowVec<N, Size>` | Univariate TTE row vector (`TE<N>` elements) |
| `TEnVec<N, M, Size>` | Multivariate TTE column vector (`TEn<N, M>` elements, default `Size = M`) |
| `TEnRowVec<N, M, Size>` | Multivariate TTE row vector (`TEn<N, M>` elements, default `Size = M`) |

---

## NumTraits Specialization

**Header:** `tax/eigen/num_traits.hpp`

```cpp
namespace Eigen {

template <typename T, int N, int M>
struct NumTraits<tax::TruncatedTaylorExpansionT<T, N, M>> : NumTraits<T> {
    using Real       = tax::TruncatedTaylorExpansionT<T, N, M>;
    using NonInteger = tax::TruncatedTaylorExpansionT<T, N, M>;
    using Literal    = tax::TruncatedTaylorExpansionT<T, N, M>;
    using Nested     = tax::TruncatedTaylorExpansionT<T, N, M>;

    enum {
        IsComplex             = 0,
        IsInteger             = 0,
        IsSigned              = 1,
        RequireInitialization = 1,
        ReadCost = NumTraits<T>::ReadCost,
        AddCost  = NumTraits<T>::AddCost,
        MulCost  = NumTraits<T>::MulCost
    };
};

}
```

This specialization allows Eigen to use TTE types as the scalar type in its
matrix and vector templates.

---

## Variable Creation

**Header:** `tax/eigen/variables.hpp`

### `tax::vector`

```cpp
template <typename TTE, typename Derived>
[[nodiscard]] auto vector(const Eigen::DenseBase<Derived>& x0) noexcept;
```

Converts an Eigen matrix or vector into a same-shaped container of TTE
variables. Each element becomes the TTE variable for its corresponding
coordinate.

| Parameter | Description |
|-----------|-------------|
| `TTE` | The TTE type, e.g., `TEn<3, 2>`. `M` must equal the total number of elements in `x0`. |
| `x0` | Compile-time-sized Eigen matrix/vector with \(M\) entries. |

**Returns:** Eigen matrix of same shape with TTE variable entries.

**Constraints:**

- `x0` must have compile-time-known dimensions.
- The total number of elements (`Rows * Cols`) must equal `M`.

### `tax::variables`

```cpp
template <typename TTE, typename Derived>
[[nodiscard]] auto variables(const Eigen::DenseBase<Derived>& x0) noexcept;
```

Creates TTE variables from an Eigen vector and returns them as a tuple,
suitable for structured bindings.

| Parameter | Description |
|-----------|-------------|
| `TTE` | The TTE type, e.g., `TEn<3, 3>`. `M` must match the vector size. |
| `x0` | Eigen vector with \(M\) entries. |

**Returns:** `std::tuple<TTE, TTE, ..., TTE>` with \(M\) elements.

**Example:**

```cpp
Eigen::Vector3d x0{1.0, 2.0, 3.0};
auto [x, y, z] = tax::variables<tax::TEn<3, 3>>(x0);
```

---

## Value Extraction

**Header:** `tax/eigen/value.hpp`

### Scalar overload

```cpp
template <typename T, int N, int M>
[[nodiscard]] constexpr T value(
    const TruncatedTaylorExpansionT<T, N, M>& f) noexcept;
```

Returns the constant term (zeroth-order coefficient) of a scalar TTE.

### Dense container overload

```cpp
template <typename Derived>
[[nodiscard]] auto value(const Eigen::DenseBase<Derived>& t);
```

Extracts the constant term from each TTE element of an Eigen matrix or
vector.

**Returns:** Eigen matrix/vector of same shape with scalar type `T`.

**Constraints:** `Derived::Scalar` must be a TTE type.

### Tensor overload

```cpp
template <typename T, int N, int M, int Rank>
[[nodiscard]] auto value(
    const Eigen::Tensor<TruncatedTaylorExpansionT<T, N, M>, Rank>& t);
```

Extracts the constant term from each TTE element of an Eigen tensor.

**Returns:** `Eigen::Tensor<T, Rank>` with the same dimensions.

**Constraints:** `Rank >= 1`.

---

## Evaluation

**Header:** `tax/eigen/eval.hpp`

### Scalar TTE with generic displacement

```cpp
template <typename T, int N, int M, typename Dx>
[[nodiscard]] constexpr T eval(
    const TruncatedTaylorExpansionT<T, N, M>& f, const Dx& dx) noexcept;
```

Evaluates a scalar TTE at displacement `dx`. The displacement type must be
accepted by `TruncatedTaylorExpansionT::eval` (scalar for univariate, array
for multivariate).

### Scalar TTE with Eigen displacement

```cpp
template <typename T, int N, int M, typename Derived>
[[nodiscard]] T eval(
    const TruncatedTaylorExpansionT<T, N, M>& f,
    const Eigen::DenseBase<Derived>& dx) noexcept;
```

Evaluates a multivariate scalar TTE at a displacement given as an Eigen
vector.

| Parameter | Description |
|-----------|-------------|
| `f` | Multivariate TTE polynomial (\(M > 1\)). |
| `dx` | Eigen vector with \(M\) entries. |

**Returns:** `T` -- the evaluated polynomial value.

### Dense container overload

```cpp
template <typename Derived, typename Dx>
[[nodiscard]] auto eval(
    const Eigen::DenseBase<Derived>& f, const Dx& dx);
```

Evaluates each TTE element of an Eigen matrix/vector at displacement `dx`.
The displacement can be a scalar, `Input` array, or Eigen vector.

**Returns:** Eigen matrix/vector of same shape with scalar type `T`.

### Tensor overload

```cpp
template <typename T, int N, int M, int Rank, typename Dx>
[[nodiscard]] auto eval(
    const Eigen::Tensor<TruncatedTaylorExpansionT<T, N, M>, Rank>& t,
    const Dx& dx);
```

Evaluates each TTE element of an Eigen tensor at displacement `dx`.

**Returns:** `Eigen::Tensor<T, Rank>` with the same dimensions.

### Vectorized univariate evaluation

```cpp
template <typename T, int N, int Dim>
[[nodiscard]] Eigen::Matrix<T, Dim, 1> eval(
    const Eigen::Matrix<TruncatedTaylorExpansionT<T, N, 1>, Dim, 1>& f,
    T dx) noexcept;
```

Fast path for univariate TTE column vectors. Assembles the coefficient
matrix \(C\) and computes the result as a single matrix-vector product
\(C \cdot [1, h, h^2, \ldots, h^N]^\top\).

| Parameter | Description |
|-----------|-------------|
| `f` | Column vector of univariate TTE elements. |
| `dx` | Scalar displacement. |

**Returns:** `Eigen::Matrix<T, Dim, 1>` of evaluated values.

---

## Derivatives

**Header:** `tax/eigen/derivative.hpp`

### Element-wise derivative (runtime multi-index)

```cpp
template <typename Derived, std::size_t M>
[[nodiscard]] auto derivative(
    const Eigen::DenseBase<Derived>& t,
    const std::array<int, M>& alpha);
```

Extracts the partial derivative specified by multi-index `alpha` from each
TTE element.

| Parameter | Description |
|-----------|-------------|
| `t` | Eigen matrix/vector of TTE elements. |
| `alpha` | Multi-index `{a_1, ..., a_M}` specifying the derivative order per variable. |

**Returns:** Eigen matrix/vector of same shape with scalar type `T`.

### Element-wise derivative (univariate shorthand)

```cpp
template <typename Derived>
[[nodiscard]] auto derivative(
    const Eigen::DenseBase<Derived>& t, int k);
```

Shorthand for univariate TTE containers. Extracts the \(k\)-th derivative
from each element.

| Parameter | Description |
|-----------|-------------|
| `t` | Eigen matrix/vector of univariate TTE elements (\(M = 1\)). |
| `k` | Derivative order (0 = value, 1 = first derivative, ...). |

### Element-wise derivative (compile-time multi-index)

```cpp
template <int... Alpha, typename Derived>
[[nodiscard]] auto derivative(const Eigen::DenseBase<Derived>& t);
```

Extracts the partial derivative specified by compile-time multi-index
`Alpha...` from each TTE element.

| Template Parameter | Description |
|--------------------|-------------|
| `Alpha...` | Derivative orders for each variable. Must have exactly \(M\) values. |

### Gradient

```cpp
template <typename T, int N, int M>
[[nodiscard]] auto gradient(
    const TruncatedTaylorExpansionT<T, N, M>& f);
```

Computes the gradient of a scalar TTE at its expansion point.

**Returns:** `Eigen::Matrix<T, M, 1>` containing
\([\partial f / \partial x_1, \ldots, \partial f / \partial x_M]^\top\).

**Requires:** \(N \ge 1\).

### Jacobian

```cpp
template <typename Derived>
[[nodiscard]] auto jacobian(const Eigen::DenseBase<Derived>& vec);
```

Computes the Jacobian matrix of a vector-valued TTE function at its
expansion point.

| Parameter | Description |
|-----------|-------------|
| `vec` | Eigen vector of \(K\) TTE elements with \(M\) variables. |

**Returns:** `Eigen::Matrix<T, K, M>` where entry \((i, j)\) is
\(\partial F_i / \partial x_j\).

### Higher-order derivative object

```cpp
template <int K, typename T, int N, int M>
[[nodiscard]] auto derivative(
    const TruncatedTaylorExpansionT<T, N, M>& f);
```

Builds the order-\(K\) derivative object of a scalar TTE at its expansion
point.

| \(K\) | Return Type | Description |
|-------|-------------|-------------|
| 0 | `T` | Function value \(f(\mathbf{x}_0)\) |
| 1 | `Eigen::Matrix<T, M, 1>` | Gradient vector |
| 2 | `Eigen::Matrix<T, M, M>` | Hessian matrix |
| \(\ge 3\) | `Eigen::Tensor<T, K, Eigen::RowMajor>` | Rank-\(K\) symmetric derivative tensor |

**Requires:** \(M > 1\), \(0 \le K \le N\).

### Tensor overloads

All `derivative` overloads for `Eigen::DenseBase` have corresponding
overloads for `Eigen::Tensor`:

```cpp
// Runtime multi-index
template <typename T, int N, int M, int Rank>
[[nodiscard]] auto derivative(
    const Eigen::Tensor<TruncatedTaylorExpansionT<T, N, M>, Rank>& t,
    const std::array<int, std::size_t(M)>& alpha);

// Univariate shorthand
template <typename T, int N, int Rank>
[[nodiscard]] auto derivative(
    const Eigen::Tensor<TruncatedTaylorExpansionT<T, N, 1>, Rank>& t,
    int k);

// Compile-time multi-index
template <int... Alpha, typename T, int N, int M, int Rank>
[[nodiscard]] auto derivative(
    const Eigen::Tensor<TruncatedTaylorExpansionT<T, N, M>, Rank>& t);
```

All require `Rank >= 1` and return `Eigen::Tensor<T, Rank>` with the same
dimensions.

---

## Map Inversion

**Header:** `tax/eigen/invert_map.hpp`

### `tax::invert`

```cpp
template <typename Derived>
[[nodiscard]] auto invert(const Eigen::DenseBase<Derived>& map_in);
```

Inverts a square Taylor map via Picard iteration.

| Parameter | Description |
|-----------|-------------|
| `map_in` | Eigen vector of \(M\) TTE components, each with \(M\) variables. |

**Returns:** Eigen vector of same shape containing the inverse map. Constant
terms of the input are stripped before inversion.

**Algorithm:**

1. Extract the Jacobian \(J\) and compute \(J^{-1}\).
2. Separate the nonlinear part \(\mathbf{N}(\mathbf{x}) = \mathbf{F}(\mathbf{x}) - J\mathbf{x}\).
3. Initialize \(\mathbf{G}_0 = J^{-1}\mathbf{u}\).
4. Iterate \(N - 1\) times:
   \(\mathbf{G}_{k+1} = J^{-1}(\mathbf{u} - \mathbf{N}(\mathbf{G}_k(\mathbf{u})))\).

**Throws:**

- `std::invalid_argument` if `map_in.size() != M` (size mismatch).
- `std::invalid_argument` if the linear part (Jacobian) is singular.

**Constraints:** `Derived::Scalar` must be a TTE type. The input must be a
vector expression.

### `tax::linear`

```cpp
template <typename DA, typename Mat>
[[nodiscard]] auto linear(
    const Mat& a,
    const Eigen::Matrix<DA, Mat::ColsAtCompileTime, 1>& vars);
```

Helper that applies a scalar matrix \(A\) to a TTE variable vector,
producing a TTE vector representing the linear map \(A\mathbf{x}\).
Used internally by `invert`.
