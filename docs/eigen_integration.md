# Eigen Integration

TAX provides adapters for working with Eigen vectors, matrices, and tensors of DA objects. Eigen is a required dependency.

Core Eigen integration headers:
- `tax/eigen/variables.hpp` for `tensor(...)` and `variables(...)`
- `tax/eigen/value.hpp` for `value(...)`
- `tax/eigen/derivative.hpp` for `derivative(...)`, `gradient(...)`, and `jacobian(...)`
- `tax/eigen/eval.hpp` for `eval(...)`

## Type Aliases

```cpp
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

template <int N, int M>
using TEnVec = VecT<TEn<N, M>, M>;

template <int N, int M>
using TEnRowVec = RowVecT<TEn<N, M>, M>;
```

## Creating DA Variables from Eigen Vectors

### `tax::tensor<DA>(x0)`

Converts an Eigen vector (or matrix) into a same-shaped Eigen container of DA variables:

```cpp
Eigen::Vector3d x0{1.0, 2.0, 3.0};
auto vars = tax::tensor<TEn<2, 3>>(x0);
// vars is Eigen::Matrix<TEn<2,3>, 3, 1>
// vars(0) = TEn<2,3>::variable<0>(x0)
// vars(1) = TEn<2,3>::variable<1>(x0)
// vars(2) = TEn<2,3>::variable<2>(x0)
```

Requirements:
- The input must be compile-time sized.
- The total number of elements must equal $M$.

### `tax::variables<DA>(x0)`

Returns a `std::tuple` of DA variables from an Eigen vector, for use with structured bindings:

```cpp
Eigen::Vector3d x0{1.0, 2.0, 3.0};
auto [x, y, z] = tax::variables<TEn<2, 3>>(x0);
```

This is equivalent to calling `TEn<2,3>::variables(...)` but accepts an Eigen vector directly.

## Extracting Scalar Results

### `tax::value(container)`

Extracts the constant term from each DA element, returning a plain scalar Eigen container of the same shape:

```cpp
Eigen::Matrix<TEn<2, 3>, 3, 1> f = /* ... */;
Eigen::Vector3d v = tax::value(f);   // v(i) = f(i).value()
```

### `tax::eval(container, dx)`

Evaluates every DA element at a displacement, returning a scalar container:

```cpp
Eigen::Vector3d dx{0.01, 0.02, 0.03};
Eigen::Vector3d result = tax::eval(f, dx);   // result(i) = f(i).eval(dx)
```

The displacement `dx` can be:
- A scalar `T` (univariate DA)
- A `Input` (`std::array<T, M>`)
- An Eigen vector (automatically converted)

### `tax::eval(scalar_da, dx)`

For a single multivariate DA scalar (not in a container):

```cpp
TEn<2, 3> g = /* ... */;
Eigen::Vector3d dx{0.1, 0.2, 0.3};
double result = tax::eval(g, dx);
```

## Derivative Extraction

### `tax::derivative(container, alpha)` (runtime)

Extracts a partial derivative from each DA element:

```cpp
Eigen::Vector3d df_dx = tax::derivative(f, std::array{1, 0, 0});
```

### `tax::derivative(container, k)` (univariate shorthand)

For univariate DA containers, pass a single integer:

```cpp
Eigen::VectorXd df = tax::derivative(f, 1);   // first derivatives
```

### `tax::derivative<Alpha...>(container)` (compile-time)

```cpp
Eigen::Vector3d d2f_dxdy = tax::derivative<1, 1, 0>(f);
```

## Gradient and Jacobian

### `tax::gradient(f)`

Computes the gradient of a scalar DA at its expansion point:

```cpp
TEn<2, 3> f = /* scalar function of 3 variables */;
Eigen::Vector3d grad = tax::gradient(f);
// grad = [∂f/∂x₀, ∂f/∂x₁, ∂f/∂x₂]
```

Requires $N \ge 1$.

### `tax::jacobian(vec)`

Computes the Jacobian matrix of a vector-valued DA function:

```cpp
Eigen::Matrix<TEn<2, 3>, 3, 1> F = /* vector field */;
Eigen::Matrix3d J = tax::jacobian(F);
// J(i, j) = ∂Fᵢ/∂xⱼ
```

Returns a $K \times M$ matrix where $K$ is the number of components in the DA vector and $M$ is the number of variables.

## Derivative Objects

For multivariate DA ($M > 1$), TAX can extract higher-order derivative information at the expansion point.

### `tax::derivative<K>(f)` (compile-time order)

Return type depends on `K`:
- `K == 0`: scalar `T` (`f.value()`)
- `K == 1`: `Eigen::Matrix<T, M, 1>` (gradient)
- `K == 2`: `Eigen::Matrix<T, M, M>` (Hessian)
- `K >= 3`: `Eigen::Tensor<T, K, Eigen::RowMajor>`

Example:

```cpp
TEn<3, 2> f = /* ... */;

// Gradient vector (Dense)
auto grad = tax::derivative<1>(f);
// grad(i) = ∂f/∂xᵢ

// Hessian matrix (Dense)
auto hess = tax::derivative<2>(f);
// hess(i, j) = ∂²f/∂xᵢ∂xⱼ

// Third-order tensor (rank-3, 2×2×2)
auto third = tax::derivative<3>(f);
// third(i, j, k) = ∂³f/∂xᵢ∂xⱼ∂xₖ
```

All mixed-partial tensors are symmetric, e.g. `hess(i, j) == hess(j, i)`.

Requires `Eigen/CXX11/Tensor` (unsupported Eigen module).

## Complete Example

```cpp
#include <tax/tax.hpp>
#include <Eigen/Dense>
#include <iostream>

int main() {
    using DA3 = tax::TEn<3, 3>;

    // Expansion point
    Eigen::Vector3d x0{1.0, 0.5, 2.0};
    auto [x, y, z] = tax::variables<DA3>(x0);

    // Scalar function
    DA3 f = tax::sin(x * y) + tax::exp(z);

    // Gradient at expansion point
    Eigen::Vector3d grad = tax::gradient(f);
    std::cout << "gradient:\n" << grad << "\n";

    // Hessian
    auto H = tax::derivative<2>(f);
    std::cout << "H(0,1) = " << H(0, 1) << "\n";

    // Vector field and its Jacobian
    Eigen::Matrix<DA3, 3, 1> F;
    F(0) = x * x + y;
    F(1) = tax::sin(z);
    F(2) = x * y * z;
    Eigen::Matrix3d J = tax::jacobian(F);
    std::cout << "Jacobian:\n" << J << "\n";

    // Evaluate at a nearby point
    Eigen::Vector3d dx{0.01, -0.02, 0.03};
    Eigen::Vector3d Fval = tax::eval(F, dx);
    std::cout << "F(x0 + dx) ≈ " << Fval.transpose() << "\n";
}
```

## Tensor Overloads

All functions above (`value`, `derivative`, `eval`) also work with `Eigen::Tensor<TruncatedTaylorExpansionT<T,N,M>, Rank>` containers (rank $\ge 1$). The interface is identical.
