# Vector (Eigen) Module

The Eigen integration module enables using `TruncatedTaylorExpansionT` types inside
Eigen matrices, vectors, and tensors. This provides seamless interoperability between
the tax library and the Eigen linear algebra ecosystem, so you can build vector-valued
Taylor maps, extract Jacobians and Hessians, evaluate polynomial flows, and invert
nonlinear maps -- all with natural Eigen syntax.

## Requirements

- **Eigen 3.4+** must be available (found via `find_package(Eigen3)` or `CMAKE_PREFIX_PATH`).
- The unsupported Eigen Tensor module is used for higher-order derivative tensors
  (\(K \ge 3\)).

## Headers

The module is composed of seven headers, all under `include/tax/eigen/`:

| Header | Purpose |
|--------|---------|
| `tax/eigen/num_traits.hpp` | `Eigen::NumTraits` specialization for `TruncatedTaylorExpansionT`, allowing Eigen to treat TTE as a scalar type |
| `tax/eigen/types.hpp` | Convenience type aliases (`Mat`, `VecT`, `TEVec`, `TEnVec`, etc.) |
| `tax/eigen/variables.hpp` | Create TTE variables from Eigen vectors: `tax::vector` and `tax::variables` |
| `tax/eigen/value.hpp` | Extract the scalar constant term from TTE containers: `tax::value` |
| `tax/eigen/eval.hpp` | Evaluate TTE polynomials at displacements: `tax::eval` (including vectorized univariate path) |
| `tax/eigen/derivative.hpp` | Compute gradients, Jacobians, Hessians, and higher-order derivative tensors: `tax::gradient`, `tax::jacobian`, `tax::derivative` |
| `tax/eigen/invert_map.hpp` | Invert a square Taylor map via Picard iteration: `tax::invert` |

All public symbols live in `namespace tax`. Implementation details are in
`namespace tax::detail`.

## Quick Start

```cpp
#include <tax/tax.hpp>

// Expansion point
Eigen::Vector3d x0{1.0, 2.0, 3.0};

// Create multivariate TTE variables (order 4, 3 variables)
auto [x, y, z] = tax::variables<tax::TEn<4, 3>>(x0);

// Build a vector-valued function F : R^3 -> R^2
Eigen::Vector2<tax::TEn<4, 3>> F;
F(0) = sin(x) * y + z;
F(1) = exp(x - y) * cos(z);

// Extract values and derivatives at the expansion point
Eigen::Vector2d vals = tax::value(F);          // F(x0)
Eigen::Matrix<double, 2, 3> J = tax::jacobian(F);  // 2x3 Jacobian

// Evaluate at a displaced point
Eigen::Vector3d dx{0.01, -0.02, 0.03};
Eigen::Vector2d result = tax::eval(F, dx);     // F(x0 + dx)
```

## Module Pages

- [Mathematical Foundations](math.md) -- theory behind vector Taylor maps,
  gradients, Jacobians, Hessians, higher-order tensors, and map inversion.
- [API Reference](api.md) -- complete function signatures and parameter descriptions.
- [Examples](examples.md) -- practical, self-contained code examples.
