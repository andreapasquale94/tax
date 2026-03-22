# Examples

This page demonstrates practical usage of the Vector (Eigen) module through
complete, self-contained examples.

---

## Creating Variables from Eigen Vectors

Use `tax::variables` with structured bindings to create TTE variables from an
Eigen vector expansion point.

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main()
{
    Eigen::Vector3d x0{1.0, 2.0, 3.0};
    auto [x, y, z] = tax::variables<tax::TEn<3, 3>>(x0);

    // Each variable is a TEn<3,3> with the correct expansion point
    // x = 1.0 + dx,  y = 2.0 + dy,  z = 3.0 + dz
    std::cout << "x value: " << x.value() << "\n";  // 1.0
    std::cout << "y value: " << y.value() << "\n";  // 2.0
    std::cout << "z value: " << z.value() << "\n";  // 3.0
}
```

Alternatively, use `tax::vector` to get an Eigen vector of TTE variables:

```cpp
Eigen::Vector3d x0{1.0, 2.0, 3.0};
auto vars = tax::vector<tax::TEn<3, 3>>(x0);
// vars is Eigen::Matrix<TEn<3,3>, 3, 1>
```

---

## Gradient Computation

Build a scalar function and extract its gradient at the expansion point.

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main()
{
    Eigen::Vector2d x0{1.0, 2.0};
    auto [x, y] = tax::variables<tax::TEn<2, 2>>(x0);

    // f(x, y) = x^2 + x*y + y^2
    tax::TEn<2, 2> f = x * x + x * y + y * y;

    auto grad = tax::gradient(f);

    // grad(0) = df/dx = 2x + y = 2(1) + 2 = 4
    // grad(1) = df/dy = x + 2y = 1 + 2(2) = 5
    std::cout << "f(x0) = " << f.value() << "\n";       // 7.0
    std::cout << "grad = [" << grad(0) << ", "
              << grad(1) << "]\n";                        // [4, 5]
}
```

---

## Jacobian of a Vector Field

Build a 3-component vector function of 2 variables and compute the
3x2 Jacobian matrix.

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main()
{
    Eigen::Vector2d x0{1.0, 2.0};
    auto [x, y] = tax::variables<tax::TEn<2, 2>>(x0);

    // F: R^2 -> R^3
    Eigen::Matrix<tax::TEn<2, 2>, 3, 1> F;
    F(0) = x * y;           // F_0 = x*y
    F(1) = sin(x);          // F_1 = sin(x)
    F(2) = x + y * y;       // F_2 = x + y^2

    auto J = tax::jacobian(F);

    // J is 3x2:
    //   dF0/dx = y = 2       dF0/dy = x = 1
    //   dF1/dx = cos(1)      dF1/dy = 0
    //   dF2/dx = 1           dF2/dy = 2y = 4
    std::cout << "Jacobian:\n" << J << "\n";
    // Expected:
    //   2        1
    //   0.5403   0
    //   1        4
}
```

---

## Hessian Matrix

Use `tax::derivative<2>(f)` to extract the Hessian matrix of a scalar
function.

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main()
{
    Eigen::Vector2d x0{1.0, 2.0};
    auto [x, y] = tax::variables<tax::TEn<2, 2>>(x0);

    // f(x, y) = x^2 + x*y + y^2
    tax::TEn<2, 2> f = x * x + x * y + y * y;

    auto H = tax::derivative<2>(f);

    // H is 2x2:
    //   d2f/dx2   = 2    d2f/dxdy = 1
    //   d2f/dydx  = 1    d2f/dy2  = 2
    std::cout << "Hessian:\n" << H << "\n";
    // Expected:
    //   2  1
    //   1  2
}
```

---

## Third-Order Derivative Tensor

Use `tax::derivative<3>(f)` to extract the rank-3 derivative tensor.

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main()
{
    Eigen::Vector2d x0{0.0, 0.0};
    auto [x, y] = tax::variables<tax::TEn<3, 2>>(x0);

    // f(x, y) = x^3 + x^2*y + x*y^2 + y^3
    tax::TEn<3, 2> f = x * x * x + x * x * y + x * y * y + y * y * y;

    auto D3 = tax::derivative<3>(f);

    // D3 is an Eigen::Tensor<double, 3, RowMajor> of shape (2, 2, 2)
    //
    // D3(0,0,0) = d3f/dx3       = 6
    // D3(0,0,1) = d3f/dx2 dy    = 2
    // D3(0,1,1) = d3f/dx dy2    = 2
    // D3(1,1,1) = d3f/dy3       = 6
    std::cout << "D3(0,0,0) = " << D3(0, 0, 0) << "\n";  // 6.0
    std::cout << "D3(0,0,1) = " << D3(0, 0, 1) << "\n";  // 2.0
    std::cout << "D3(0,1,1) = " << D3(0, 1, 1) << "\n";  // 2.0
    std::cout << "D3(1,1,1) = " << D3(1, 1, 1) << "\n";  // 6.0
}
```

---

## Evaluating a Taylor Map

Build a vector of TTE and evaluate the polynomial map at a displacement from
the expansion point.

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main()
{
    Eigen::Vector2d x0{0.0, 0.0};
    auto [x, y] = tax::variables<tax::TEn<4, 2>>(x0);

    // Build a Taylor map F: R^2 -> R^2
    Eigen::Matrix<tax::TEn<4, 2>, 2, 1> F;
    F(0) = sin(x) + cos(y);
    F(1) = exp(x + y);

    // Evaluate at displacement dx = (0.1, 0.2)
    Eigen::Vector2d dx{0.1, 0.2};
    auto result = tax::eval(F, dx);

    // result(0) ≈ sin(0.1) + cos(0.2) ≈ 0.0998 + 0.9801 = 1.0799
    // result(1) ≈ exp(0.3) ≈ 1.3499
    std::cout << "F(x0 + dx) = [" << result(0) << ", "
              << result(1) << "]\n";

    // Compare with exact values
    std::cout << "Exact:       [" << std::sin(0.1) + std::cos(0.2)
              << ", " << std::exp(0.3) << "]\n";
}
```

For univariate TTE vectors, evaluation is automatically vectorized using a
matrix-vector product:

```cpp
    auto t = tax::TE<10>::variable(0.0);
    Eigen::Matrix<tax::TE<10>, 3, 1> g;
    g(0) = sin(t);
    g(1) = cos(t);
    g(2) = exp(t);

    // This uses the fast matrix-vector product path
    auto vals = tax::eval(g, 0.5);
    // vals ≈ [sin(0.5), cos(0.5), exp(0.5)]
```

---

## Inverting a Taylor Map

Build a nonlinear map, invert it, and verify the round-trip property
\(\mathbf{F}(\mathbf{G}(\mathbf{u})) \approx \mathbf{u}\).

```cpp
#include <tax/tax.hpp>
#include <iostream>
#include <cmath>

int main()
{
    constexpr int N = 15;

    Eigen::Vector2d x0{0.0, 0.0};
    auto [x, y] = tax::variables<tax::TEn<N, 2>>(x0);

    // Build a nonlinear map F: R^2 -> R^2
    Eigen::Matrix<tax::TEn<N, 2>, 2, 1> F;
    F(0) = x + y + x * y;       // Linear part: x + y, nonlinear: x*y
    F(1) = y + x * x;           // Linear part: y,     nonlinear: x^2

    // Invert the map
    auto G = tax::invert(F);

    // Verify round-trip: F(G(u)) should equal u
    Eigen::Vector2d u{0.01, -0.01};

    // First apply G, then F
    auto intermediate = tax::eval(G, u);
    auto round_trip   = tax::eval(F, intermediate);

    std::cout << "u           = [" << u(0) << ", " << u(1) << "]\n";
    std::cout << "F(G(u))     = [" << round_trip(0) << ", "
              << round_trip(1) << "]\n";
    std::cout << "Round-trip error: "
              << std::abs(round_trip(0) - u(0)) + std::abs(round_trip(1) - u(1))
              << "\n";
    // Expected: round-trip error < 1e-13
}
```

The inversion also works for univariate maps. For example, inverting
\(\sin(x)\) recovers \(\arcsin(x)\):

```cpp
    constexpr int N = 9;
    auto x = tax::TE<N>::variable(0.0);

    Eigen::Matrix<tax::TE<N>, 1, 1> sinMap;
    sinMap(0) = sin(x);

    auto asinMap = tax::invert(sinMap);
    // asinMap(0) matches the Taylor series of arcsin(x) to order 9
```
