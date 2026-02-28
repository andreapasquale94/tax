# tax

`tax` is a header-only C++23 Differential Algebra (DA) library for truncated multivariate Taylor expansions.

It provides:
- Materialized DA objects (`tax::TDA<T, N, M>`)
- Expression templates for zero-temporary composition
- Arithmetic on DA expressions

## Requirements

- C++23 compiler
- CMake 4.2+

## Install / Include

This project currently builds as an interface library. Add it with CMake and include the umbrella header:

```cpp
#include <tax/tax.hpp>
```

You can also include narrower headers (`<tax/da.hpp>`, `<tax/operators.hpp>`, etc.) if preferred.

## Quick Start (Univariate)

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    using tax::DA;

    // 5th-order DA variable expanded around x0 = 1.0
    auto x = DA<5>::variable(1.0);

    // Build expression lazily, materialize once
    DA<5> f = tax::sin(x) + tax::square(x) / 2.0;

    std::cout << "f(x0) = " << f.value() << "\n";
    std::cout << "f'(x0) = " << f.derivative({1}) << "\n";
}
```

## Multivariate Example

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    using tax::DAn;

    // 3rd-order in 2 variables, expansion point (x0, y0) = (1, 2)
    auto [x, y] = DAn<3, 2>::variables({1.0, 2.0});

    DAn<3, 2> f = tax::sin(x + y);

    // Coefficient of dx^1 dy^1 term
    std::cout << f.coeff({1, 1}) << "\n";

    // Mixed partial d^2f/(dx dy) at expansion point
    std::cout << f.derivative({1, 1}) << "\n";
}
```

## Build and Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Notes

- This is an in-progress library (`0.1.0`) and installation/export rules are not finalized yet.
