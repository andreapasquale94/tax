# tax

[![Tests](https://github.com/andreapasquale94/tax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax/actions/workflows/tests.yml)
[![Sanitizers](https://github.com/andreapasquale94/tax/actions/workflows/sanitizers.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax/actions/workflows/sanitizers.yml)
[![codecov](https://codecov.io/gh/andreapasquale94/tax/graph/badge.svg?branch=main)](https://codecov.io/gh/andreapasquale94/tax)

`tax` is a header-only C++23 Differential Algebra (DA) library for truncated univariate and multivariate Taylor expansions.

## Highlights

- Compile-time fixed order/variable count (`TDA<T, N, M>`)
- Lazy expression templates for arithmetic and math composition
- Direct access to coefficients and partial derivatives at expansion points

## Requirements

- C++23 compiler
- CMake 4.2+

## Build And Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /your/install/prefix
```

## Use From Another CMake Project

```cmake
find_package(tax CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tax::tax)
```

If installed to a non-standard prefix, pass it via `CMAKE_PREFIX_PATH`:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/your/install/prefix
```

## API Overview

Include everything:

```cpp
#include <tax/tax.hpp>
```

Core types:

- `tax::TDA<T, N, M>`: materialized DA polynomial (`N` max total order, `M` variables)
- `tax::DA<N>`: alias for `tax::TDA<double, N, 1>`
- `tax::DAn<N, M>`: alias for `tax::TDA<double, N, M>`

Factories and accessors:

- `DA<N>::variable(x0)` for univariate variables
- `DAn<N, M>::variable<I>(x0)` and `DAn<N, M>::variables(x0)` for multivariate variables
- `constant(v)`, `value()`, `coeff(alpha)`, `derivative(alpha)`

Supported operations include:

- Arithmetic: `+`, `-`, `*`, `/` between DA expressions and scalars
- Unary math: `abs`, `square`, `cube`, `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `log`, `log10`, `exp`
- Binary math: `pow`, `atan2`, `hypot`

## Quick Start (Univariate)

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    using tax::DA;

    auto x = DA<5>::variable(1.0);              // expansion point x0 = 1
    DA<5> f = tax::sin(x) + tax::square(x) / 2; // lazy expression, one materialization

    std::cout << "f(x0)  = " << f.value() << "\n";
    std::cout << "f'(x0) = " << f.derivative({1}) << "\n";
}
```

## Quick Start (Multivariate)

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    using tax::DAn;

    auto [x, y] = DAn<3, 2>::variables({1.0, 2.0}); // expansion point (1, 2)
    DAn<3, 2> f = tax::sin(x + y);

    std::cout << "coeff(dx dy) = " << f.coeff({1, 1}) << "\n";
    std::cout << "d2f/dxdy     = " << f.derivative({1, 1}) << "\n";
}
```
