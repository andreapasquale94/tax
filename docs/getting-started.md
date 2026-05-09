# Getting started

This page walks through installing tax, integrating it into a CMake
project, and running the brief's worked example.

## Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| C++ compiler | GCC 13, Clang 17 | C++23 is required |
| CMake | 3.20 | header-only consumer; no toolchain pinning |
| Eigen | 3.4 | provides `Eigen::Matrix`, `Eigen::VectorX` |

## Build & test

```sh
git clone https://github.com/andreapasquale94/tax
cd tax

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The whole library is header-only; the build only compiles the test
executables and (optionally) the Python module.

### CMake options

| Option | Default | Effect |
|--------|---------|--------|
| `TAX_BUILD_TEST`   | `ON`  | Build the GoogleTest suite. |
| `TAX_BUILD_PYTHON` | `OFF` | Build the nanobind Python bindings (requires `pip install nanobind`). |

### With Python bindings

```sh
pip install nanobind pytest
cmake -S . -B build -DTAX_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The bindings produce a `tax` package staged in
`build/python/tax/`. Use it directly with
`PYTHONPATH=build/python python -c "import tax"`.

## Consuming tax in your project

```cmake
find_package(Eigen3 3.4 REQUIRED CONFIG)
add_subdirectory(third_party/tax)   # or use find_package(tax) once installed

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE tax::tax)
```

`tax::tax` is an `INTERFACE` library that propagates the include
directory and the C++23 requirement.

## Hello, Taylor

```cpp title="hello.cpp"
#include <array>
#include <iostream>
#include <tax/tax.hpp>

int main() {
    auto [u, v] = tax::TEn<3, 2>::variables(std::array{1.0, 2.0});

    auto f = u * tax::sin(v) + u * v;     // ET expression — no allocation

    tax::TEn<3, 2> result;
    result <<= f;                          // streaming sweep fills `result`

    std::cout << "f       = " << result.value()           << "\n";
    std::cout << "df/du   = " << result.derivative<1, 0>() << "\n";
    std::cout << "df/dv   = " << result.derivative<0, 1>() << "\n";
    std::cout << "f(0.1)  = " << result.eval({0.1, 0.05}) << "\n";
}
```

```sh
g++ -std=c++23 -O2 -I include hello.cpp -o hello
./hello
```

Expected output (within float rounding):

```
f       = 2.9092974268256817
df/du   = 2.9092974268256817
df/dv   = 0.5838531634528576
f(0.1)  = 3.0145553...
```

## Next steps

- [**Concepts**](concepts/index.md) — the two architectural pillars
  (slice-streamed ETs, dual sizing).
- [**Guide**](guide/index.md) — using arithmetic, math functions,
  multivariate variables, derivatives.
- [**API reference**](api.md) — every public symbol.
- [**Python**](python.md) — the `tax` Python package.
