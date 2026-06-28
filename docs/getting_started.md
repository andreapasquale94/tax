# Getting Started

Install `tax`, build the tests, and write your first Taylor expansion.

---

## Requirements

| Tool | Minimum version |
|---|---|
| C++ compiler | GCC 13, Clang 17, Apple Clang 16 (C++23 with concepts) |
| CMake | 3.28 |
| Eigen | 3.4 |
| GoogleTest | fetched automatically by CMake if missing |

## Build & test from source

```bash
git clone https://github.com/andreapasquale94/tax.git
cd tax
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### CMake options

| Option | Default | Description |
|---|:-:|---|
| `TAX_BUILD_UNITTESTS` | `ON`  | Build the GoogleTest unit-test suite |
| `TAX_BUILD_REGRESSIONS` | `OFF` | Build the DACE-based regression tests |

The fast Cauchy kernel paths (`TAX_USE_UNROLL` for $M=1$, `TAX_USE_STENCIL`
for $M \ge 2$) default to on in `<tax/expansion/detail/stencil_config.hpp>`; pre-define the
macro to `0` (identically in every translation unit) to fall back to the
loop kernel.

### Install and consume from another CMake project

```bash
cmake --install build --prefix /your/install/prefix
```

```cmake
find_package(tax CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tax::tax)
```

If installed to a non-standard prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/your/install/prefix
```

---

## Your first expansion

Everything is reachable through a single umbrella header:

```cpp
#include <tax/tax.hpp>          // core type + named expansions + Eigen integration
#include <iostream>

int main() {
    auto x = tax::TE<5>::variable(0.0);   // 5th-order variable at x₀ = 0
    tax::TE<5> f = sin(x);                // propagate the whole Taylor series

    std::cout << f.value()         << '\n';   // sin(0)  = 0
    std::cout << f.derivative<1>() << '\n';   // cos(0)  = 1
    std::cout << f.derivative<3>() << '\n';   // −cos(0) = −1
    std::cout << f.eval(0.3)       << '\n';   // ≈ sin(0.3)
}
```

One evaluation pass yields the value **and** every derivative up to order 5.

---

## Next steps

- [Guide / Variables & Expressions](guide/expressions.md) — create variables and build expressions.
- [Guide / Extracting Results](guide/results.md) — coefficients, derivatives, evaluation.
- [Concepts / Foundations & Ordering](concepts/foundations.md) — what a truncated Taylor expansion is.
- [Reference / TaylorExpansion API](reference/core.md) — every public method, listed.
