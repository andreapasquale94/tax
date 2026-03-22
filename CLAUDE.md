# CLAUDE.md — AI Assistant Guide for `tax`

## Project Overview

**tax** is a header-only C++23 library for **Truncated Algebraic eXpansions (TAX)** — truncated multivariate Taylor polynomials that propagate complete Taylor series through arbitrary expressions. In a single evaluation pass, it yields function values and all partial derivatives up to order N.

- **Version:** 0.1.0
- **License:** BSD 3-Clause
- **C++ Standard:** C++23 (required)
- **Build system:** CMake

---

## Repository Structure

```
tax/
├── include/tax/          # Header-only library (the entire library lives here)
│   ├── tax.hpp           # Umbrella header — users include only this
│   ├── tte.hpp           # Core TruncatedTaylorExpansionT<T,N,M> class
│   ├── ads.hpp           # Facade: includes all ads/
│   ├── kernels.hpp       # Facade: includes all kernels/
│   ├── operators.hpp     # Facade: includes all operators/
│   ├── utils.hpp         # Facade: includes all utils/
│   ├── ads/              # Automatic Domain Splitting (ADS)
│   ├── expr/             # Expression template nodes (lazy evaluation)
│   ├── kernels/          # Series computation kernels (recurrence relations)
│   ├── ode/              # Taylor ODE integrator
│   ├── operators/        # Free-function operators and math functions
│   ├── utils/            # Type traits, combinatorics, enumeration
│   └── eigen/            # Eigen3 integration helpers
├── tests/                # Google Test suite (29 test executables, 440 tests)
│   ├── ads/              # ADS tree and runner tests
│   ├── core/             # Basic TTE construction, nesting, composition, deriv/integ
│   ├── expr/             # Expression template correctness
│   ├── kernels/          # Kernel algorithm verification
│   ├── foundation/       # Combinatorics and enumeration utilities
│   ├── eigen/            # Eigen integration tests
│   ├── ode/              # Taylor integrator and ADS-integrated ODE tests
│   ├── dace/             # Optional DACE comparative tests
│   ├── testUtils.hpp     # Shared test helpers and macros
│   └── CMakeLists.txt
├── benchmarks/           # Google Benchmark suite
├── docs/                 # Markdown documentation
├── cmake/                # CMake package config template
├── tools/                # install_eigen.sh helper script
├── .github/workflows/    # CI: tests.yml, sanitizers.yml
├── .clang-format         # Code style configuration
├── CMakeLists.txt        # Root CMake configuration
└── README.md
```

---

## Building

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Test
ctest --test-dir build --output-on-failure
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `TAX_BUILD_TEST` | `ON` | Build Google Test suite |
| `TAX_BUILD_BENCHMARK` | `OFF` | Build Google Benchmark suite |
| `TAX_USE_DACE` | `OFF` | Enable DACE comparison (fetched automatically) |

### With Benchmarks

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DTAX_BUILD_BENCHMARK=ON -DTAX_USE_DACE=ON
cmake --build build-bench --target bench_univariate -j
./build-bench/benchmarks/bench_univariate
```

### With Coverage

```bash
cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS=--coverage
cmake --build build-cov
ctest --test-dir build-cov --output-on-failure
gcovr --root . build-cov --filter "^include/tax/"
```

### Dependencies

- **Required:** Eigen3 ≥ 3.4 (must be installed or pointed to via `CMAKE_PREFIX_PATH`)
- **Optional:** DACE v2.1.0 — fetched automatically via CMake FetchContent if `TAX_USE_DACE=ON`
- **Test framework:** Google Test v1.17 — fetched automatically if not found
- **Benchmark framework:** Google Benchmark v1.9 — fetched automatically if not found

---

## Core Concepts

### The Main Type

```cpp
tax::TruncatedTaylorExpansionT<T, N, M>
// T = scalar type (double, float)
// N = truncation order
// M = number of variables
```

Convenient aliases:
```cpp
tax::TE<N>        // univariate: TruncatedTaylorExpansionT<double, N, 1>
tax::TEn<N, M>    // multivariate: TruncatedTaylorExpansionT<double, N, M>
```

### Creating Variables

```cpp
// Univariate
auto x = tax::TE<3>::variable(x0);       // x = x0 + 1*dx

// Multivariate (structured bindings)
auto [x, y] = tax::TEn<3, 2>::variables(x0, y0);
```

### Using the Library

```cpp
#include <tax/tax.hpp>

auto x = tax::TE<5>::variable(1.0);
auto f = sin(x) * exp(x);

double val  = f.value();           // constant term
double df   = f.derivative(1);     // first derivative
double d2f  = f.derivative(2);     // second derivative
auto   p    = f.eval(dx);          // evaluate Taylor polynomial at x0+dx
```

### Differentiation and Integration

TTE objects support symbolic partial differentiation and integration:

```cpp
auto x = tax::TE<5>::variable(1.0);
auto f = sin(x);

// Compile-time variable index
auto df = f.deriv<0>();       // d/dx sin(x) = cos(x)
auto F  = f.integ<0>();       // integral of sin(x) dx

// Runtime variable index
auto df2 = f.deriv(0);       // equivalent to deriv<0>()
auto F2  = f.integ(0);       // equivalent to integ<0>()

// Multivariate
auto [x, y] = tax::TEn<3, 2>::variables(1.0, 2.0);
auto g = x * x * y;
auto dg_dx = g.deriv<0>();   // partial derivative w.r.t. x
auto dg_dy = g.deriv<1>();   // partial derivative w.r.t. y
auto Gx    = g.integ<0>();   // integral w.r.t. x
```

### Coefficient Storage

- Coefficients stored in `std::array<T, nCoefficients>` (stack-allocated, no heap)
- Graded-lexicographic ordering: all degree-0 first, then degree-1, etc.
- Size: `nCoefficients = C(N+M, M)` (binomial coefficient)
- `coeff(k)` retrieves raw Taylor coefficient; `derivative(k)` applies `k!` scaling

---

## Architecture: Expression Templates

The library uses lazy evaluation via expression templates to avoid materializing intermediate TTE objects:

```
User writes:  sin(x * y + z)
Builds tree:  UnaryExpr<sin, ProductExpr<x, y>, z>
Evaluated:    Only when assigned to TruncatedTaylorExpansionT
```

Key design choices:
- **Sum flattening:** `a + b + c + d` → single `SumExpr<a,b,c,d>` (one pass)
- **Product flattening:** `a * b * c` → single `ProductExpr<a,b,c>` (rolling Cauchy product)
- **Leaf fast-paths:** Binary ops on materialized TTE objects take shortcuts
- **CRTP base:** `Expr<Derived, T, N, M>` unifies all nodes

### Key Files in `expr/`

| File | Purpose |
|------|---------|
| `base.hpp` | CRTP `Expr<>` base with evaluation interface |
| `arithmetic_ops.hpp` | Op tags: `OpAdd`, `OpSub`, `OpMul`, `OpDiv` |
| `bin_expr.hpp` | Binary expression nodes |
| `sum_expr.hpp` | Flattened N-ary sum |
| `product_expr.hpp` | Flattened N-ary product |
| `unary_expr.hpp` | Unary function applications |
| `func_expr.hpp` | Generic function call nodes |
| `math_ops.hpp` | High-level math wrappers |

---

## Kernels

All mathematical operations are implemented as degree-by-degree recurrence relations in `kernels/`:

| File | Operations |
|------|-----------|
| `algebra.hpp` | reciprocal, sqrt, cbrt, square, cube |
| `cauchy.hpp` | Cauchy product, self-product, accumulate |
| `trigonometric.hpp` | sin, cos, tan, asin, acos, atan |
| `transcendental.hpp` | exp, log, sinh, cosh, tanh, and inverses |
| `ops.hpp` | Utility helpers |

Kernel optimisations:
- **Symmetry exploitation:** `cauchySelfProduct` enumerates unordered pairs for ~2x fewer multiplications
- **Univariate fast-paths:** `if constexpr (M == 1)` branches avoid multi-index overhead
- **Incremental `sq` tracking in `seriesCbrt`:** Maintains `out^2` degree-by-degree for O(N^2) instead of O(N^3)

When adding a new mathematical function, implement the recurrence relation in the appropriate kernel file, then expose it via `operators/math_unary.hpp` or `operators/math_binary.hpp`.

---

## Eigen Integration

Located in `include/tax/eigen/`. Enables using TTE types inside Eigen matrices/vectors.

```cpp
#include <tax/tax.hpp>

Eigen::Vector2d x0 = {1.0, 2.0};
auto [x, y] = tax::variables<tax::TEn<3,2>>(x0);

Eigen::Vector2<tax::TEn<3,2>> f = {sin(x), cos(y)};

auto vals = tax::value(f);          // Eigen::Vector2d of constant terms
auto grad = tax::gradient(f[0], 2); // Gradient of f[0] w.r.t. 2 variables
auto J    = tax::jacobian(f, 2);    // 2x2 Jacobian matrix
```

Key helpers in `eigen/`:
- `tax::vector<TTE>(x0)` — element-wise Eigen vector → TTE vector
- `tax::variables<TTE>(x0)` — structured-binding access
- `tax::value(container)` — extract constant terms
- `tax::eval(container, dx)` — evaluate at displacement
- `tax::gradient(f, M)` — gradient vector
- `tax::jacobian(F, M)` — Jacobian matrix

---

## Taylor ODE Integrator

Located in `include/tax/ode/`. Provides adaptive Taylor-method integration for scalar and vector ODEs with optional Automatic Domain Splitting (ADS).

### Scalar ODE

```cpp
#include <tax/tax.hpp>

// dx/dt = x  →  x(t) = exp(t)
auto f = [](const auto& x, [[maybe_unused]] const auto& t) { return x; };

auto sol = tax::ode::integrate<25>(f, 1.0, 0.0, 2.0, 1e-16);
// sol.t  — step times
// sol.x  — state at each step time
// sol(t) — dense output at any time via Taylor polynomial evaluation
```

### Vector ODE

```cpp
// Harmonic oscillator: dx/dt = v, dv/dt = -x
auto f = [](auto& dx, const auto& x, [[maybe_unused]] const auto& t) {
    dx(0) = x(1);
    dx(1) = -x(0);
};

Eigen::Vector2d x0{1.0, 0.0};
auto sol = tax::ode::integrate<25>(f, x0, 0.0, 2*M_PI, 1e-16);
```

### ADS-Integrated ODE Propagation

Propagate a set of initial conditions with automatic domain splitting:

```cpp
// Propagate an initial-condition box with ADS
tax::Box<double, 2> ic_box{
    .center = {1.0, 0.0},
    .halfWidth = {0.1, 0.1}
};

auto tree = tax::ode::integrateAds<25, 6>(f, ic_box, 0.0, 10.0,
    1e-16,   // step tolerance
    1e-3,    // ADS splitting tolerance
    30,      // max split depth
    500      // max steps per subdomain
);

// Iterate results
for (int i : tree.doneLeaves()) {
    const auto& leaf = tree.node(i).leaf();
    // leaf.tte.state — DA polynomial flow map
    // leaf.box       — subdomain of initial conditions
}
```

### Key Files in `ode/`

| File | Purpose |
|------|---------|
| `taylor_integrator.hpp` | Umbrella header |
| `step.hpp` | Single Taylor step (scalar + vector) |
| `stepsize.hpp` | Jorba-Zou adaptive step-size control |
| `integrate.hpp` | Full integration loop with dense output |
| `solution.hpp` | `TaylorSolution` container with `operator()` dense output |
| `integrate_ads.hpp` | ADS-integrated ODE propagation (`integrateAds`, `propagateBox`, `stepDa`, `makeDaState`) |

---

## Automatic Domain Splitting (ADS)

Located in `include/tax/ads/`. Implements the algorithm from Wittig et al. (2015) for adaptive polynomial approximation over large domains.

### Standalone Function Approximation

```cpp
#include <tax/ads.hpp>

// Approximate f(x) = exp(-x^2) on [-3, 3]
auto f = [](const auto& x) { return exp(-x * x); };

tax::Box<double, 1> domain{.center = {0.0}, .halfWidth = {3.0}};
auto runner = tax::makeAdsRunner<10, 1>(f, 1e-5);
auto tree = runner.run(domain);

// Each done leaf contains a polynomial valid on its subdomain
for (int i : tree.doneLeaves()) {
    const auto& leaf = tree.node(i).leaf();
    // leaf.tte — polynomial approximation
    // leaf.box — subdomain
}

// Point lookup: O(depth) binary tree walk
int idx = tree.findLeaf({1.5});
```

### Key Files in `ads/`

| File | Purpose |
|------|---------|
| `box.hpp` | `Box<T,M>` axis-aligned hyperrectangle |
| `ads_node.hpp` | `AdsNode<TTE>` — leaf/internal variant node |
| `ads_tree.hpp` | `AdsTree<TTE>` — arena-based binary tree with work queue |
| `ads_runner.hpp` | `AdsRunner<N,M,F>` — the ADS algorithm driver |

### Architecture

- **Arena-based tree:** All nodes live in a contiguous `std::vector`, referenced by index (no pointer invalidation)
- **Work queue:** BFS processing via `std::deque`
- **O(1) leaf removal:** Swap-and-pop in the leaf list
- **Split-dimension selection:** Variable contributing most to degree-N truncation error

---

## Code Conventions

### Naming

| Category | Convention | Examples |
|----------|-----------|---------|
| Types/Classes | `PascalCase` | `TruncatedTaylorExpansionT`, `MultiIndex`, `AdsTree`, `AdsRunner`, `AdsNode`, `FlowMap`, `TaylorSolution` |
| Template params | `UPPERCASE` or short | `T`, `N`, `M`, `P`, `D`, `Derived` |
| Free functions & methods | `camelCase` | `variable()`, `flatIndex()`, `seriesReciprocal()`, `deriv()`, `integ()`, `findLeaf()`, `addLeaf()`, `markDone()`, `integrateAds()`, `makeAdsRunner()` |
| Local variables | `snake_case` | `n_coeff`, `dx`, `half_width` |
| Namespaces | `lowercase` | `tax`, `tax::detail`, `tax::ode` |
| Op tags | `PascalCase` with prefix | `OpAdd`, `OpSub`, `OpMul` |
| Type aliases | Short uppercase | `TE<N>`, `TEn<N,M>` |

### C++ Patterns

- **`constexpr` everywhere:** All size calculations, index mappings, and coefficient operations must be `constexpr`
- **`noexcept` on all operations:** For zero-overhead guarantees (exception: methods that `throw`, e.g. runtime-index `deriv(int)`)
- **No heap allocation in core library:** Use `std::array` for fixed-size storage; `std::vector` is acceptable only in ODE/ADS modules for variable-length solutions and dynamic tree growth
- **Concepts:** Use `tax::Scalar` concept (wraps `std::floating_point`) for scalar template parameters
- **`if constexpr`:** Used for compile-time branching between univariate (M=1) and multivariate cases
- **`[[nodiscard]]`:** Applied to accessor methods, computation results, and expensive operations
- **Internal details in `tax::detail`:** Do not expose implementation internals in `tax::`; ODE internals use `tax::ode::detail`

### Formatting

Enforced by `.clang-format` (Google style, customized):
- Indent: **4 spaces** (no tabs)
- Column limit: **100 characters**
- Brace wrapping: new line after class/struct/function/namespace/control statements
- Spaces inside parentheses and angle brackets

Run clang-format before committing:
```bash
clang-format -i include/tax/**/*.hpp
```

---

## Testing

### Structure

Tests are organized by feature, one `.cpp` per concern. Each produces a standalone test executable:

```
tests/core/        — TTE constructors, variable factories, composition, deriv/integ
tests/expr/        — One file per math function (sin, exp, log, pow, etc.)
tests/kernels/     — Direct kernel algorithm verification
tests/foundation/  — Combinatorics and enumeration
tests/eigen/       — Eigen integration
tests/ads/         — ADS tree structure and runner (Gaussian approximation)
tests/ode/         — Taylor integrator (scalar, vector, ADS-integrated)
tests/dace/        — Comparative tests against DACE (optional)
```

### Writing Tests

```cpp
#include <gtest/gtest.h>
#include <tax/tax.hpp>
#include "testUtils.hpp"   // ExpectCoeffsNear, kTol

TEST(ExprSin, UnivariateOrder3) {
    auto x = tax::TE<3>::variable(0.0);
    auto f = sin(x);
    // Expected coefficients for sin(x) at x=0: [0, 1, 0, -1/6]
    tax::TE<3> expected{0.0, 1.0, 0.0, -1.0/6.0};
    ExpectCoeffsNear<tax::TE<3>>(f, expected, kTol);
}
```

- Use `ExpectCoeffsNear<TTE_type>(actual, expected, tol)` for coefficient-wise comparison
- Default tolerance: `kTol = 1e-10`
- Each new math function or kernel needs a corresponding test in `tests/expr/` or `tests/kernels/`

### Running Tests

```bash
ctest --test-dir build --output-on-failure
# Or run a single test executable:
./build/tests/testExprTrig
```

---

## CI/CD

### Workflows

**`tests.yml`** — Triggered on push, pull_request, and manual dispatch:
- Matrix: Ubuntu + macOS × GCC + Clang × Eigen 3.4.0 + Eigen 5.0.0 (8 combinations)
- Build type: Release
- Includes DACE for comparison tests
- Coverage job: GCC/Ubuntu/Debug, reports to Codecov (filter: `^include/tax/`)

**`sanitizers.yml`** — Manual dispatch only:
- Precheck: same matrix as tests.yml
- Sanitizer jobs: ASAN, UBSAN, TSAN
- Build type: RelWithDebInfo with `-O1 -fno-omit-frame-pointer -g`

### Before Submitting a PR

1. All 29 test executables pass locally (440 individual tests)
2. Code is formatted with `clang-format`
3. No new dynamic allocations introduced in core library
4. New math operations have kernel tests AND expression tests
5. Eigen helpers have tests in `tests/eigen/`
6. ODE/ADS changes have tests in `tests/ode/` or `tests/ads/`

---

## Adding a New Mathematical Function

1. **Kernel:** Implement the degree-by-degree recurrence in the appropriate `kernels/` file
2. **Operator:** Expose via a free function in `operators/math_unary.hpp` (or `math_binary.hpp`)
3. **Expression node:** If the function has special structure, add a node in `expr/`; otherwise use `UnaryExpr`
4. **Tests:** Add `tests/expr/testExpr<FunctionName>.cpp` with univariate and multivariate cases
5. **CMakeLists:** Register the new test file in `tests/CMakeLists.txt`
6. **Docs:** Update `docs/math_operations.md` with the recurrence relation

---

## Common Pitfalls

- **Do not use `std::vector` or `new` in core library:** The core TTE type must remain allocation-free; `std::vector` is only acceptable in ODE/ADS modules
- **Do not break `constexpr`:** All index arithmetic must stay compile-time
- **graded-lex ordering is sacred:** The coefficient order (`flatIndex`) is used everywhere — never change it
- **M=0 is invalid:** Always assert or static_assert that M ≥ 1
- **Concepts vs. SFINAE:** Prefer C++20 concepts (`requires`, `Scalar` concept) over SFINAE
- **Include the umbrella header in tests:** Use `#include <tax/tax.hpp>`, not individual sub-headers
- **Expression templates store references:** When using ADS or ODE with expression-returning callables, arguments must be taken by `const&` — by-value copies dangle once the function returns its lazy expression

---

## Documentation

- `docs/getting_started.md` — Installation, basic usage, key concepts
- `docs/api_reference.md` — Complete API reference
- `docs/math_operations.md` — Mathematical recurrence relations for all operations
- `docs/eigen_integration.md` — Eigen helper reference
- `README.md` — Project overview with quick-start examples
- Doxygen: `doxygen Doxyfile` generates HTML docs from header comments
