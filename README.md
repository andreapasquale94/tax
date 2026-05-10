# tax

Header-only C++23 library for **truncated multivariate Taylor expansions**
(Differential Algebra).  Each value carries a polynomial in M variables
truncated at total degree N; arithmetic and math operations propagate the
full Taylor jet in a single evaluation pass.

```cpp
#include <tax/tax.hpp>

auto [u, v] = tax::TEn<3, 2>::variables(std::array{1.0, 2.0});

auto f = u * tax::sin(v) + u * v;     // lazy expression, no allocation yet

tax::TEn<3, 2> result;
result <<= f;                          // streaming degree-by-degree sweep

double f0    = result.value();         // function value at the centre
double dfdu  = result.derivative({1, 0});  // partial w.r.t. u
double fhat  = result.eval(std::array{0.1, 0.05});  // Taylor polynomial eval
```

## Building

Requires CMake ≥ 3.20, a C++23 compiler, and Eigen ≥ 3.4.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Architecture in one paragraph

A single storage template — `TaylorExpansionT<T, int Order, int Vars>`,
modelled on `Eigen::Matrix<T, Rows, Cols>` with `Eigen::Dynamic` (= -1)
as the runtime-size sentinel — covers both compile-time-fixed sizes
(aliases `TE<N>` / `TEn<N, M>`) and runtime-fixed sizes (alias
`DynTE<T>`). Both configurations satisfy the same `tax::TaxExpression`
concept and feed the same slice-aware coefficient kernels (Cauchy
convolution, exp, log, sin/cos pair, sinh/cosh pair, sqrt, ...).  Expression templates evaluate **degree-by-degree**: view-like
nodes (`AddExpr`, `NegExpr`, `ScalarMulExpr`, ...) allocate nothing and
return lazy slice views; buffered nodes (`MulExpr`, `DivExpr`, all
transcendentals) own a coefficient buffer and fill slices monotonically.
The `<<=` driver writes directly into the destination so the top of the
tree allocates nothing either.  Mixed static/dynamic expressions are
rejected at compile time.

See [`CLAUDE.md`](CLAUDE.md) for the complete architecture and
contributor guide.

## License

BSD 3-Clause — see [`LICENSE`](LICENSE).
