# Arithmetic

All four binary operators (`+`, `-`, `*`, `/`), unary `-`, and scalar
combinations are overloaded for any `tax::TaxExpression`. Operators
return ET nodes; the result is materialised when assigned via `.eval()`.

## The big picture

```cpp
tax::TE<5> x = tax::TE<5>::variable(1.0);
tax::TE<5> y = tax::TE<5>::variable(2.0);

auto expr   = (x + y) * (x - y);   // ET tree, no allocation yet
auto result = expr.eval();         // streaming sweep yields a TE<5>
```

The temporaries created inside the operator chain — `AddExpr`,
`SubExpr`, `MulExpr` — survive only until the end of the
`.eval()` statement, which is when the streaming driver consumes them.

!!! warning "Don't outlive the full expression"
    `auto e = (x + y) * (x - y);` keeps `e` valid only within its
    initialisation statement. After the semicolon the inner `AddExpr`
    and `SubExpr` temporaries are destroyed, and `e` has dangling
    references. Always feed the chain straight into `.eval()` or a
    function that consumes it within the same full expression.

## Operators

| Form | ET node returned | Allocates? |
|------|------------------|------------|
| `a + b`  | `AddExpr<L, R>`        | No |
| `a - b`  | `SubExpr<L, R>`        | No |
| `-a`     | `NegExpr<E>`            | No |
| `a + c`  | `ScalarAddExpr<E>`      | No |
| `c + a`  | `ScalarAddExpr<E>`      | No |
| `a - c`  | `ScalarAddExpr<E>`      | No |
| `c - a`  | `ScalarAddExpr<NegExpr<E>>` | No |
| `c * a`  | `ScalarMulExpr<E>`      | No |
| `a * c`  | `ScalarMulExpr<E>`      | No |
| `a / c`  | `ScalarMulExpr<E>` (with `1/c`) | No |
| `a * b`  | `MulExpr<L, R>`         | Yes (Cauchy buffer) |
| `a / b`  | `DivExpr<L, R>`         | Yes (recurrence buffer) |

Where `a, b` are `TaxExpression`s and `c` is a `Scalar`. The first six
rows are view-like nodes — they hand back lazy slices computed on
demand and own no `coeffs_` buffer.

## Scalar handling

Scalar operands fold into `ScalarAddExpr` or `ScalarMulExpr`. These
shift only the `(d=0, i=0)` element (for add) or scale every
coefficient (for mul); they don't materialise a full constant TTE
operand.

```cpp
auto x = tax::TE<3>::variable(0.0);

tax::TE<3> a, b, c;
a = (x + 7.0).eval();            // shifts only the constant term
b = (7.0 - x).eval();            // (-x) + 7.0
c = (3.0 * x).eval();            // every coefficient * 3.0
```

## Mixed kinds are rejected

```cpp
auto sx = tax::TE<3>::variable(1.0);
auto dx = tax::DynTE<>::variable(1.0, 3, 1, 0);

auto bad = sx + dx;   // ❌ SameKindExpression<TE, DynTE> fails
```

The error points at the `requires` clause on the `+` operator. To
bridge the two paths, copy coefficients explicitly into the target
storage type.

## Static-size compatibility check

Within the static path, dimensions must match:

```cpp
auto a = tax::TEn<3, 2>::constant(1.0);
auto b = tax::TEn<3, 3>::constant(1.0);

auto bad = a + b;     // ❌ VarsAtCompileTime mismatch
```

The `.eval()` driver also asserts the destination matches the expression's
order and variable count when both are static.

## Putting it together

```cpp
#include <array>
#include <tax/tax.hpp>

int main() {
    auto [u, v] = tax::TEn<3, 2>::variables(std::array{1.0, 2.0});

    auto poly   = (u - v) * (u + v) - 2.0 * u + 1.0;
    auto result = poly.eval();

    // (u - v)(u + v) - 2u + 1 = u^2 - v^2 - 2u + 1
    // value at (1, 2): 1 - 4 - 2 + 1 = -4
    return static_cast<int>(result.value());
}
```
