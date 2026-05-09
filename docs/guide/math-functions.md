# Math functions

All math functions live in `namespace tax` as templates constrained on
`TaxExpression`. They return buffered ET nodes that fill their
coefficient buffer degree-by-degree.

## Catalogue

| Function | Recurrence basis | ET node |
|----------|------------------|---------|
| `tax::exp(x)`   | `dE/dx = E · du/dx`                      | `ExpExpr<E>` |
| `tax::log(x)`   | `u · d(log u)/dx = du/dx`                | `LogExpr<E>` |
| `tax::sqrt(x)`  | `out² = u`                                | `SqrtExpr<E>` |
| `tax::square(x)`| Cauchy product of x with itself          | `SquareExpr<E>` |
| `tax::cube(x)`  | `square(x) * x` (composed, no new node)  | `MulExpr<SquareExpr<E>, E>` |
| `tax::sin(x)`   | sin/cos paired Euler-operator recurrence | `SinExpr<E>` (= `SinCosExpr<E, true>`) |
| `tax::cos(x)`   | sin/cos paired                            | `CosExpr<E>` (= `SinCosExpr<E, false>`) |
| `tax::tan(x)`   | `sin / cos` (composed)                    | `DivExpr<SinExpr<E>, CosExpr<E>>` |
| `tax::sinh(x)`  | sinh/cosh paired                          | `SinhExpr<E>` |
| `tax::cosh(x)`  | sinh/cosh paired                          | `CoshExpr<E>` |
| `tax::tanh(x)`  | `sinh / cosh`                             | `DivExpr<SinhExpr<E>, CoshExpr<E>>` |

Each call returns a fresh node; the underlying `coeffs_` buffer is
sized like the operand's storage. View-like operators feeding into a
math function are advanced through their own `slice(d)`s.

## Examples

### Exponential and logarithm

```cpp
auto x = tax::TE<4>::variable(0.0);
tax::TE<4> r;
r <<= tax::exp(x);   // 1 + x + x²/2 + x³/6 + x⁴/24

auto y = tax::TE<4>::variable(1.0);
tax::TE<4> s;
s <<= tax::log(y);   // 0 + (y-1) - (y-1)²/2 + (y-1)³/3 - …
                      // ≈ x - x²/2 + x³/3 - x⁴/4 in dx around y=1
```

`log` requires `value() > 0`. `sqrt` requires `value() > 0`. `exp` is
unconditional.

### Trigonometry

```cpp
auto x = tax::TE<5>::variable(0.7);
tax::TE<5> identity;
identity <<= tax::square(tax::sin(x)) + tax::square(tax::cos(x));
// identity.value() ≈ 1.0 to floating-point precision; all higher
// coefficients ≈ 0.
```

`sin` and `cos` use a paired recurrence — `SinExpr` keeps a cos buffer
internally and vice versa, since each one's recurrence reads the
other's lower-degree coefficients. Calling both from a single
expression instantiates two independent paired evaluators (the
optimisation of sharing the pair across `sin(x)` and `cos(x)` calls
is a future-work item, not currently implemented).

### Square, cube, integer powers

```cpp
auto x = tax::TE<5>::variable(0.5);

tax::TE<5> a, b, c;
a <<= tax::square(x);   // x² via self-Cauchy
b <<= tax::cube(x);     // square(x) * x
c <<= x * x * x;        // identical result; one extra MulExpr in the tree
```

`tax::square(x)` is preferred over `x * x` only when you want the
buffer-sharing optimisation in `SquareExpr`; numerically the two are
equivalent.

## Composing with arithmetic

ET nodes compose freely. The following all work:

```cpp
result <<= tax::sin(u + v) * tax::exp(-u);
result <<= tax::sqrt(1.0 + tax::square(u));
result <<= tax::log(tax::cos(u) + 2.0);
```

Each buffered node along the chain keeps its own `coeffs_`; view-like
nodes between them allocate nothing.
