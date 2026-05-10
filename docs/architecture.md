# Architecture

This page traces the flow of a single `.eval()` statement from operator
overload to streaming sweep, then catalogues the recurrences each
buffered node uses. Read [Concepts](concepts/index.md) first for the
high-level picture.

## End-to-end flow

```cpp
auto [u, v] = tax::TEn<3, 2>::variables(std::array{1.0, 2.0});
auto f = u * tax::sin(v) + u * v;
auto result = f.eval();
```

The single statement does the following:

1. **Operator chain builds an ET tree.** `tax::sin(v)` constructs a
   `SinExpr<TEn>` (a buffered node with sin/cos buffers). `u *
   tax::sin(v)` builds `MulExpr<TEn, SinExpr<TEn>>`; the inner
   `SinExpr` is captured **by value** (`etstore_t<E> = E` for
   inheritors of `ExprTag`). Storage operands like `u` and `v` are
   captured **by const reference**. The final `+` produces an
   `AddExpr<MulExpr, MulExpr>` that owns both child nodes.

2. **`f` is a temporary**, alive for the full expression
   `auto result = f.eval();`. The `Expr<Derived>::eval()` body —
   defined out-of-line in `storage/tte.hpp` once `TaylorExpansionT`
   is complete — constructs a fresh result of matching shape and
   drives the streaming sweep into it.

3. **Driver loop.** `eval()` walks `d = 0, 1, …, order` and for
   each `d`:
   - calls `f.advanceTo(d)`. The `AddExpr` forwards to its children;
     the `MulExpr`s call their inputs' `advanceTo(d)` and then run
     `kernels::cauchyMulComputeDegree<Scalar>(d, nvars(), lhs_,
     rhs_, out_e)`. The `SinExpr` runs the paired sin/cos
     recurrence.
   - reads `f.slice(d)`. For the top-level `AddExpr` this is a
     `ParentSliceView` that recursively pulls
     `lhs.slice(d).coeff(i) + rhs.slice(d).coeff(i)` on every
     element access.
   - copies element-wise into `result.slice(d)` (a real
     `Eigen::VectorBlock`).

4. **No temporary `Eigen::Vector` is ever allocated for any view-like
   step.** The two `MulExpr`s and the `SinExpr` each own one
   `coeffs_` (and `SinExpr` an extra `cos_` for the pair). That is
   the entire allocation footprint of the chain.

## How view-like nodes hand back slices

Each view-like ET (`AddExpr`, `SubExpr`, `NegExpr`, `ScalarMulExpr`,
`ScalarAddExpr`) returns a `ParentSliceView<Parent>` from `slice(d)`.
The view holds a const reference to the parent ET node — whose
lifetime is the surrounding full expression — and the degree being
viewed. On every `coeff(i)` it dispatches to
`parent.coeffAtSlice(d, i)`, which reads the underlying storage
directly. Element accesses traverse stable references all the way
down to the leaves; the view itself owns no buffer.

## Kernel signatures

The slice-aware Cauchy primitive:

```cpp
template <class T, class A, class B, class Out>
void cauchyAccumulateSlice(std::size_t eA, std::size_t eB,
                           std::size_t nvars,
                           T scale, const A& a, const B& b, Out& out);
```

`a` and `b` only need `coeff(i)`; `out` needs `coeffRef(i)`.
`Eigen::VectorBlock` satisfies both; `ParentSliceView` satisfies
input-side only — fine for kernel inputs.

## Recurrences (degree-d slice from lower slices)

Let `f`, `u` denote TTEs and `f_d` the degree-d coefficient slice.

**Multiplication.** Cauchy product:

$$
f_\alpha = \sum_{\beta + \gamma = \alpha} a_\beta\, b_\gamma
\quad\text{equivalently}\quad
f_d = \sum_{e=0}^{d} a_e * b_{d-e}.
$$

**Division** from `b · f = a`:

$$
f_d = \frac{1}{b_0}\Bigl(a_d - \sum_{e=1}^{d} b_e * f_{d-e}\Bigr).
$$

**Reciprocal** from `b · f = 1`:

$$
f_0 = 1/b_0,\qquad
f_d = -\frac{1}{b_0}\sum_{e=1}^{d} b_e * f_{d-e}\quad (d \ge 1).
$$

**Square root** from `f² = u`:

$$
f_0 = \sqrt{u_0},\qquad
f_d = \frac{1}{2 f_0}\Bigl(u_d - \sum_{e=1}^{d-1} f_e * f_{d-e}\Bigr).
$$

**Exponential** (Euler operator: `E[F] = F · E[u]`):

$$
f_0 = \exp(u_0),\qquad
f_\alpha = \frac{1}{|\alpha|}
  \sum_{\substack{\gamma \le \alpha \\ |\gamma| \ge 1}}
   |\gamma|\, u_\gamma\, f_{\alpha - \gamma}.
$$

In slice form for `d \ge 1`:

$$
f_d = \frac{1}{d}\sum_{e=1}^{d} e\, (u_e * f_{d-e}).
$$

**Logarithm** from `u · E[f] = E[u]`:

$$
f_0 = \log(u_0),\qquad
f_\alpha = \frac{u_\alpha}{u_0}
  - \frac{1}{|\alpha|\, u_0}\!\!
   \sum_{\substack{\beta \le \alpha\\ 1 \le |\beta| < |\alpha|}}
   (|\alpha| - |\beta|)\, u_\beta\, f_{\alpha - \beta}.
$$

**sin / cos** (paired):

$$
\sin_d = \frac{1}{d}\sum_{e=1}^{d} e\,(u_e * \cos_{d-e}),\quad
\cos_d = -\frac{1}{d}\sum_{e=1}^{d} e\,(u_e * \sin_{d-e}).
$$

**sinh / cosh** (paired, no sign flip on the second equation):

$$
\sinh_d = \frac{1}{d}\sum_{e=1}^{d} e\,(u_e * \cosh_{d-e}),\quad
\cosh_d = \frac{1}{d}\sum_{e=1}^{d} e\,(u_e * \sinh_{d-e}).
$$

In every case, the inner `*` is a Cauchy slice convolution carried
out by `cauchyAccumulateSlice` with the appropriate scale.

## Multi-index layout

Coefficients are laid out **graded reverse-lex**:

- Lower total degree first; within a degree, smaller alpha at the
  rightmost axis comes first.
- Degree-d block size: `degreeSize(d, M) = C(d+M-1, M-1)`.
- Cumulative offset to the start of degree-d: `degreeOffset(d, M) =
  C(d+M-1, M)`.

`flatIndex` and `flatIndexWithinDegree` are the inverse maps. Both
are `constexpr` so they evaluate at compile time when used through
the template-parameter accessor forms (`coeff<1, 0>()`).

## Operand storage trait

```cpp
template <class E>
using etstore_t = std::conditional_t<
    std::is_base_of_v<expr::ExprTag, E>,
    E,            // ET nodes are stored by value
    const E&>;    // storage types are stored by const&
```

Each ET node also publishes its compile-time identity through four
constants matching Eigen's traits convention:

```cpp
static constexpr int  OrderAtCompileTime;   // the operand's Order template arg
static constexpr int  VarsAtCompileTime;    //          "
static constexpr bool IsStatic;             // both are not Eigen::Dynamic
static constexpr bool IsDynamic;            // !IsStatic
```

This pattern keeps the leaves cheap (no copying user-owned TTEs into
ET nodes) while making the inner ET tree self-owning (each parent
copies its ET children into itself, surviving the constructor
boundary). Buffered nodes' `mutable` `coeffs_` and `next_` members
are duplicated across the copy boundary; the copy starts fresh and
fills its own slices.

## Source map

| Path | Contents |
|---|---|
| `include/tax/util/`           | binomial coefficients, multi-index helpers |
| `include/tax/kernels/`        | slice-aware Cauchy primitives + math recurrences |
| `include/tax/storage/tte.hpp` | unified `TaylorExpansionT<T, Order, Vars>` (static + dynamic) |
| `include/tax/expr/base.hpp`   | `ExprTag`, `etstore_t`, `coeffs_for_t`, concepts |
| `include/tax/expr/view_nodes.hpp`     | view-like ET nodes + `ParentSliceView` |
| `include/tax/expr/buffered_nodes.hpp` | buffered ET nodes |
| `include/tax/ops/arithmetic.hpp`      | `+`, `-`, `*`, `/`, scalar variants |
| `include/tax/ops/math.hpp`            | `sin`, `cos`, `exp`, `log`, … |
| `python/src/tax_module.cpp`           | nanobind module |
