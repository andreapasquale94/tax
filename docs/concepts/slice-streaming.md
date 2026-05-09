# Slice streaming

ET nodes in tax do not materialise intermediate polynomials. They
evaluate **degree-by-degree**: a driver loop calls `advanceTo(d)` for
`d = 0, 1, …, N`, and after each call reads `slice(d)` —
the contiguous block of degree-d coefficients.

## The driver loop

`operator<<=` on either storage type delegates to
`tax::detail::streamingAssign`:

```cpp
for (std::size_t d = 0; d <= dst.order(); ++d) {
    expr.advanceTo(d);
    auto out_d = dst.slice(d);
    auto in_d  = expr.slice(d);
    for (Eigen::Index i = 0; i < in_d.size(); ++i) {
        out_d.coeffRef(i) = in_d.coeff(i);
    }
}
```

The loop writes directly into the destination's coefficient buffer.
The top of the tree allocates nothing.

## View-like nodes

`AddExpr`, `SubExpr`, `NegExpr`, `ScalarMulExpr`, `ScalarAddExpr`
own no buffer. Their `slice(d)` returns a custom
`ParentSliceView` that holds a stable reference to the parent ET node
and the degree being viewed. Element accesses recurse into the
parent's `coeffAtSlice(d, i)`:

```cpp
template <class L, class R>
class AddExpr : public Expr<AddExpr<L, R>>
{
public:
    Scalar coeffAtSlice(std::size_t d, std::size_t i) const {
        return lhs_.slice(d).coeff(i)
             + rhs_.slice(d).coeff(i);
    }

    ParentSliceView<AddExpr> slice(std::size_t d) const noexcept {
        return ParentSliceView<AddExpr>(*this, d);
    }
};
```

`ParentSliceView` is read-only and exposes only `coeff(i)` /
`size()` — exactly what kernels need on the input side.

!!! note "Why not a lazy Eigen expression?"
    Returning a `CwiseBinaryOp` from `slice(d)` is unsafe: it
    captures the operand `VectorBlock`s by reference, but those
    blocks are temporaries that die when `slice` returns. The
    custom `ParentSliceView` instead holds the parent ET (whose
    lifetime extends through the full enclosing expression) and
    walks down to leaf storage on every access.

## Buffered nodes

`MulExpr`, `DivExpr`, `SquareExpr`, `SqrtExpr`, `ExpExpr`, `LogExpr`,
`SinCosExpr`, `SinhCoshExpr` each own a `coeffs_` buffer sized like a
TTE in the same kind. Their `advanceTo(d)` fills the d-slice (and any
auxiliary state — `SinCosExpr` keeps the cos buffer alive alongside
sin) using the appropriate kernel:

```cpp
void advanceTo(std::size_t d) const {
    if (d < next_) return;
    lhs_.advanceTo(d);
    rhs_.advanceTo(d);
    for (std::size_t e = next_; e <= d; ++e) {
        auto out_e = slice(e);
        kernels::cauchyMulComputeDegree<Scalar>(
            e, nvars(), lhs_, rhs_, out_e);
    }
    next_ = d + 1;
}
```

`coeffs_` and `next_` are `mutable` so that a buffered node can sit
behind a `const&` in a parent ET tree and still be advanced.

!!! warning "Buffered nodes structurally must allocate"
    Cauchy convolution computes `c_α = Σ a_β · b_γ` over all `β + γ
    = α`. Computing `c_d` from `c_{d-1}` requires every prior
    `a_e` and `b_e`. There is no architecture trick that
    eliminates the operand history, so don't try to remove
    `coeffs_` from buffered ET nodes.

## Operand storage

ET nodes face a lifetime question: an intermediate `NegExpr<R>` built
inside `c - r` is a prvalue, alive only for its constructor's full
expression. Storing it by `const&` in a parent
`ScalarAddExpr<NegExpr<R>>` would dangle. tax solves this with a
trait:

```cpp
template <class E>
using etstore_t = std::conditional_t<
    std::is_base_of_v<ExprTag, E>,
    E,         // ET nodes are stored by value
    const E&>; // storage types are stored by const&
```

Storage types (which the user owns and keeps alive) bind by reference;
ET temporaries are copied into their parent and survive the
constructor's scope. Buffered nodes' `mutable` state is duplicated
across the copy boundary, but the copy starts fresh and computes
its own slices.

## Kernel signatures

Slice-aware kernels take operand objects (anything with `slice`)
and write into the destination's slice. Cauchy convolution is the
common primitive:

```cpp
// out[α-rank] += scale * a[β-rank] * b[γ-rank]
// for all (β, γ) with |β| = eA, |γ| = eB, α = β + γ.
template <class T, class A, class B, class Out>
void cauchyAccumulateSlice(std::size_t eA, std::size_t eB,
                           std::size_t nvars,
                           T scale,
                           const A& a, const B& b, Out& out);
```

Both inputs only need `coeff(i)`; the output needs `coeffRef(i)`.
Lazy `ParentSliceView`s satisfy the input contract;
`Eigen::VectorBlock`s satisfy both.

## Why this matters

A naive expression-template library materialises every intermediate
polynomial: `(a + b) * (c + d)` would allocate four `Eigen::Vector`s
in the worst case. tax allocates exactly one buffer per `MulExpr`,
`DivExpr`, or transcendental — the irreducible storage demanded by
Cauchy convolution — and writes the final polynomial straight into
the user's destination. There are no temporary TTE objects.
