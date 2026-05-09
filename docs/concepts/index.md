# Concepts

tax stands on two architectural decisions that shape the rest of the
library. Read them in order; everything else is mechanical from here.

<div class="tax-features" markdown>

<div class="tax-feature" markdown>
### [Slice streaming](slice-streaming.md)

How expression templates evaluate degree-by-degree without
materialising whole intermediate polynomials, when buffered nodes
must allocate, and what `coeffs_` exists for.
</div>

<div class="tax-feature" markdown>
### [Static vs. dynamic](static-vs-dynamic.md)

Two storage types share one mathematical kernel set: a compile-time
`TruncatedTaylorExpansionT<T, N, M>` and a runtime-sized
`DynamicTaylorExpansion<T>`. Why mixed expressions are rejected at
compile time.
</div>

</div>

## Vocabulary

A few terms recur throughout the docs:

**Truncated Taylor expansion (TTE).**
A polynomial in M variables, truncated at total degree N. Stored as a
flat `Eigen::Matrix` of `monomialCount(N, M) = C(N+M, M)` doubles.

**Multi-index.**
An M-tuple `α = (α₀, …, α_{M-1})` of non-negative integers.
`|α| = α₀ + … + α_{M-1}` is its total degree. Multi-indices are passed
across function boundaries as `std::span<const std::size_t>`.

**Graded reverse-lex order.**
The total-degree-first layout that lets the degree-d slice live in a
contiguous block of the coefficient buffer. Lower degree first; within
a degree, smaller alpha at the rightmost axis first.

**Slice.**
A degree-d slice is a contiguous Eigen `VectorBlock` of size
`degreeSize(d, M) = C(d+M-1, M-1)`. Storage types and buffered ET
nodes own dense slices; view-like ET nodes hand back a
`ParentSliceView` that computes elements lazily.

**Streaming.**
Driving an ET tree degree-by-degree via `advanceTo(d)` followed by
reading `slice(d)`. The `<<=` operator on either storage type
runs the loop.

**View-like vs. buffered ET node.**

- **View-like** — `AddExpr`, `SubExpr`, `NegExpr`, `ScalarMulExpr`,
  `ScalarAddExpr`. Allocate nothing; produce coefficients on demand.
- **Buffered** — `MulExpr`, `DivExpr`, `SquareExpr`, `SqrtExpr`,
  `ExpExpr`, `LogExpr`, `SinCosExpr`, `SinhCoshExpr`. Own a `coeffs_`
  buffer; fill it monotonically. Cauchy convolution and the standard
  DA recurrences require operand history, so the buffer is structural
  not optional.

**Static vs. dynamic kind.**
Whether a TTE has compile-time-fixed sizes (static) or runtime-fixed
sizes (dynamic). Captured by the `kStatic` constant on every
`TaxExpression`. Mixed-kind expressions are rejected by the
`SameKindExpression` concept.
