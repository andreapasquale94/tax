# tax Reorg — Phase 3: extract core/meta.hpp + core/axis.hpp; collapse Merge — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the shared axis metaprogramming out of `core/named.hpp` into `core/meta.hpp` (string/type-list primitives) and `core/axis.hpp` (axis types + lookups + remap), so `core/mixed_named.hpp` no longer `#include`s `core/named.hpp` just to borrow it; then collapse the near-duplicate `Merge` (named) and `MergeOrdered` (mixed) families into one policy-parameterised `Merge`.

**Architecture:** Phase 3 of the reorg (spec `docs/superpowers/specs/2026-06-28-tax-library-reorganization-design.md`, the M-A "shared meta + axis machinery" item; the `AxisCarrier` CRTP idea is **dropped** — the named/mixed classes are not true twins). Builds on Phase 2 (branch `claude/tax-library-reorg`, at `0044383`). Directories stay `core/`/`series/` (the tree move is Phase 6).

**Tech Stack:** Header-only C++23 templates/concepts; GoogleTest; mamba `tax` env.

## Global Constraints

- C++23; `constexpr` everywhere in the dense core; no heap in the dense core; graded-lex ordering sacred; kernel macros in-header (ODR); `M ≥ 1`.
- Build/test (repo root, mamba `tax` env active):
  `source /Users/andrea/miniforge3/etc/profile.d/conda.sh && conda activate tax`
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
  Full suite is **56 tests**, 100% passing, through every task (pure refactor — no test count change).
- `clang-format` touched files; preserve indented `#    define` PP directives (none expected in P3 files).
- **Namespace is preserved:** the moved symbols stay in their current namespace `tax::named::detail` (and the public `tax::named::FixedString`/`tax::named::Axis` spellings stay valid). Only the *file* they live in changes. (Generalizing the namespace to `tax::detail` is out of scope here — it would churn every `detail::` use; revisit at the P6 tree move if desired.)

**TDD note:** both tasks are refactors validated by the 56-test suite (which already exercises named + mixed composition, embed/slice, and merge across operands) plus targeted greps. No new runtime tests; Task 2 relies on the existing named/mixed composition tests to prove the unified `Merge` behaves identically for both combine policies.

---

### Task 1: Extract core/meta.hpp + core/axis.hpp; stop mixed borrowing named.hpp

**Files:**
- Create: `include/tax/core/meta.hpp` — the string + type-list primitives, moved verbatim from `core/named.hpp`: `FixedString`, `compareFixed`, `fixedCompare`, `TypeList`, `Prepend`, `MergeFold` (the generic left-fold). Keep them in `namespace tax::named::detail` (and `FixedString` in `tax::named` if it's currently re-exported there — preserve exactly the current namespacing).
- Create: `include/tax/core/axis.hpp` — the axis types + lookups + remap, moved verbatim from `core/named.hpp`: `Axis`, `axisSign`, `Merge`/`MergeChoose` (leave these here for now — Task 2 collapses them), `OffsetOf`, `DimOfName`, `TotalDim`, `IsCanonical`, `IsSubsetOf`, `buildAxisMapImpl`, `buildAxisMap`. `core/axis.hpp` includes `core/meta.hpp` (it uses `TypeList`/`Prepend`/`FixedString`).
- Modify: `include/tax/core/named.hpp` — delete the moved definitions; add `#include <tax/core/axis.hpp>` (which transitively brings `meta.hpp`); keep everything named-specific (the `NamedExpansion` class, `MergedNamedExpansion`, the named factories, embed/slice members, re-exports).
- Modify: `include/tax/core/mixed_named.hpp` — replace `#include <tax/core/named.hpp>` with `#include <tax/core/axis.hpp>` **iff** mixed_named.hpp uses nothing else from named.hpp (verify in Step 1); otherwise keep both includes. It must still see `FixedString`/`TypeList`/`Prepend`/`axisSign`/`OffsetOf`/`DimOfName`/`TotalDim`/`IsCanonical`/`IsSubsetOf`/`buildAxisMap` (all now via `core/axis.hpp`).

**Interfaces:**
- Consumes: nothing new.
- Produces: `<tax/core/meta.hpp>` and `<tax/core/axis.hpp>` carrying the shared machinery (unchanged symbols/namespace); `core/mixed_named.hpp` no longer depends on `core/named.hpp` for meta.

- [ ] **Step 1: Determine whether mixed_named.hpp needs anything named-specific**

Run:
```bash
grep -nE "NamedExpansion|MergedNamedExpansion|named::variable|named::variables" include/tax/core/mixed_named.hpp
```
If this returns nothing (expected — mixed only borrows the meta/axis primitives), Step 6 will replace its `core/named.hpp` include with `core/axis.hpp`. If it DOES reference a named-specific symbol, keep the `core/named.hpp` include in addition to `core/axis.hpp` and note it. Also confirm the exact namespace the meta lives in:
```bash
grep -nE "^namespace|^\}  // namespace" include/tax/core/named.hpp | head
```
(Expected: a `tax::named` block with a nested `detail` block. Preserve exactly that nesting in the new files.)

- [ ] **Step 2: Create core/meta.hpp**

Create `include/tax/core/meta.hpp` with `#pragma once`, the includes the moved code needs (`<array>` is NOT needed here; `<cstddef>` for `std::size_t`; whatever `compareFixed`/`FixedString` use), the same `namespace tax::named { ... namespace detail { ... } ... }` nesting as `named.hpp`, and these symbols **moved verbatim** from `core/named.hpp` (cut them from named.hpp): `FixedString` (struct + its members), `compareFixed`, `fixedCompare`, `TypeList`, `Prepend`, and `MergeFold` (the generic left-fold at named.hpp ~lines 159-169). Keep `FixedString` at whatever scope it currently has (it is used as an NTTP `template < FixedString Name >`, so it is likely in `tax::named` directly, not `detail` — preserve that). Add a short file-header comment: "Compile-time string + type-list primitives shared by the named and mixed axis layers."

- [ ] **Step 3: Create core/axis.hpp**

Create `include/tax/core/axis.hpp` with `#pragma once`, `#include <array>`, `#include <cstddef>`, `#include <tax/core/meta.hpp>`, the same namespace nesting, and these symbols **moved verbatim** from `core/named.hpp`: `Axis`, `axisSign`, `Merge`/`MergeChoose` (the named single-order family — Task 2 will generalize), `OffsetOf`, `DimOfName`, `TotalDim`, `IsCanonical`, `IsSubsetOf`, `buildAxisMapImpl`, `buildAxisMap`. (`Axis` is the public `tax::named::Axis` — keep its scope.) File-header comment: "Named axis types, axis-set lookups, merge, and the source→target remap shared by the named and mixed layers."

- [ ] **Step 4: Trim core/named.hpp and include axis.hpp**

In `core/named.hpp`, delete every definition moved in Steps 2-3 (they now live in `meta.hpp`/`axis.hpp`). Add `#include <tax/core/axis.hpp>` near the top (replacing the now-removed primitives). Leave all named-specific code intact: `NamedExpansion`, `MergedNamedExpansion`/`RebindNamed` (if present), the `variable`/`variables` factories, the class's embed/slice/deriv/integ members, and the `tax::` re-exports. Verify `core/named.hpp` still compiles standalone conceptually (it will be built via the suite).

- [ ] **Step 5: Verify no symbol was lost or duplicated**

Run:
```bash
for s in FixedString compareFixed fixedCompare TypeList Prepend MergeFold Axis axisSign OffsetOf DimOfName TotalDim IsCanonical IsSubsetOf buildAxisMap; do
  echo -n "$s: "; grep -rln "struct $s\b\|constexpr.* $s\b\|using $s\b\|$s =" include/tax/core/meta.hpp include/tax/core/axis.hpp | tr '\n' ' '; echo
done
echo "--- named.hpp must NOT still define them ---"
grep -nE "struct FixedString|struct TypeList|struct Prepend|struct Axis\b|struct OffsetOf|struct DimOfName|struct MergeFold" include/tax/core/named.hpp || echo "named.hpp clean of moved defs"
```
Expected: each symbol defined in exactly one of meta.hpp/axis.hpp; named.hpp no longer defines them.

- [ ] **Step 6: Repoint mixed_named.hpp**

In `include/tax/core/mixed_named.hpp`, replace `#include <tax/core/named.hpp>` with `#include <tax/core/axis.hpp>` (or add `core/axis.hpp` and keep `core/named.hpp` only if Step 1 found a named-specific dependency). Then:
```bash
grep -n "tax/core/named.hpp" include/tax/core/mixed_named.hpp || echo "mixed no longer includes named.hpp"
```

- [ ] **Step 7: Build + full suite**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
Expected: build EXIT 0; `100% tests passed … out of 56`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor: extract core/meta.hpp + core/axis.hpp from named.hpp\n\nLift the shared compile-time string/type-list primitives (FixedString,\nTypeList, Prepend, MergeFold) into core/meta.hpp and the axis types + lookups +\nremap (Axis, axisSign, Merge, OffsetOf, DimOfName, TotalDim, IsCanonical,\nIsSubsetOf, buildAxisMap) into core/axis.hpp. core/mixed_named.hpp now includes\ncore/axis.hpp instead of core/named.hpp -- it no longer depends on the named\nlayer just to borrow shared machinery. Symbols + namespace unchanged.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Collapse Merge + MergeOrdered into one policy-parameterised Merge

**Files:**
- Modify: `include/tax/core/axis.hpp` — generalize `Merge`/`MergeChoose` to take a same-name `Combine` policy; provide the named default policy.
- Modify: `include/tax/core/mixed_named.hpp` — delete `MergeOrdered`/`MergeOrderedChoose`/`MergeFoldOrdered`; supply the mixed (max-order) combine policy and use the unified `Merge`/`MergeFold`.

**Interfaces:**
- Consumes: the `Merge`/`MergeChoose` now in `core/axis.hpp` (Task 1).
- Produces: a single `Merge< A, B, Combine >` (default `Combine = SameNameRequireEqual`) where `Combine::template apply< A0, B0 >` yields the merged axis for two same-named axes. The named layer keeps current behavior (default policy); the mixed layer passes a `MaxOrder` policy.

- [ ] **Step 1: Generalize Merge in core/axis.hpp**

The only difference between the named `MergeChoose<0>` (require equal dim, take A0) and the mixed `MergeOrderedChoose<0>` (require equal dim, take `OrderedAxis<name, dim, max(order)>`) is the same-name combine. Introduce a policy and thread it through. Replace the `Merge`/`MergeChoose` family in `core/axis.hpp` with:
```cpp
// Same-name combine policy for Merge: given two same-named axes, produce the
// merged axis. The default requires identical dimension and keeps the first
// (single-order named layer). The mixed layer supplies a max-order policy.
struct SameNameRequireEqual
{
    template < typename A0, typename B0 >
    struct apply
    {
        static_assert( A0::dim == B0::dim,
                       "named axis used with inconsistent dimension across operands" );
        using type = A0;
    };
};

template < int Cmp, typename A, typename B, typename Combine >
struct MergeChoose;

template < typename A, typename B, typename Combine = SameNameRequireEqual >
struct Merge;

template < typename... Bs, typename Combine >
struct Merge< TypeList<>, TypeList< Bs... >, Combine >
{
    using type = TypeList< Bs... >;
};
template < typename A0, typename... As, typename Combine >
struct Merge< TypeList< A0, As... >, TypeList<>, Combine >
{
    using type = TypeList< A0, As... >;
};
template < typename A0, typename... As, typename B0, typename... Bs, typename Combine >
struct Merge< TypeList< A0, As... >, TypeList< B0, Bs... >, Combine >
    : MergeChoose< axisSign< A0, B0 >, TypeList< A0, As... >, TypeList< B0, Bs... >, Combine >
{
};

// A0 < B0 : take A0
template < typename A0, typename... As, typename B0, typename... Bs, typename Combine >
struct MergeChoose< -1, TypeList< A0, As... >, TypeList< B0, Bs... >, Combine >
{
    using type = typename Prepend<
        A0, typename Merge< TypeList< As... >, TypeList< B0, Bs... >, Combine >::type >::type;
};
// A0 > B0 : take B0
template < typename A0, typename... As, typename B0, typename... Bs, typename Combine >
struct MergeChoose< 1, TypeList< A0, As... >, TypeList< B0, Bs... >, Combine >
{
    using type = typename Prepend<
        B0, typename Merge< TypeList< A0, As... >, TypeList< Bs... >, Combine >::type >::type;
};
// A0 == B0 (same name) : combine via policy, advance both
template < typename A0, typename... As, typename B0, typename... Bs, typename Combine >
struct MergeChoose< 0, TypeList< A0, As... >, TypeList< B0, Bs... >, Combine >
{
    using Merged = typename Combine::template apply< A0, B0 >::type;
    using type = typename Prepend<
        Merged, typename Merge< TypeList< As... >, TypeList< Bs... >, Combine >::type >::type;
};
```
Generalize `MergeFold` (in `core/meta.hpp` or `axis.hpp`, wherever it lives) to forward a `Combine` policy too:
```cpp
template < typename Combine, typename Acc, typename... Rest >
struct MergeFoldWith
{
    using type = Acc;
};
template < typename Combine, typename Acc, typename First, typename... Rest >
struct MergeFoldWith< Combine, Acc, First, Rest... >
{
    using type =
        typename MergeFoldWith< Combine, typename Merge< Acc, First, Combine >::type, Rest... >::type;
};
```
Keep the existing `MergeFold<Acc, Rest...>` as a thin alias forwarding `SameNameRequireEqual` so named call sites are unchanged:
```cpp
template < typename Acc, typename... Rest >
using MergeFold = MergeFoldWith< SameNameRequireEqual, Acc, Rest... >;
```
(If `MergeFold` is currently a `struct` with `::type`, keep that shape: make it derive from `MergeFoldWith< SameNameRequireEqual, Acc, Rest... >`.)

- [ ] **Step 2: Verify named call sites still compile (default policy)**

The named layer's `Merge< A, B >` / `MergeFold< ... >` uses now bind `Combine = SameNameRequireEqual` by default — no named call-site edits should be needed. Build just the named tests to confirm before touching mixed:
```bash
cmake --build build --target test_named -j 2>&1 | tail -3
```
Expected: builds. (If a named call site passed `Merge<A,B>` positionally with a 3rd arg, fix it — none expected.)

- [ ] **Step 3: Replace the mixed MergeOrdered family with the policy**

In `include/tax/core/mixed_named.hpp`, delete `MergeOrderedChoose`, `MergeOrdered`, and `MergeFoldOrdered` (the parallel family). Add a mixed combine policy next to `OrderedAxis`:
```cpp
struct SameNameMaxOrder
{
    template < typename A0, typename B0 >
    struct apply
    {
        static_assert( A0::dim == B0::dim,
                       "named axis used with inconsistent dimension across operands" );
        using type =
            OrderedAxis< A0::name, A0::dim, ( A0::order > B0::order ? A0::order : B0::order ) >;
    };
};
```
Then replace every `MergeOrdered< ListA, ListB >::type` with `Merge< ListA, ListB, SameNameMaxOrder >::type`, and every `MergeFoldOrdered< Acc, Rest... >::type` with `MergeFoldWith< SameNameMaxOrder, Acc, Rest... >::type`. (Grep `MergeOrdered\|MergeFoldOrdered` in mixed_named.hpp first to get the exact use sites — `MergedMixedTaylorExpansion` at ~line 119 and the `slice()` fold.)

- [ ] **Step 4: Verify the old family is gone**

```bash
grep -rn "MergeOrdered\|MergeFoldOrdered\|MergeOrderedChoose" include/ && echo "STILL PRESENT — fix" || echo "MergeOrdered family removed"
```
Expected: removed.

- [ ] **Step 5: Build + full suite**

Run: `cmake --build build -j && ctest --test-dir build -j`
Expected: `100% tests passed … out of 56`. The existing mixed tests (`test_mixed_named`, `test_mixed_named_la`) exercise union-with-max-order across operands and slice — they prove the unified `Merge` + `SameNameMaxOrder` matches the old `MergeOrdered` behavior. If a mixed composition test fails, the `SameNameMaxOrder` policy or a use-site substitution is wrong — STOP and report (do not loosen a test).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor: collapse Merge + MergeOrdered into one policy-parameterised Merge\n\nThe named (take-first) and mixed (max-order) axis merges differed only in the\nsame-name combine step. Generalize core/axis.hpp Merge to take a Combine policy\n(default SameNameRequireEqual); the mixed layer supplies SameNameMaxOrder and\ndrops its parallel MergeOrdered/MergeFoldOrdered family.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Phase exit criteria

- `core/meta.hpp` + `core/axis.hpp` exist with the shared machinery (each symbol defined once); `core/named.hpp` no longer defines them and includes `core/axis.hpp`; `core/mixed_named.hpp` includes `core/axis.hpp` (and not `core/named.hpp`, unless a named-specific dep was found).
- The `MergeOrdered` family is gone; a single policy-parameterised `Merge` serves both layers.
- `ctest` → `100% tests passed … out of 56`. Two commits landed.

## Self-review (completed)

- **Spec coverage (P3 slice):** extract `meta.hpp`/`axis.hpp` + stop mixed borrowing named (M-A 3a) → Task 1; collapse `Merge`/`MergeOrdered` (M-A 3b) → Task 2; `AxisCarrier` CRTP explicitly dropped (not attempted). ✔
- **Placeholders:** none — Task 1 enumerates the exact symbols to move + verification greps; Task 2 gives the full unified `Merge`/`MergeChoose`/`MergeFoldWith` code + both combine policies + the use-site substitution rule.
- **Type/name consistency:** symbols keep their `tax::named[::detail]` namespace; `MergeFold` preserved as a default-policy alias so named call sites are untouched; `SameNameRequireEqual` (named default) and `SameNameMaxOrder` (mixed) realize the only behavioral difference (take-first vs max-order); `Combine::template apply<A0,B0>::type` is the single combine hook.
- **Risk note:** Task 2 is delicate compile-time metaprogramming. If the policy generalization proves intractable, it is acceptable to ship Task 1 alone (the higher-value extraction that severs mixed→named) and defer the `Merge` collapse — report rather than force it.
