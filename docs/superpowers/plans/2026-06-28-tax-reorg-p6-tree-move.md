# tax Reorg — Phase 6: the directory tree move + facades + M-B — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the source tree to the feature-module layout — `core/`→`expansion/`, `kernels/`→`expansion/detail/`, `operators/`→`expansion/ops/`, `series/`→`bases/`, `la/`→`la/` (namespace **stays `tax::la`**) — add the `expansion.hpp`/`bases.hpp`/`la.hpp` facades and shrink the umbrella; then move the mixed *type* into `tax::mixed` (the deferred M-B).

**Architecture:** Phase 6 of the reorg (spec `docs/superpowers/specs/2026-06-28-tax-library-reorganization-design.md`, the tree move + facades + M-B). Builds on Phase 5 (branch `claude/tax-library-reorg`, at `5fe5fa2`). **Scope: directory moves only — filenames are KEPT** (e.g. `bases/chebyshev_basis.hpp`, `expansion/mixed_named.hpp`); file renames (`*_basis.hpp`→`*.hpp`, `mixed_named`→`mixed`) are deferred (note for a later optional polish). The `la/` directory holds namespace `tax::la` (intentional dir/namespace divergence, D5).

**Tech Stack:** Header-only C++23; Eigen3; GoogleTest; mamba `tax` env.

## Global Constraints

- C++23; `constexpr` core; no heap in dense core; graded-lex ordering sacred; kernel config macros in-header (ODR) — **the `TAX_USE_*`/`TAX_STENCIL_MAX_BYTES` macros live in `kernels/cauchy.hpp` + `kernels/cauchy_stencil.hpp`; after the move they're in `expansion/detail/`; their `#ifndef`-guarded definitions and include order must be preserved** (do not let the move change which file defines them or their order). M≥1.
- Build/test (repo root, mamba `tax` env active):
  `source /Users/andrea/miniforge3/etc/profile.d/conda.sh && conda activate tax`
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
  Suite is **58 tests**, 100% passing, after each task.
- `clang-format`: only touched-by-hand files; the bulk move is `git mv` + include-path sed (no formatting). Preserve indented `#    define` PP directives.
- Consumer contract stays "include `<tax/tax.hpp>`". Internal `tax::` API names unchanged by Tasks 1-2 (pure relocation). Task 3 moves the mixed type's namespace (with `using` surfacing under `tax::` preserved).

**TDD note:** Tasks 1-2 are mechanical relocations verified by the 58-test suite + grep checks (no stale paths). Task 3 (M-B) is a namespace move verified by the suite + the existing mixed tests.

---

### Task 1: Move all five directories + repoint every include path

**Files:** `git mv` of the five directories; global include-path rewrite across `include/` + `tests/`.

**Interfaces:**
- Produces: the new directory layout `expansion/` (with `detail/` = former kernels, `ops/` = former operators), `bases/` (former series), `la/` (former la); every `#include` repointed. `io/` unchanged. The umbrella still lists the (repointed) specific headers — facades are Task 2.

- [ ] **Step 1: Move the directories with git mv (order matters: nest kernels/operators under expansion)**

```bash
cd /Users/andrea/Documents/Codes/tax
git mv include/tax/core include/tax/expansion
git mv include/tax/kernels include/tax/expansion/detail
git mv include/tax/operators include/tax/expansion/ops
git mv include/tax/series include/tax/bases
git mv include/tax/la include/tax/la
# io/ stays.
```
Verify the new tree: `find include/tax -maxdepth 2 -type d | sort` should show `expansion`, `expansion/detail`, `expansion/ops`, `expansion/scheme`, `expansion/storage`, `bases`, `la`, `io`.

- [ ] **Step 2: Repoint every include path (global, ordered so the longest prefixes win)**

Apply these substitutions to every `.hpp`/`.cpp` under `include/` and `tests/`. **Order is critical** — rewrite `core/kernels`-style nested first is N/A; instead rewrite the leaf dirs that move INTO expansion BEFORE the generic `core→expansion`, because `tax/kernels/` and `tax/operators/` are siblings of `core/`, not under it, so order among the five is independent. Use:
```bash
cd /Users/andrea/Documents/Codes/tax
FILES=$(grep -rl "tax/core/\|tax/kernels/\|tax/operators/\|tax/series/\|tax/la/\|tax/la\.hpp\|tax/series\.hpp" include tests)
for f in $FILES; do
  perl -pi -e '
    s{tax/kernels/}{tax/expansion/detail/}g;
    s{tax/operators/}{tax/expansion/ops/}g;
    s{tax/core/}{tax/expansion/}g;
    s{tax/series/}{tax/bases/}g;
    s{tax/series\.hpp}{tax/bases.hpp}g;
    s{tax/la/}{tax/la/}g;
    s{tax/la\.hpp}{tax/la.hpp}g;
  ' "$f"
done
```
(`tax/series/` is rewritten before `tax/series.hpp` is matched by the separate `series\.hpp` rule — both rules run per file, the `/` rule only matches `series/`, the `.hpp` rule only `series.hpp`; they don't overlap. Same for `la`.)

- [ ] **Step 3: Rename the two existing top-level facades to match (la.hpp→la.hpp, series.hpp→bases.hpp)**

```bash
git mv include/tax/la.hpp include/tax/la.hpp
git mv include/tax/series.hpp include/tax/bases.hpp
```
These two facades' *internal* includes were already repointed in Step 2 (their `tax/la/…`→`tax/la/…`, `tax/series/…`→`tax/bases/…`). The umbrella's `#include <tax/la.hpp>`/`<tax/series.hpp>` were repointed to `<tax/la.hpp>`/`<tax/bases.hpp>` in Step 2 as well.

- [ ] **Step 4: Verify no stale paths and the tree is coherent**

```bash
grep -rn "tax/core/\|tax/kernels/\|tax/operators/\|tax/series/\|tax/series\.hpp\|tax/la/\|tax/la\.hpp" include tests \
  | grep -v "// " || echo "no stale include paths"
ls include/tax/core include/tax/kernels include/tax/operators include/tax/series include/tax/la 2>/dev/null || echo "old dirs gone"
```
Expected: "no stale include paths" (ignore any prose comments that mention old names — those are cosmetic; fix obvious ones if trivial) and "old dirs gone". If a real `#include` of an old path remains, repoint it.

- [ ] **Step 5: Build + full suite**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
Expected: build EXIT 0; `100% tests passed … out of 58`. (Pure relocation — the build is the proof every path resolved.) If the build fails on a missing header, a path was missed — find it via the error and repoint.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor!: move the source tree to the feature-module layout\n\ncore/->expansion/, kernels/->expansion/detail/, operators/->expansion/ops/,\nseries/->bases/, la/->la/ (namespace stays tax::la). Repoint every include\npath; rename the la.hpp/series.hpp facades to la.hpp/bases.hpp. Pure\nrelocation (git mv + include sed); filenames kept; 58/58.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Add the expansion.hpp facade; shrink the umbrella to four facades

**Files:**
- Create: `include/tax/expansion.hpp` (facade aggregating the expansion capability)
- Modify: `include/tax/tax.hpp` (shrink to the four facades)
- (`bases.hpp` and `la.hpp` already exist from Task 1; confirm they aggregate their layers.)

**Interfaces:**
- Consumes: the relocated headers from Task 1.
- Produces: `<tax/expansion.hpp>` (carrier + named/mixed + operators + Taylor basis), `<tax/bases.hpp>`, `<tax/la.hpp>` as the three sub-facades; `<tax/tax.hpp>` = `version.hpp` + those three + `io/series.hpp`.

- [ ] **Step 1: Read the current umbrella to capture the exact dependency order**

`tax.hpp` currently `#include`s the specific headers in a deliberate dependency order. Read it. The expansion-capability headers (everything that is NOT a `bases/`, `la/`, or `io/` header, and not `version.hpp`) move INTO `expansion.hpp` in the SAME relative order.

- [ ] **Step 2: Create include/tax/expansion.hpp**

`#pragma once`; file-header comment ("Facade for the core expansion capability: the Expansion carrier, schemes, storage, the Taylor basis policy, named + mixed expansions, and the operator surface. Kernels are an internal detail (expansion/detail/)."). Then `#include` — in the SAME order they appear in the current `tax.hpp` — every expansion-capability header the umbrella lists, with paths in the new layout, e.g.:
```cpp
#include <tax/expansion/concepts.hpp>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/enumeration.hpp>
#include <tax/expansion/storage/dense.hpp>
#include <tax/expansion/storage/sparse.hpp>
#include <tax/expansion/expansion.hpp>
#include <tax/expansion/named.hpp>
#include <tax/expansion/ops/named_arithmetic.hpp>
#include <tax/expansion/ops/named_math_unary.hpp>
#include <tax/expansion/ops/named_math_binary.hpp>
#include <tax/expansion/mixed_named.hpp>
#include <tax/expansion/ops/mixed_arithmetic.hpp>
#include <tax/expansion/ops/mixed_math_unary.hpp>
#include <tax/expansion/ops/mixed_math_binary.hpp>
#include <tax/expansion/ops/arithmetic.hpp>
#include <tax/expansion/ops/math_unary.hpp>
#include <tax/expansion/ops/math_binary.hpp>
#include <tax/expansion/promote.hpp>
```
(Match the EXACT set + order from the current `tax.hpp`, repointed. `concepts`/`meta`/`axis`/`basis`/`taylor_basis`/`scheme` are pulled transitively by `expansion.hpp` + `named.hpp`; include explicitly only what the umbrella already listed, to preserve behavior.)

- [ ] **Step 3: Shrink the umbrella**

Replace the body of `include/tax/tax.hpp` (keeping its opening comment + `#pragma once`) with:
```cpp
#include <tax/version.hpp>
#include <tax/expansion.hpp>
#include <tax/bases.hpp>
#include <tax/la.hpp>
#include <tax/io/series.hpp>
```
(Order: expansion → bases → la → io, the same downward order as before. `bases.hpp` needs the carrier + Taylor basis from `expansion.hpp`; `la.hpp` needs both; `io` needs all — so this order is correct.)

- [ ] **Step 4: Verify the facades are self-consistent**

```bash
grep -c "#include" include/tax/expansion.hpp   # > 10
grep -n "#include" include/tax/tax.hpp           # exactly the 5 facade lines
```

- [ ] **Step 5: Build + full suite**

Run: `cmake --build build -j && ctest --test-dir build -j`
Expected: build EXIT 0; `100% tests passed … out of 58`. If a symbol is now undefined, the facade include order is off — compare against the pre-shrink `tax.hpp` order and fix.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor: add expansion.hpp facade; shrink umbrella to four facades\n\ntax.hpp is now version.hpp + expansion.hpp + bases.hpp + la.hpp +\nio/series.hpp. The new expansion.hpp aggregates the core capability (carrier +\nnamed/mixed + operators + Taylor basis; kernels are the internal\nexpansion/detail/). Same include order, behavior unchanged; 58/58.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: Move the mixed type into tax::mixed (deferred M-B)

**Files:**
- Modify: `include/tax/expansion/mixed_named.hpp` (move `MixedTaylorExpansion`/`OrderedAxis`/`MTE` from `tax::named` to `tax::mixed`; the shared `detail` helpers it uses are in `tax::named::detail` — qualify or alias)
- Modify: `include/tax/expansion/ops/mixed_*.hpp` (the mixed operators currently in `tax::named` — move to `tax::mixed`)
- Modify: `include/tax/la/mixed_named.hpp` (NumTraits + the mixed la helpers currently in `tax::named` — move to `tax::mixed`; update the `tax::` re-export in `la/exports.hpp`)
- Modify: `include/tax/la/exports.hpp` (the named/mixed surfacing)

**Interfaces:**
- Consumes: the relocated tree.
- Produces: `tax::mixed::MixedTaylorExpansion`/`OrderedAxis`/`MTE` (the *type* now lives in `tax::mixed` alongside its factories); still surfaced under `tax::` via `using mixed::…`. The `tax::named` namespace no longer holds the mixed type.

- [ ] **Step 1: Inventory the mixed type's `tax::named` footprint + the detail helpers it borrows**

```bash
grep -rn "MixedTaylorExpansion\|OrderedAxis\|MergedMixedTaylorExpansion\|RebindMixed\|AxesToMixedScheme\|SameNameMaxOrder\|OrderOfName\|ReplaceAxisOrder" include/tax/expansion/mixed_named.hpp | head -40
grep -rn "namespace tax::named\|namespace tax::mixed\|detail::" include/tax/expansion/mixed_named.hpp | head
```
Identify which symbols are mixed-specific (move to `tax::mixed`) vs borrowed from `tax::named::detail` (FixedString/TypeList/Prepend/axisSign/OffsetOf/DimOfName/TotalDim/IsCanonical/IsSubsetOf/buildAxisMap/Merge — these now live in `expansion/axis.hpp`+`meta.hpp`, namespace `tax::named::detail`). The borrowed ones STAY in `tax::named::detail`; the mixed type's body must reference them as `tax::named::detail::X` (or add `namespace tax::mixed::detail { namespace nd = tax::named::detail; }` and use `nd::X`).

- [ ] **Step 2: Move the mixed TYPE + its mixed-specific detail into `tax::mixed`**

In `expansion/mixed_named.hpp`: change the enclosing `namespace tax::named { … }` (the block holding `OrderedAxis`, the mixed-specific `detail` family — `AxesToMixedScheme`, `MergeOrdered`-replacement `SameNameMaxOrder`, `RebindMixed`, `MergedMixedTaylorExpansion`, `OrderOfName`, `ReplaceAxisOrder`, `MergeFoldWith` usage —, `MixedTaylorExpansion`, `MTE`) to `namespace tax::mixed { … }`. Inside, qualify the borrowed shared helpers as `tax::named::detail::…` (or via the `nd` alias). Keep `FixedString`/`Axis` references working (they are `tax::named::…`; use `tax::named::FixedString` for the NTTP, or `using tax::named::FixedString;`). The factories already in `tax::mixed` (`variable`/`variables`) now sit in the same namespace — drop their `tax::named::` qualification on the type.

- [ ] **Step 3: Surface the mixed type under `tax::`**

Replace the old `namespace tax { using named::MixedTaylorExpansion; using named::MTE; using named::OrderedAxis; }` with `using mixed::MixedTaylorExpansion; using mixed::MTE; using mixed::OrderedAxis;` (type names don't collide; this keeps `tax::MTE` etc. valid).

- [ ] **Step 4: Move the mixed operators + la helpers to `tax::mixed`**

- `expansion/ops/mixed_arithmetic.hpp`, `mixed_math_unary.hpp`, `mixed_math_binary.hpp`: change `namespace tax::named` → `namespace tax::mixed` (the operators dispatch on `MixedTaylorExpansion`, now in `tax::mixed`, so ADL finds them there). The `detail::MergedMixedTaylorExpansion` references resolve to `tax::mixed::detail` now (it moved with the type).
- `la/mixed_named.hpp`: the NumTraits specialization is in namespace `Eigen` (references `tax::named::MixedTaylorExpansion` → update to `tax::mixed::MixedTaylorExpansion`). The per-axis `gradient`/`hessian`/`jacobian` for mixed + `is_mixed` are in `tax::named` → move to `tax::mixed`.
- `la/exports.hpp`: the `using named::gradient;` etc. captured the mixed overloads (which were in `tax::named`); now add `using mixed::gradient; using mixed::hessian; using mixed::jacobian;` so the mixed la helpers are still surfaced under `tax::`.

- [ ] **Step 5: Verify + build + full suite**

```bash
grep -rn "tax::named::MixedTaylorExpansion\|named::MixedTaylorExpansion" include/ && echo "STILL REFS named::Mixed — fix" || echo "mixed type fully in tax::mixed"
cmake --build build -j && ctest --test-dir build -j
```
Expected: the grep echoes success; `100% tests passed … out of 58`. The mixed tests (`test_mixed_named`, `test_mixed_named_la`, `test_mixed_te`) exercise the type, operators, and la helpers — they prove the namespace move is complete and ADL still resolves. If a mixed test fails to compile, a reference or an ADL path was missed — STOP and report (this is the delicate task; do not hack).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor!: move the mixed type into tax::mixed (M-B)\n\nMixedTaylorExpansion/OrderedAxis/MTE + their operators and la helpers move from\ntax::named to tax::mixed (alongside the tax::mixed factories); shared axis meta\nstays in tax::named::detail (qualified). Surfaced under tax:: via using mixed::.\nThe mixed type no longer lives in the named namespace; 58/58.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Phase exit criteria

- Tree is `expansion/` (+ `detail/`, `ops/`, `scheme/`, `storage/`), `bases/`, `la/`, `io/`; old dir names gone; no stale include paths.
- Umbrella = `version.hpp` + `expansion.hpp` + `bases.hpp` + `la.hpp` + `io/series.hpp`; the three facades aggregate their layers.
- The mixed type lives in `tax::mixed` (M-B), surfaced under `tax::`.
- `ctest` → `100% tests passed … out of 58`. Three commits landed.

## Self-review (completed)

- **Spec coverage (P6 slice):** the five directory moves + facades + umbrella shrink (the tree move) → Tasks 1-2; M-B (mixed type → tax::mixed) → Task 3. File renames (`*_basis`→`*`, `mixed_named`→`mixed`) deferred (noted) to bound risk. ✔
- **Placeholders:** none — exact `git mv` sequence, the ordered perl repoint, the facade skeleton (match-the-umbrella-order instruction), the M-B namespace-move steps with the borrowed-detail-qualification rule, and grep verifications.
- **Risk notes:** Task 1 is wide but pure-mechanical (build verifies every path). Task 2's only risk is facade include ORDER — mitigated by "match the current tax.hpp order." Task 3 (M-B) is the delicate one (namespace move + ADL); it has a STOP-and-report guard and is isolated as the last task so Tasks 1-2 land independently. The ODR-sensitive `TAX_USE_*` macros move with `kernels/`→`expansion/detail/` but keep their in-header `#ifndef` definitions + order (Global Constraints).
