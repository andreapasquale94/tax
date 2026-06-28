# tax Reorg — Phase 5: la re-export consolidation + self-complete la.hpp — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `<tax/la.hpp>` self-contained, and replace the three scattered `using`-re-export blocks with a single `la/exports.hpp` assembly point that surfaces BOTH the dense `tax::la` helpers (fixing the `tax::gradient(te)` gap) and the named/mixed helpers into `tax::`, so the documented `tax::` spelling resolves uniformly.

**Architecture:** Phase 5 of the reorg (spec `docs/superpowers/specs/2026-06-28-tax-library-reorganization-design.md`, D6 **revised** to an exports.hpp assembly point — NOT inline namespaces — plus M-F self-complete). Builds on Phase 4 (branch `claude/tax-library-reorg`, at `0acd4da`). Directories stay current (the tree move is Phase 6). The NumTraits factory, traits unification, and giving mixed the point-form overloads are **deferred** (spec §8).

**Tech Stack:** Header-only C++23; Eigen3; GoogleTest; mamba `tax` env.

## Global Constraints

- C++23; `constexpr` core; no heap in dense core; graded-lex ordering sacred; kernel macros in-header (ODR); M≥1.
- Build/test (repo root, mamba `tax` env active):
  `source /Users/andrea/miniforge3/etc/profile.d/conda.sh && conda activate tax`
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
  Suite is **57 tests** entering P5; Task 2 adds one test → **58** at phase end.
- `clang-format` touched files; preserve indented `#    define` PP directives (none in these files).
- After P5: `<tax/la.hpp>` compiles standalone with full NumTraits (incl. mixed); a single `la/exports.hpp` is the only place surfacing the la/named helpers into `tax::`; `tax::gradient(te)`, `tax::gradient<"x">(ne)`, `tax::hessian`, `tax::jacobian`, `tax::eval`, `tax::variables`, `tax::invert`, `tax::value` all resolve.

**TDD note:** Task 1 is a one-line self-completion (verified by the existing suite). Task 2 adds a resolution test — write it first (RED: `tax::gradient(te)`/`tax::invert` don't resolve), then add `la/exports.hpp` (GREEN).

---

### Task 1: Self-complete la.hpp

**Files:**
- Modify: `include/tax/la.hpp` (add the missing `mixed_named.hpp` include)
- Modify: `include/tax/tax.hpp` (drop the now-redundant separate `la/mixed_named.hpp` include)

**Interfaces:**
- Produces: `<tax/la.hpp>` alone provides `Eigen::NumTraits<MixedTaylorExpansion>` + the mixed per-axis `gradient`/`hessian`/`jacobian` (previously only reachable via the full umbrella).

- [ ] **Step 1: Add mixed_named to the la facade**

In `include/tax/la.hpp`, add (keeping the includes sorted with the others):
```cpp
#include <tax/la/mixed_named.hpp>
```

- [ ] **Step 2: Drop the redundant umbrella line**

In `include/tax/tax.hpp`, delete the standalone line `#include <tax/la/mixed_named.hpp>` (it is now pulled by `<tax/la.hpp>`, which the umbrella already includes). `#pragma once` makes this safe either way, but removing it keeps the umbrella honest.

- [ ] **Step 3: Build + full suite**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
Expected: build EXIT 0; `100% tests passed … out of 57`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(printf 'fix(la): self-complete la.hpp (include mixed_named)\n\nM-F: <tax/la.hpp> alone lacked NumTraits<MixedTaylorExpansion> + the mixed\nper-axis gradient/hessian/jacobian (the umbrella patched it in separately).\nInclude mixed_named.hpp from the la facade; drop the now-redundant standalone\numbrella include.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: la/exports.hpp — one assembly point; surface dense + named helpers to tax::

**Files:**
- Create: `include/tax/la/exports.hpp`
- Modify: `include/tax/la/named.hpp` (delete the `namespace tax { using named::… }` block, lines ~299-307)
- Modify: `include/tax/la/mixed_named.hpp` (delete the `namespace tax { using named::… }` block, lines ~144-149)
- Modify: `include/tax/la/values.hpp` (delete ONLY the `using tax::la::value;` line ~123; KEEP the `tax::value` template overloads that follow)
- Modify: `include/tax/la.hpp` (include `la/exports.hpp` LAST)
- Create: `tests/eigen/test_tax_spelling.cpp` (resolution test)
- Modify: `tests/CMakeLists.txt` (register it)

**Interfaces:**
- Consumes: all the la helper definitions (`tax::la::gradient/hessian/jacobian/derivative/eval/variables/invert/value`; `tax::named::gradient/hessian/jacobian/value/eval/variables`).
- Produces: a single `la/exports.hpp` surfacing them under `tax::`; the documented `tax::FN` spelling resolves for dense, named, AND mixed expansions uniformly.

- [ ] **Step 1: Write the failing resolution test**

Create `tests/eigen/test_tax_spelling.cpp`:
```cpp
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <tax/tax.hpp>

// The documented public spelling is tax::FN — it must resolve for plain (dense)
// expansions AND named expansions uniformly (previously tax::gradient(te) did
// not resolve; only tax::gradient<"x">(ne) did).

TEST( TaxSpelling, GradientResolvesForDenseAndNamed )
{
    // Dense: f(x,y) = x*y at (0,0) on TE<3,2>; grad = [y, x] = [0,0] here, but
    // the point is that tax::gradient(te) RESOLVES (the LA-5 gap).
    typename tax::TE< 3, 2 >::Input p{ 0.0, 0.0 };
    auto fx = tax::TE< 3, 2 >::variable< 0 >( p );
    auto fy = tax::TE< 3, 2 >::variable< 1 >( p );
    auto g  = tax::gradient( fx * fy );  // <-- previously unresolved at tax::
    EXPECT_EQ( g.size(), 2 );

    // Named: ∂(x*p)/∂x via the axis-addressed form, also under tax::.
    auto nx = tax::variable< "x", 1 >( 0.0 );
    auto np = tax::variable< "p", 1 >( 0.0 );
    auto gn = tax::gradient< "x" >( nx * np, Eigen::Vector2d{ 0.3, -0.4 } );
    EXPECT_EQ( gn.size(), 1 );

    // tax::value / tax::invert also resolve at tax::.
    EXPECT_DOUBLE_EQ( tax::value( fx ), 0.0 );
    Eigen::Matrix< tax::TE< 3, 2 >, 2, 1 > F;
    F( 0 ) = fx;
    F( 1 ) = fy;
    auto Finv = tax::invert( F );  // <-- tax::invert resolves
    EXPECT_EQ( Finv.size(), 2 );
}
```
Register: `tax_add_test(test_tax_spelling SOURCES eigen/test_tax_spelling.cpp)`.
> If a helper name in this test (e.g. the exact `variable<"x",1>` factory spelling or the point-form `gradient<"x">(f, at)` signature) doesn't match the codebase, adjust the test to the real API — inspect `core/named.hpp` / `la/named.hpp` — do NOT invent signatures. The test's purpose is only to prove `tax::gradient`/`tax::invert`/`tax::value` RESOLVE for dense + named.

- [ ] **Step 2: RED — confirm tax::gradient(te) / tax::invert don't resolve yet**

Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target test_tax_spelling -j
```
Expected: FAIL — `tax::gradient(fx*fy)` and `tax::invert(F)` are unresolved (the dense la helpers live in `tax::la` only and are not surfaced to `tax::`). Record the errors. (If `tax::gradient(te)` unexpectedly already resolves, note it and keep going — the test still guards the consolidation.)

- [ ] **Step 3: Create la/exports.hpp**

Create `include/tax/la/exports.hpp`:
```cpp
#pragma once

// Single assembly point: surface the linear-algebra helpers under `tax::` so the
// documented `tax::FN(...)` spelling resolves for dense, named, and mixed
// expansions uniformly. Included LAST by <tax/la.hpp>, after every overload is
// defined, so one set of using-declarations captures the complete overload set
// (no need to re-issue per header). `tax::value`/`tax::truncate` already have
// their own tax:: definitions (la/values.hpp, la/truncate.hpp); the `value`
// using here folds in the Eigen-matrix overload.

#include <tax/la/derivatives.hpp>
#include <tax/la/invert.hpp>
#include <tax/la/mixed_named.hpp>
#include <tax/la/named.hpp>
#include <tax/la/values.hpp>

namespace tax
{
// Dense / basis-generic la helpers (these were previously NOT surfaced to tax::).
using la::derivative;
using la::eval;
using la::gradient;
using la::hessian;
using la::invert;
using la::jacobian;
using la::value;
using la::variables;

// Named + mixed per-axis helpers (mixed overloads live in tax::named too).
using named::eval;
using named::gradient;
using named::hessian;
using named::jacobian;
using named::value;
using named::variables;
}  // namespace tax
```

- [ ] **Step 4: Delete the three scattered re-export blocks**

- `include/tax/la/named.hpp`: delete the trailing `namespace tax { using named::eval; … using named::variables; }` block (≈ lines 295-307, including its comment).
- `include/tax/la/mixed_named.hpp`: delete the trailing `namespace tax { using named::gradient; using named::hessian; using named::jacobian; }` block (≈ lines 144-149, including its comment).
- `include/tax/la/values.hpp`: delete ONLY the single `using tax::la::value;` line (≈ line 123). **KEEP** the `tax::value` template overloads (the `is_te_v` and `std::is_arithmetic_v` ones) and the surrounding `namespace tax { … }` — only the re-export `using` line moves to exports.hpp.

- [ ] **Step 5: Include exports.hpp last in the la facade**

In `include/tax/la.hpp`, add as the LAST include (after all other `la/*` includes):
```cpp
#include <tax/la/exports.hpp>
```

- [ ] **Step 6: GREEN — build the test + full suite**

Run:
```bash
cmake --build build --target test_tax_spelling -j && ./build/tests/test_tax_spelling
cmake --build build -j && ctest --test-dir build -j
```
Expected: the target compiles and passes; full suite `100% tests passed … out of 58`. If the build reports an AMBIGUITY (e.g. `tax::variables` or `tax::value` ambiguous between a la and a named/core overload), that is a real collision the consolidation surfaced — STOP and report the exact ambiguity (do not silently drop a `using` that hides a needed overload).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor(la): single la/exports.hpp assembly point for the tax:: surface\n\nH-E (D6 revised to an exports.hpp assembly point, not inline namespaces):\nreplace the three scattered using-re-export blocks (named.hpp, mixed_named.hpp,\nvalues.hpp) with one la/exports.hpp, included last by la.hpp. It surfaces the\ndense la helpers too (fixes the tax::gradient(te)/tax::invert gap), so tax::FN\nresolves for dense, named, and mixed uniformly. Resolution test added.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Phase exit criteria

- `<tax/la.hpp>` is self-contained (includes mixed_named); `la/exports.hpp` is the single tax:: assembly point (the 3 scattered blocks gone); `tax::gradient`/`hessian`/`jacobian`/`eval`/`variables`/`invert`/`value` resolve for dense + named + mixed.
- `ctest` → `100% tests passed … out of 58`. Two commits landed.

## Self-review (completed)

- **Spec coverage (P5 slice):** self-complete la.hpp (M-F) → Task 1; consolidate the re-export dance + surface dense helpers (H-E, D6-revised) → Task 2. NumTraits factory / traits unification / mixed point-form explicitly deferred (spec §8). ✔
- **Placeholders:** none — exact include edits, the full exports.hpp content, the exact blocks to delete (with the values.hpp keep-the-overloads caveat), and a concrete RED/GREEN resolution test.
- **Risk note:** the only real risk is a `using`-introduced ambiguity in `tax::` (e.g. `variables`/`value` across la + named + core). Step 6 calls this out as a STOP-and-report condition rather than papering over it. No inline namespaces (per the revised D6), so the whole-surface-promotion risk is avoided.
