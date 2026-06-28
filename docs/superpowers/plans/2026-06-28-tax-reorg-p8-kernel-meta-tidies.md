# Phase 8 — Kernel/Meta Tidies (F8 remainder) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the low-risk kernel/meta cleanups from F8: give the `TAX_USE_*` dispatch macros a single home (`stencil_config.hpp`), delete two dead `<N,M>` forwarding shims, consolidate the scattered sparse operators into one `ops/sparse.hpp`, and centralize the duplicated unary-function list into one `unary_functions.def` x-macro.

**Architecture:** All four items are organization/DRY cleanups with **no behavior change**. The macros currently live (and are partially duplicated) in `expansion/detail/cauchy.hpp` and `cauchy_stencil.hpp`; the sparse `TaylorExpansion` operators are scattered across three dense `ops/*.hpp` files; and the same 20-function unary list is hand-copied into four headers. Each task removes one source of duplication while keeping every kernel definition, overload, and output identical.

**Tech Stack:** Header-only C++23, GoogleTest, CMake, the mamba `tax` env (conda clang++ + Eigen). Build/test:
`mamba run -n tax cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure`

## Global Constraints

- **No behavior change.** Every kernel, operator overload, recurrence, and printed/computed result stays identical. These are moves, deletions of dead code, and macro-factoring only. The full suite must stay green (currently 58 CTest targets pass) at every task boundary.
- **ODR macro discipline (critical).** `TAX_USE_UNROLL`, `TAX_USE_STENCIL`, `TAX_STENCIL_MAX_BYTES` must have exactly ONE definition site after this phase (`stencil_config.hpp`), each guarded by `#ifndef` (a project may pre-define them; the value must be identical project-wide). `stencil_config.hpp` must be `#include`d **before** any `#if TAX_USE_*` / `#if ... TAX_STENCIL_MAX_BYTES` that depends on it. Do not reintroduce these macros as build-system definitions.
- **Graded-lex ordering sacred; constexpr/noexcept preserved; no heap in the dense core** (sparse storage may use `std::vector`, unchanged).
- **ADL preserved.** All `tax::` operators stay in `namespace tax`; moving them between headers within `namespace tax` must not change which overload resolves at any call site.
- **`clang-format` touched files only**, preserving the repo's indented-preprocessor convention (e.g. `#    define`). Do not mass-reformat.
- Commit only; do NOT push. Append these trailers to every commit message:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_012wDzKTxBLPT1uBfmpqAsY7`

---

## File Structure

- `include/tax/expansion/detail/stencil_config.hpp` — **new.** Single home for the three `TAX_USE_*` macros (Task 1).
- `include/tax/expansion/detail/cauchy.hpp` — drops the inline macro block; includes `stencil_config.hpp` (Task 1).
- `include/tax/expansion/detail/cauchy_stencil.hpp` — drops the DUPLICATE `TAX_STENCIL_MAX_BYTES` fallback; includes `stencil_config.hpp`; the wrong "the two headers include each other" comment removed/corrected (Task 1).
- `include/tax/expansion/detail/recurrence_stencil.hpp`, `mixed_stencils.hpp`, `include/tax/expansion/scheme/mixed.hpp` — include `stencil_config.hpp` directly (Task 1).
- `include/tax/expansion/detail/algebra.hpp` — delete the two dead `<N,M>` shims (Task 1).
- `include/tax/expansion/ops/sparse.hpp` — **new.** All sparse `TaylorExpansion` operators (Task 2).
- `include/tax/expansion/ops/arithmetic.hpp`, `math_unary.hpp`, `math_binary.hpp` — drop their sparse blocks + now-unused sparse includes (Task 2).
- `include/tax/expansion.hpp` — wire in `ops/sparse.hpp` (Task 2).
- `include/tax/expansion/ops/unary_functions.def` — **new.** The single unary-function x-macro list (Task 3).
- `include/tax/expansion/ops/math_unary.hpp`, `named_math_unary.hpp`, `mixed_math_unary.hpp` — drive their lists from the `.def` (Task 3).

---

### Task 1: `stencil_config.hpp` + comment fix + dead-shim deletion

**Files:**
- Create: `include/tax/expansion/detail/stencil_config.hpp`
- Modify: `include/tax/expansion/detail/cauchy.hpp`, `cauchy_stencil.hpp`, `recurrence_stencil.hpp`, `mixed_stencils.hpp`
- Modify: `include/tax/expansion/scheme/mixed.hpp`
- Modify: `include/tax/expansion/detail/algebra.hpp` (delete shims)

**Interfaces:**
- Produces: macros `TAX_USE_UNROLL`, `TAX_USE_STENCIL`, `TAX_STENCIL_MAX_BYTES` from `stencil_config.hpp` (same values as today).
- Removes: `tax::detail::kernels::cauchySelfProduct<T,int N,int M>` and `seriesReciprocal<T,int N,int M>` (the `<N,M>` int-form shims — the `<T,Scheme>` forms remain).

- [ ] **Step 1: Baseline green**

Run: `mamba run -n tax cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure 2>&1 | tail -5`
Expected: clean build; all tests pass (58 CTest targets).

- [ ] **Step 2: Create `stencil_config.hpp`**

Create `include/tax/expansion/detail/stencil_config.hpp` with exactly the macro block currently in `cauchy.hpp` (moved verbatim, including its explanatory comments):

```cpp
#pragma once

#include <cstddef>

// Kernel dispatch configuration. Defaults ON here (not in the build system)
// so every consumer gets the fast paths regardless of how the headers are
// consumed. A project may pre-define either macro to 0 to fall back to the
// loop kernel, but the value MUST be identical in every translation unit
// linked together — differing values change inline definitions (ODR).
#ifndef TAX_USE_UNROLL
#    define TAX_USE_UNROLL 1
#endif
#ifndef TAX_USE_STENCIL
#    define TAX_USE_STENCIL 1
#endif
// Upper bound (bytes) on any precomputed stencil table (Cauchy or recurrence).
// Beyond this the dispatch falls back to the constexpr loop kernel instead of
// materialising a huge static table — which would otherwise be a hard compile
// error for many-variable, high-order expansions. Configured in-header like the
// other knobs; a project may pre-define it, but the value must be identical in
// every translation unit (ODR).
#ifndef TAX_STENCIL_MAX_BYTES
#    define TAX_STENCIL_MAX_BYTES ( static_cast< std::size_t >( 64 ) << 20 )
#endif
```

- [ ] **Step 3: Point `cauchy.hpp` at the new config header**

In `include/tax/expansion/detail/cauchy.hpp`, DELETE the inline macro block (the comment + the three `#ifndef/#define/#endif` groups — currently lines 7–26) and instead add, in the include section (before the `#if TAX_USE_UNROLL` include block):

```cpp
#include <tax/expansion/detail/stencil_config.hpp>
```

The `#if TAX_USE_UNROLL … #endif` / `#if TAX_USE_STENCIL … #endif` include blocks stay exactly as they are (now resolved using the macros from `stencil_config.hpp`).

- [ ] **Step 4: De-duplicate `cauchy_stencil.hpp` and fix the wrong comment**

In `include/tax/expansion/detail/cauchy_stencil.hpp`:
- Add `#include <tax/expansion/detail/stencil_config.hpp>` to its includes.
- DELETE its local duplicate fallback (currently lines 13–14):
  ```cpp
  #ifndef TAX_STENCIL_MAX_BYTES
  #    define TAX_STENCIL_MAX_BYTES ( static_cast< std::size_t >( 64 ) << 20 )
  #endif
  ```
  (Keep any surrounding `#endif` that belongs to a different guard — only remove this `TAX_STENCIL_MAX_BYTES` `#ifndef/#define/#endif` triple.)
- Fix the factually-wrong comment at line 11. It currently reads (approximately):
  `// regardless of include order (the two headers include each other). A project`
  Replace the parenthetical so it no longer claims the headers include each other, e.g.:
  `// regardless of include order (the config macros come from stencil_config.hpp). A project`
  (Preserve the rest of the sentence/comment.)

- [ ] **Step 5: Make the other macro consumers include the config directly**

So none of them depends on getting the macros transitively through `cauchy.hpp`, add `#include <tax/expansion/detail/stencil_config.hpp>` to each of these (near the top of their include list, before any `#if TAX_USE_*` use):
- `include/tax/expansion/detail/recurrence_stencil.hpp` (it currently includes `cauchy.hpp` "// TAX_USE_STENCIL configuration"; you may keep that include or replace its comment, but ensure `stencil_config.hpp` is included).
- `include/tax/expansion/detail/mixed_stencils.hpp` (uses `TAX_STENCIL_MAX_BYTES`).
- `include/tax/expansion/scheme/mixed.hpp` (uses `TAX_USE_STENCIL`).

- [ ] **Step 6: Build to verify the macro move (no ODR / undefined-macro errors)**

Run: `mamba run -n tax cmake --build build -j 2>&1 | tail -20`
Expected: clean build. (A missing include would surface as `TAX_USE_STENCIL`/`TAX_STENCIL_MAX_BYTES` undefined or the wrong dispatch.)

- [ ] **Step 7: Delete the two dead `<N,M>` shims in `algebra.hpp`**

In `include/tax/expansion/detail/algebra.hpp`, delete these two function templates (the int-form `<T, int N, int M>` shims that merely forward to the `<T, Scheme>` form). Delete each function in full, with its doc comment:

```cpp
template < typename T, int N, int M >
constexpr void cauchySelfProduct( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& f ) noexcept
{
    tax::cauchySelfProduct< T, tax::IsotropicScheme< N, M > >( out, f );
}
```

```cpp
template < typename T, int N, int M >
constexpr void seriesReciprocal( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    seriesReciprocal< T, tax::IsotropicScheme< N, M > >( out, a );
}
```

Leave the `<T, Scheme>` forms (and everything else in the file) intact.

- [ ] **Step 8: Build + full suite (build confirms the shims were truly dead)**

Run: `mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure 2>&1 | tail -8`
Expected: clean build (if any call site bound to the deleted `<N,M>` form, the build would now fail — it must not); all tests pass. If the build fails with an unresolved `cauchySelfProduct`/`seriesReciprocal`, the shim was NOT dead — STOP and report BLOCKED with the call site.

- [ ] **Step 9: Verify single macro-definition home**

Run: `grep -rn "define TAX_USE_UNROLL\|define TAX_USE_STENCIL\|define TAX_STENCIL_MAX_BYTES" include/`
Expected: exactly three hits, all in `include/tax/expansion/detail/stencil_config.hpp`.

- [ ] **Step 10: clang-format and commit**

```bash
clang-format -i include/tax/expansion/detail/stencil_config.hpp \
  include/tax/expansion/detail/cauchy.hpp include/tax/expansion/detail/cauchy_stencil.hpp \
  include/tax/expansion/detail/recurrence_stencil.hpp include/tax/expansion/detail/mixed_stencils.hpp \
  include/tax/expansion/scheme/mixed.hpp include/tax/expansion/detail/algebra.hpp
git add include/tax/expansion/detail/stencil_config.hpp include/tax/expansion/detail/cauchy.hpp \
  include/tax/expansion/detail/cauchy_stencil.hpp include/tax/expansion/detail/recurrence_stencil.hpp \
  include/tax/expansion/detail/mixed_stencils.hpp include/tax/expansion/scheme/mixed.hpp \
  include/tax/expansion/detail/algebra.hpp
git commit -m "refactor(detail): single stencil_config.hpp for TAX_USE_* macros; drop dead shims"
```

---

### Task 2: consolidate sparse operators into `ops/sparse.hpp`

The sparse `TaylorExpansion` operators are scattered through three dense operator headers. Move them verbatim into one `ops/sparse.hpp`, drop the now-unused sparse includes from the dense headers, and wire the new file into the umbrella. Pure relocation — the operators, their signatures, and `namespace tax` are unchanged, so ADL is preserved.

**Files:**
- Create: `include/tax/expansion/ops/sparse.hpp`
- Modify: `include/tax/expansion/ops/arithmetic.hpp` (remove the sparse block + sparse includes)
- Modify: `include/tax/expansion/ops/math_unary.hpp` (remove sparse `sqrt`/`reciprocal` + sparse include)
- Modify: `include/tax/expansion/ops/math_binary.hpp` (remove sparse `pow` + sparse include)
- Modify: `include/tax/expansion.hpp` (include `ops/sparse.hpp`)

**Interfaces:**
- Moves (verbatim, all in `namespace tax`): from `arithmetic.hpp` the entire "Sparse arithmetic" block (`using Sparse = storage::Sparse;` + the operators S+S, S−S, −S, S+scalar, scalar+S, S−scalar, scalar−S, S\*scalar, scalar\*S, S/scalar, S\*S, S/S — currently lines ~240–423); from `math_unary.hpp` the sparse `sqrt` and `reciprocal` overloads (~lines 80–101); from `math_binary.hpp` the sparse `pow` overload (~lines 79–86).
- `ops/sparse.hpp` consumes: `sparse_cauchy.hpp`, `sparse_subs.hpp`, the `Expansion`/`TaylorExpansion` type, `IsotropicScheme`, `storage::Sparse`.

- [ ] **Step 1: Create `ops/sparse.hpp` with the needed includes and namespace**

Create `include/tax/expansion/ops/sparse.hpp`:

```cpp
#pragma once

// All sparse-storage TaylorExpansion operators (arithmetic + the sparse math
// overloads sqrt / reciprocal / pow). Consolidated here so the dense operator
// headers carry only dense overloads. Pure relocation — every overload is
// unchanged and stays in namespace tax, preserving ADL.

#include <cstddef>
#include <tax/expansion/detail/sparse_cauchy.hpp>
#include <tax/expansion/detail/sparse_subs.hpp>
#include <tax/expansion/expansion.hpp>
#include <tax/expansion/scheme/isotropic.hpp>
#include <tax/expansion/storage/sparse.hpp>

namespace tax
{

// (moved blocks go here)

}  // namespace tax
```

- [ ] **Step 2: Move the sparse arithmetic block from `arithmetic.hpp`**

Cut from `include/tax/expansion/ops/arithmetic.hpp` the entire sparse section — the `// Sparse arithmetic: …` banner comment, the `using Sparse = storage::Sparse;` alias, and every sparse operator through the end of that block (the `Sparse / Sparse` operator, currently ending around line 423) — and paste it verbatim inside the `namespace tax { … }` of `ops/sparse.hpp`. Do not alter any operator body. Then in `arithmetic.hpp` remove the now-unused includes `#include <tax/expansion/detail/sparse_cauchy.hpp>` and `#include <tax/expansion/detail/sparse_subs.hpp>` (verify nothing dense in `arithmetic.hpp` still references a `*Sparse` kernel before removing — if something does, keep that include).

- [ ] **Step 3: Move the sparse `sqrt`/`reciprocal` from `math_unary.hpp`**

Cut from `include/tax/expansion/ops/math_unary.hpp` the two sparse overloads under the `// Sparse overloads: sqrt, reciprocal` comment (the `sqrt(const …Sparse&)` and `reciprocal(const …Sparse&)` functions, ~lines 80–101) and paste them verbatim into `ops/sparse.hpp`'s `namespace tax`. Remove `#include <tax/expansion/detail/sparse_subs.hpp>` from `math_unary.hpp` (the remaining dense overloads use `algebra.hpp`/`transcendental.hpp` kernels, not the sparse subs — verify before removing).

- [ ] **Step 4: Move the sparse `pow` from `math_binary.hpp`**

Cut from `include/tax/expansion/ops/math_binary.hpp` the sparse `pow(const …Sparse&, int)` overload (under `// Sparse `f^n` …`, ~lines 79–86) and paste it verbatim into `ops/sparse.hpp`. Remove `#include <tax/expansion/detail/sparse_subs.hpp>` from `math_binary.hpp` if nothing else there needs it (verify).

- [ ] **Step 5: Wire `ops/sparse.hpp` into the umbrella facade**

In `include/tax/expansion.hpp`, add after the dense math line (`#include <tax/expansion/ops/math_binary.hpp>`, line 22):

```cpp
#include <tax/expansion/ops/sparse.hpp>
```

- [ ] **Step 6: Build + full suite**

Run: `mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure 2>&1 | tail -10`
Expected: clean build; all tests pass — including the sparse suite (`mamba run -n tax ctest --test-dir build -R sparse --output-on-failure` should list the sparse tests all green). Sparse operators must still resolve by ADL exactly as before.

- [ ] **Step 7: Verify the dense headers no longer carry sparse operators**

Run: `grep -rn "storage::Sparse\|, Sparse >" include/tax/expansion/ops/arithmetic.hpp include/tax/expansion/ops/math_unary.hpp include/tax/expansion/ops/math_binary.hpp`
Expected: no sparse operator definitions remain in those three files (a stray `using`/comment is acceptable only if it is not an operator). All sparse operators now live in `ops/sparse.hpp`.

- [ ] **Step 8: clang-format and commit**

```bash
clang-format -i include/tax/expansion/ops/sparse.hpp include/tax/expansion/ops/arithmetic.hpp \
  include/tax/expansion/ops/math_unary.hpp include/tax/expansion/ops/math_binary.hpp include/tax/expansion.hpp
git add include/tax/expansion/ops/sparse.hpp include/tax/expansion/ops/arithmetic.hpp \
  include/tax/expansion/ops/math_unary.hpp include/tax/expansion/ops/math_binary.hpp include/tax/expansion.hpp
git commit -m "refactor(ops): consolidate sparse operators into ops/sparse.hpp"
```

---

### Task 3: central `unary_functions.def`

The identical 20-function unary list is hand-copied into `math_unary.hpp` (dense, with a constexpr/runtime split), `named_math_unary.hpp` (the forwarding list AND the re-export list), and `mixed_math_unary.hpp`. Replace all four copies with one x-macro file. The re-export list is exactly the 17 "RT" (runtime) functions, so it too is driven by the `.def` (the 3 "CE" ops — `square`/`cube`/`reciprocal` — have no `std` analogue and are re-exported separately, unchanged).

**Files:**
- Create: `include/tax/expansion/ops/unary_functions.def`
- Modify: `include/tax/expansion/ops/math_unary.hpp`
- Modify: `include/tax/expansion/ops/named_math_unary.hpp`
- Modify: `include/tax/expansion/ops/mixed_math_unary.hpp`

**Interfaces:**
- Produces: `unary_functions.def`, an include-time x-macro list. Consumers `#define TAX_UNARY_CE(NAME,KERNEL)` and `TAX_UNARY_RT(NAME,KERNEL)`, `#include` the `.def`, then `#undef` both.
- The set of generated `tax::` unary functions and their definitions are byte-for-byte unchanged.

- [ ] **Step 1: Create the `.def` (preserve `math_unary.hpp`'s exact order and CE/RT split)**

Create `include/tax/expansion/ops/unary_functions.def`. Note: this file is intentionally guard-free (it is `#include`d multiple times) and must NOT have `#pragma once`:

```cpp
// X-macro list of the unary math functions, single source of truth for every
// consumer (dense ops, named, mixed, and the named re-exports). Include this
// after defining TAX_UNARY_CE / TAX_UNARY_RT; #undef them afterwards.
//
//   TAX_UNARY_CE(name, kernel) — pure-recurrence ops (constexpr): no std analogue
//   TAX_UNARY_RT(name, kernel) — runtime ops: the kernel evaluates std::fn at the point
//
// (No include guard / #pragma once on purpose — this file is re-included.)

TAX_UNARY_CE( square, seriesSquare )
TAX_UNARY_CE( cube, seriesCube )
TAX_UNARY_CE( reciprocal, seriesReciprocal )

TAX_UNARY_RT( sqrt, seriesSqrt )
TAX_UNARY_RT( cbrt, seriesCbrt )
TAX_UNARY_RT( exp, seriesExp )
TAX_UNARY_RT( log, seriesLog )
TAX_UNARY_RT( sinh, seriesSinh )
TAX_UNARY_RT( cosh, seriesCosh )
TAX_UNARY_RT( tanh, seriesTanh )
TAX_UNARY_RT( asinh, seriesAsinh )
TAX_UNARY_RT( acosh, seriesAcosh )
TAX_UNARY_RT( atanh, seriesAtanh )
TAX_UNARY_RT( erf, seriesErf )
TAX_UNARY_RT( sin, seriesSin )
TAX_UNARY_RT( cos, seriesCos )
TAX_UNARY_RT( tan, seriesTan )
TAX_UNARY_RT( asin, seriesAsin )
TAX_UNARY_RT( acos, seriesAcos )
TAX_UNARY_RT( atan, seriesAtan )
```

- [ ] **Step 2: Drive `math_unary.hpp` from the `.def`**

In `include/tax/expansion/ops/math_unary.hpp`, KEEP the `TAX_UNARY_OP_CE` and `TAX_UNARY_OP` macro definitions. Replace the 20 explicit `TAX_UNARY_OP_CE(...)`/`TAX_UNARY_OP(...)` invocation lines (currently ~45–74) with:

```cpp
#define TAX_UNARY_CE( NAME, KERNEL ) TAX_UNARY_OP_CE( NAME, KERNEL )
#define TAX_UNARY_RT( NAME, KERNEL ) TAX_UNARY_OP( NAME, KERNEL )
#include <tax/expansion/ops/unary_functions.def>
#undef TAX_UNARY_CE
#undef TAX_UNARY_RT
```

Keep the existing `#undef TAX_UNARY_OP` / `#undef TAX_UNARY_OP_CE` that follow.

- [ ] **Step 3: Drive `named_math_unary.hpp`'s forwarding list AND re-export list from the `.def`**

In `include/tax/expansion/ops/named_math_unary.hpp`:
- Replace the 20 explicit `TAX_NAMED_UNARY_FN(...)` lines with (keeping the `TAX_NAMED_UNARY_FN` macro definition above):

```cpp
#define TAX_UNARY_CE( NAME, KERNEL ) TAX_NAMED_UNARY_FN( NAME )
#define TAX_UNARY_RT( NAME, KERNEL ) TAX_NAMED_UNARY_FN( NAME )
#include <tax/expansion/ops/unary_functions.def>
#undef TAX_UNARY_CE
#undef TAX_UNARY_RT
```

- Replace the 17 explicit `TAX_REEXPORT_UNARY(...)` lines (the re-export list — exactly the RT functions) with (keeping the `TAX_REEXPORT_UNARY` macro definition above):

```cpp
#define TAX_UNARY_CE( NAME, KERNEL )  // square/cube/reciprocal: no std analogue
#define TAX_UNARY_RT( NAME, KERNEL ) TAX_REEXPORT_UNARY( NAME )
#include <tax/expansion/ops/unary_functions.def>
#undef TAX_UNARY_CE
#undef TAX_UNARY_RT
```

Leave the trailing `using named::cube;` (and any sibling `using named::square;`/`using named::reciprocal;`) re-exports exactly as they are.

- [ ] **Step 4: Drive `mixed_math_unary.hpp` from the `.def`**

In `include/tax/expansion/ops/mixed_math_unary.hpp`, replace the 20 explicit `TAX_MIXED_UNARY_FN(...)` lines with (keeping the `TAX_MIXED_UNARY_FN` macro definition above):

```cpp
#define TAX_UNARY_CE( NAME, KERNEL ) TAX_MIXED_UNARY_FN( NAME )
#define TAX_UNARY_RT( NAME, KERNEL ) TAX_MIXED_UNARY_FN( NAME )
#include <tax/expansion/ops/unary_functions.def>
#undef TAX_UNARY_CE
#undef TAX_UNARY_RT
```

- [ ] **Step 5: Build + full suite**

Run: `mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure 2>&1 | tail -10`
Expected: clean build; all tests pass. The operator suites (`mamba run -n tax ctest --test-dir build -R "unary|trig|transcendental|named|mixed" --output-on-failure`) exercise every function in the list across dense/named/mixed; all must stay green. If any function fails to resolve, an entry was dropped/renamed in the `.def` — fix the `.def`, not the test.

- [ ] **Step 6: Verify the duplication is gone**

Run: `grep -c "TAX_NAMED_UNARY_FN(\|TAX_MIXED_UNARY_FN(\|TAX_UNARY_OP_CE(\|TAX_UNARY_OP(\|TAX_REEXPORT_UNARY(" include/tax/expansion/ops/math_unary.hpp include/tax/expansion/ops/named_math_unary.hpp include/tax/expansion/ops/mixed_math_unary.hpp`
Expected: each file shows only its macro *definition* line(s), not the ~20 invocation lines (the invocations now come from the single `.def`). The 20-name list exists once, in `unary_functions.def`.

- [ ] **Step 7: clang-format and commit**

```bash
clang-format -i include/tax/expansion/ops/math_unary.hpp include/tax/expansion/ops/named_math_unary.hpp \
  include/tax/expansion/ops/mixed_math_unary.hpp
git add include/tax/expansion/ops/unary_functions.def include/tax/expansion/ops/math_unary.hpp \
  include/tax/expansion/ops/named_math_unary.hpp include/tax/expansion/ops/mixed_math_unary.hpp
git commit -m "refactor(ops): central unary_functions.def x-macro list"
```

Note: `clang-format` may not format `.def` files (no recognized extension) — that is fine; leave `unary_functions.def` unformatted.

---

## Self-Review Notes (controller)

- **Spec coverage (F8 remainder, P8):** `stencil_config.hpp` extract + wrong-comment fix (T1); dead `<N,M>` shim deletion (T1); `ops/sparse.hpp` consolidation (T2); central `unary_functions.def` (T3). The `Merge`/`MergeOrdered` collapse and the `AxisCarrier`-drop parts of F8 were already done in P3 — not repeated here.
- **`chebyshev_math.hpp` is intentionally NOT folded into the `.def`:** it uses a different macro shape (`TAX_CHEB_UNARY(NAME, EXPR)`) over a different subset (the 17 transcendentals, no `square`/`cube`/`reciprocal`, evaluated through `std::fn(v)`). Forcing it into the shared `.def` would add conditional complexity for no real DRY win; left as-is.
- **ODR safety (T1):** after T1 there is exactly one definition home for each `TAX_USE_*` macro (verified by grep in Step 9), each `#ifndef`-guarded; every consumer includes `stencil_config.hpp` before use; the duplicate `TAX_STENCIL_MAX_BYTES` fallback in `cauchy_stencil.hpp` is removed.
- **Dead-shim safety (T1):** deletion is build-verified — if a caller bound to the `<N,M>` form existed, Step 8 fails and the task reports BLOCKED rather than guessing.
- **Behavior preservation:** every task is a move / dead-code deletion / macro-factoring; no kernel body, operator, or output changes. Full suite green at each boundary.
