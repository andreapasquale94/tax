# Phase 9 — Naming Sweep + Docs Refresh (FINAL) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the public-API naming decisions (D2/D4, spec §5) as a sequence of atomic, build-verified renames — `*Series`→`*Expansion`, delete the redundant `TaylorSeries`/`NamedSeries`/`TEn`, `MixedTE`→`BoxTE`, `MixedTaylorExpansion`→`MixedExpansion` (with `mixed_named.hpp`→`mixed.hpp`) — then refresh the documentation to reflect the entire reorganization's final state.

**Architecture:** Each rename is a pure identifier/file rename with no behavior change: the type/alias it names is unchanged, only its spelling. The terse Taylor shorthands (`TE`/`STE`/`NE`/`MTE`) and the generic `Series`/`Expansion` aliases are retained per D4. The final task is a documentation pass (CLAUDE.md, README, docs/) that brings the prose in line with the post-reorg tree, namespaces, and names.

**Tech Stack:** Header-only C++23, GoogleTest, CMake, the mamba `tax` env. Build/test:
`mamba run -n tax cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure`

## Global Constraints

- **No behavior change.** Every rename is spelling-only; the full suite stays green (58 CTest targets) at each task boundary.
- **Word-boundary-exact renames (critical).** Rename ONLY the exact identifier, never a substring or a similarly-named identifier. In particular:
  - `*Series` alias renames must NOT touch the printing surface: `tax::series()` (lowercase fn), `SeriesOptions`, `SeriesStyle`, `ScalarSeriesProxy`, `MatrixSeriesProxy`, `writeSeries`, `writeOrthoSeries`, the file `io/series.hpp`, the `tests/series/` directory, or `test_series*.cpp` filenames. Also do NOT rename the retained generic `Series` alias.
  - `TEn`→delete must not touch `TE`, `STE`, `TEVec`, `NE`, etc.
  - `MixedTE`→`BoxTE` must not touch `MTE`, `MixedScheme`, `MixedTaylorExpansion`, `MixedExpansion`.
  - `MixedTaylorExpansion`→`MixedExpansion` must not touch `TaylorExpansion` or `NamedTaylorExpansion`.
- **Rename method (macOS gotcha).** BSD `sed` lacks reliable `\b`, and `perl -i` is flaky on this machine. Use **Python `re.sub(r'\bIDENT\b', 'NEW', text)`** per file for word-boundary-exact renames. Suggested pattern (stdlib only):
  ```bash
  python3 - <<'PY'
  import re, subprocess, pathlib
  OLD, NEW = r'\bChebyshevSeries\b', 'ChebyshevExpansion'
  roots = ['include', 'tests', 'docs', 'README.md', 'CLAUDE.md']
  files = subprocess.run(['grep','-rl','ChebyshevSeries',*roots],
                         capture_output=True, text=True).stdout.split()
  for f in files:
      p = pathlib.Path(f); p.write_text(re.sub(OLD, NEW, p.read_text()))
  print('updated', len(files), 'files')
  PY
  ```
  After each rename, VERIFY with `grep -rn "\bOLD\b" include tests docs README.md CLAUDE.md` → empty (no stray occurrences).
- **Retain terse + generic aliases** (D4): `TE`/`STE`/`NE`/`MTE`/`BoxTE`, generic `Series`/`Expansion`, `TaylorExpansion`/`NamedExpansion`/`ChebyshevExpansion`/`LegendreExpansion`/`HermiteExpansion`/`MixedExpansion`. Delete only `TEn`, `TaylorSeries`, `NamedSeries`.
- **One rename per commit.** Within a task, each distinct rename/delete is its own commit; the build is green at every commit.
- `clang-format -i` only touched `.hpp`/`.cpp` files (not `.md`/`.def`), preserving the indented-PP convention. Commit only; do NOT push. Append to every commit message:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_012wDzKTxBLPT1uBfmpqAsY7`

---

## File Structure (definition sites)

- `include/tax/bases/aliases.hpp` — `ChebyshevSeries`/`LegendreSeries`/`HermiteSeries` (rename), `TaylorSeries` (delete); generic `Series` retained.
- `include/tax/expansion/named.hpp:271` — `NamedSeries` (delete).
- `include/tax/expansion/expansion.hpp` — `TEn` (delete, line ~361), `MixedTE` (rename→`BoxTE`, line ~586).
- `include/tax/expansion/mixed_named.hpp` → renamed to `include/tax/expansion/mixed.hpp`; `MixedTaylorExpansion`→`MixedExpansion`; `MTE` alias retained.
- `include/tax/la/mixed_named.hpp` → renamed to `include/tax/la/mixed.hpp` (consistency).
- Usages across `include/`, `tests/`, `docs/`, `README.md`, `CLAUDE.md`.

---

### Task 1: orthogonal `*Series` → `*Expansion`

Rename the three orthogonal per-basis aliases to the `Expansion` carrier noun (D2). The generic `Series` alias and all printing identifiers are left untouched.

**Files:** `include/tax/bases/aliases.hpp` (defs) + all usages in `include/`, `tests/`, `docs/`, `README.md`, `CLAUDE.md`.

**Interfaces:** Produces `tax::ChebyshevExpansion<N,M,T>`, `tax::LegendreExpansion<N,M,T>`, `tax::HermiteExpansion<N,M,T>` (same template params, same definitions). Removes the `*Series` spellings.

- [ ] **Step 1: Baseline green.** `mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure 2>&1 | tail -5` → 58/58.
- [ ] **Step 2: Rename `ChebyshevSeries`→`ChebyshevExpansion`** across `include tests docs README.md CLAUDE.md` using the Python `re.sub(r'\bChebyshevSeries\b', 'ChebyshevExpansion', …)` pattern. Update the doc comment in `aliases.hpp` if it says "Chebyshev … expansion" — fine as is, but ensure the `using` name changed.
- [ ] **Step 3: Verify + build + commit.** `grep -rn "\bChebyshevSeries\b" include tests docs README.md CLAUDE.md` → empty. Build + `ctest` → 58/58. `git commit -m "refactor(naming): ChebyshevSeries -> ChebyshevExpansion"`.
- [ ] **Step 4: Rename `LegendreSeries`→`LegendreExpansion`** (same method). Verify grep empty, build 58/58, commit `refactor(naming): LegendreSeries -> LegendreExpansion`.
- [ ] **Step 5: Rename `HermiteSeries`→`HermiteExpansion`** (same method). Verify grep empty, build 58/58, commit `refactor(naming): HermiteSeries -> HermiteExpansion`.
- [ ] **Step 6: Guard check.** Confirm the printing surface is untouched: `grep -rn "\bseries\b\|SeriesOptions\|writeSeries\|writeOrthoSeries\|ScalarSeriesProxy" include/tax/io/series.hpp | head` still present; `git diff --name-only BASE..HEAD` does NOT include `io/series.hpp` for a *Series-alias reason (it may legitimately appear only if a doc/usage there referenced the alias — there should be none). `clang-format -i` any touched `.hpp/.cpp`.

---

### Task 2: delete redundant aliases (`TaylorSeries`, `NamedSeries`, `TEn`)

These are fully redundant with `TaylorExpansion` / `NamedExpansion` / `TE`. Delete each alias and migrate its usages to the canonical spelling.

**Files:** `include/tax/bases/aliases.hpp` (`TaylorSeries`), `include/tax/expansion/named.hpp:271` (`NamedSeries`), `include/tax/expansion/expansion.hpp:~361` (`TEn`) + usages.

**Interfaces:** Removes `tax::TaylorSeries`, `tax::NamedSeries`, `tax::TEn`. Their roles are served by existing spellings — but note the ARITY DIFFERENCE: `TaylorSeries<N,M,T>` (N,M,T order) is NOT the same arity as `TaylorExpansion<T,Scheme,Storage>` (the carrier's Taylor alias). So `TaylorSeries` usages migrate PER-SITE, not by rename:
  - explicit-`T` site `TaylorSeries<N, M, T>` → `TaylorExpansion<T, IsotropicScheme<N, M>>`.
  - `double` site `TaylorSeries<N[, M]>` → `TE<N[, M]>` (the terse double alias).
`TEn<N,M>` → `TE<N,M>` (identical arity, clean rename). `NamedSeries` is unused outside its own def/re-export (delete the lines).

- [ ] **Step 1: Migrate the `TaylorSeries` call sites.** There are exactly two kinds (verify with `grep -rn "\bTaylorSeries\b" include tests`):
  - `include/tax/bases/convert.hpp` — three sites `TaylorSeries< N, 1, T >` (generic `T`): rewrite each to `TaylorExpansion< T, IsotropicScheme< N, 1 > >`.
  - `tests/series/test_series_convert.cpp` and `tests/series/test_series_taylor.cpp` — `using tax::TaylorSeries;` + sites like `TaylorSeries< 4 >`, `TaylorSeries< 5 >`, `TaylorSeries< 8 >` (double): replace the `using` with `using tax::TE;` and each `TaylorSeries< K >` with `TE< K >`.
  Then DELETE the `using TaylorSeries` line from `include/tax/bases/aliases.hpp`. (`IsotropicScheme` is already in scope in `convert.hpp`; if not, add `#include <tax/expansion/scheme/isotropic.hpp>`.)
- [ ] **Step 2: Verify + build + commit.** `grep -rn "\bTaylorSeries\b" include tests docs README.md CLAUDE.md` → empty (reword any doc mention to `TaylorExpansion`/`TE`). Build + ctest 58/58 (`test_series_convert`, `test_series_taylor` exercise the migrated sites). Commit `refactor(naming): delete redundant TaylorSeries (use TE / TaylorExpansion)`.
- [ ] **Step 3: Delete `NamedSeries`.** Migrate any `\bNamedSeries\b` usages → `NamedExpansion` (tests=0; docs only — reword doc references), then delete the `using NamedSeries` line in `named.hpp`. Verify grep empty, build 58/58, commit `refactor(naming): delete redundant NamedSeries (use NamedExpansion)`.
- [ ] **Step 4: Delete `TEn`.** Migrate `\bTEn\b` usages → `TE` (1 test + docs). CRITICAL: anchored `\bTEn\b` only — must not touch `TE`/`STE`/`TEVec`. Then delete the `using TEn` line in `expansion.hpp`. Verify `grep -rn "\bTEn\b" include tests docs README.md CLAUDE.md` → empty, build 58/58, commit `refactor(naming): delete redundant TEn (use TE)`.

---

### Task 3: `MixedTE` → `BoxTE`

Rename the unnamed anisotropic alias (`Expansion` over `MixedScheme`) from the `MixedTE`/`MTE` name-trap to `BoxTE` (spec §5: deliberately not `MixedTaylorSeries`).

**Files:** `include/tax/expansion/expansion.hpp:~586` (def) + usages in `include/`, `tests/`, `docs/`, `README.md`, `CLAUDE.md`.

**Interfaces:** Produces `tax::BoxTE<Groups...>` (= `Expansion<double, TaylorBasis, MixedScheme<Groups...>, storage::Dense>`). Removes `tax::MixedTE`.

- [ ] **Step 1: Rename `\bMixedTE\b`→`BoxTE`** across `include tests docs README.md CLAUDE.md` (Python pattern). CRITICAL guard: anchored `\bMixedTE\b` only — must NOT touch `MTE`, `MixedScheme`, `MixedTaylorExpansion`, `MixedExpansion`.
- [ ] **Step 2: Verify + build + commit.** `grep -rn "\bMixedTE\b" include tests docs README.md CLAUDE.md` → empty; confirm `MTE`/`MixedScheme`/`MixedTaylorExpansion` counts unchanged (`grep -rc` before/after). Build + ctest 58/58. `clang-format -i` touched files. Commit `refactor(naming): MixedTE -> BoxTE (remove MTE name-trap)`.

---

### Task 4: `MixedTaylorExpansion` → `MixedExpansion`; `mixed_named.hpp` → `mixed.hpp`

The largest rename: the named mixed-order type and its files. `MTE` is retained as the terse alias (now `= MixedExpansion<double, Axes...>`).

**Files:**
- `include/tax/expansion/mixed_named.hpp` → `git mv` to `include/tax/expansion/mixed.hpp`.
- `include/tax/la/mixed_named.hpp` → `git mv` to `include/tax/la/mixed.hpp` (consistency).
- `MixedTaylorExpansion`→`MixedExpansion` across `include/`, `tests/`, `docs/`, `README.md`, `CLAUDE.md`.
- Repoint every `#include <tax/expansion/mixed_named.hpp>` → `<tax/expansion/mixed.hpp>` and `<tax/la/mixed_named.hpp>` → `<tax/la/mixed.hpp>`.

**Interfaces:** Produces `tax::mixed::MixedExpansion<T, Axes...>` (surfaced as `tax::MixedExpansion`), `tax::MTE<Axes...>` retained. Removes `MixedTaylorExpansion`.

- [ ] **Step 1: `git mv` the two files.** `git mv include/tax/expansion/mixed_named.hpp include/tax/expansion/mixed.hpp` and `git mv include/tax/la/mixed_named.hpp include/tax/la/mixed.hpp`.
- [ ] **Step 2: Repoint includes.** Rename the include paths `tax/expansion/mixed_named.hpp`→`tax/expansion/mixed.hpp` and `tax/la/mixed_named.hpp`→`tax/la/mixed.hpp` across `include/ tests/` (Python `re.sub` on the literal path strings — these are not word-boundary-sensitive; a plain `.replace` is fine). Update any header-comment self-reference at the top of the two moved files.
- [ ] **Step 3: Rename the type `\bMixedTaylorExpansion\b`→`MixedExpansion`** across `include tests docs README.md CLAUDE.md` (Python `\b` pattern). CRITICAL guard: must NOT touch `TaylorExpansion` or `NamedTaylorExpansion` (the `\bMixedTaylorExpansion\b` anchor handles this since the match must start at `Mixed`). Confirm the `MTE` alias body becomes `MixedExpansion<double, Axes...>`.
- [ ] **Step 4: Verify.** `grep -rn "\bMixedTaylorExpansion\b" include tests docs README.md CLAUDE.md` → empty. `grep -rn "mixed_named.hpp" include tests` → empty (all includes repointed; a historical mention in `docs/superpowers/` plan/spec files is acceptable — those are dated records, do not rewrite them). Confirm `TaylorExpansion`/`NamedTaylorExpansion` counts unchanged.
- [ ] **Step 5: Build + test + commit.** `mamba run -n tax cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure 2>&1 | tail -8` → clean, 58/58 (the mixed suites `test_mixed_named`, `test_mixed_named_la`, `test_expansion_vectors` exercise the renamed type). `clang-format -i` touched `.hpp/.cpp`. Commit `refactor(naming): MixedTaylorExpansion -> MixedExpansion; mixed_named.hpp -> mixed.hpp`.

---

### Task 5: holistic documentation refresh

Bring the prose docs in line with the final post-reorg state (tree, namespaces, names). This is the deferred docs pass — CLAUDE.md's "Repository Structure" has been stale since the P6 tree move, and several names changed across P7–P9.

**Files:** `CLAUDE.md` (Repository Structure + any stale paths/names), `README.md` (architecture/usage snippets), `docs/` prose that describes structure/names (NOT the dated `docs/superpowers/specs|plans/*` records — those are historical and stay as-written).

**Interfaces:** None (docs only). No code or test change.

- [ ] **Step 1: Inventory the final state.** Capture the current tree and names as ground truth:
  - `find include/tax -type f | sort` (the real header layout: `expansion/` with `detail/`, `ops/`, `scheme/`, `storage/`; `bases/`; `la/`; `io/`; facades `expansion.hpp`, `bases.hpp`, `la.hpp`, `tax.hpp`).
  - The final namespaces: `tax`, `tax::detail::kernels`, `tax::named`, `tax::mixed`, `tax::la`.
  - The final public names: carrier `Expansion`; Taylor `TaylorExpansion`/`TE`/`STE`; named `NamedExpansion`/`NE`; mixed `MixedExpansion`/`MTE`; box `BoxTE`; orthogonal `ChebyshevExpansion`/`LegendreExpansion`/`HermiteExpansion`; generic `Series`. (No `TEn`, `TaylorSeries`, `NamedSeries`, `MixedTE`, `MixedTaylorExpansion`, `Batch`.)
- [ ] **Step 2: Rewrite CLAUDE.md "Repository Structure".** Replace the stale tree (which still shows `core/`, `kernels/`, `operators/`, `series/`, `ode/`, `ads/`, `Batch`) with the real `find` output above, and update the prose bullets describing each directory's role (kernels are now `expansion/detail/`, operators `expansion/ops/`, orthogonal bases `bases/`, etc.). Update the "Main Type" / aliases / "Mixed-order" sections to the final names (`MixedExpansion`, `BoxTE`; drop `TEn`/`MixedTE`/`*Series`/`Batch`).
- [ ] **Step 3: Refresh README.md.** Update any directory tree, type names, and code snippets to the final names/paths. Verify any `#include`/alias in a README example compiles conceptually against the final API (e.g. `tax::ChebyshevExpansion`, `tax::MixedExpansion`/`MTE`).
- [ ] **Step 4: Sweep remaining docs prose.** `grep -rn "core/\|kernels/\|operators/\|\bseries/\|MixedTaylorExpansion\|MixedTE\b\|TEn\b\|ChebyshevSeries\|TaylorSeries\|NamedSeries\|Batch" docs/ | grep -v "docs/superpowers/"` and fix the non-historical prose hits (guide/reference/concepts/internals). Leave `docs/superpowers/specs|plans/*` untouched (dated records).
- [ ] **Step 5: Verify docs build (if applicable) + commit.** If MkDocs config builds locally, optionally sanity-check; otherwise just confirm no broken internal links were introduced by name changes. No `ctest` needed (docs-only), but DO run a final `mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure 2>&1 | tail -5` to confirm the tree is still 58/58 (sanity). Commit `docs: refresh CLAUDE.md/README/docs for the final reorg structure and names`.

---

## Self-Review Notes (controller)

- **Spec coverage (§5 / D2 / D4 / P9):** orthogonal `*Series`→`*Expansion` (T1); delete `TaylorSeries`/`NamedSeries`/`TEn` (T2); `MixedTE`→`BoxTE` (T3); `MixedTaylorExpansion`→`MixedExpansion` + `mixed_named.hpp`→`mixed.hpp` (T4); docs refresh incl. the deferred CLAUDE.md Repository Structure (T5). The `Series` arity fix (§5 line 152) is ALREADY DONE (aliases already take `M=1`) — no task needed. `Batch`/`Batchd`/`Batchf` and the `TE` K-param were already removed (P0) — not repeated.
- **Retained names (D4):** `TE`/`STE`/`NE`/`MTE`/`BoxTE`, generic `Series`/`Expansion`, `TaylorExpansion`/`NamedExpansion`/`MixedExpansion`/`ChebyshevExpansion`/`LegendreExpansion`/`HermiteExpansion`. Verify each still resolves after the sweep (the build + a `tax::` spelling test cover this).
- **Rename safety:** word-boundary-exact (Python `\b`), per-identifier guards against look-alikes, grep-empty verification after each, build green per commit. The printing `series()` surface and the historical `docs/superpowers/` records are explicitly out of scope.
- **Companion `ode/ads` plugin** (separate repo) will need the same `*Series→*Expansion` / `MixedTaylorExpansion→MixedExpansion` migration — out of scope here; note in the final report so the maintainer can propagate.
