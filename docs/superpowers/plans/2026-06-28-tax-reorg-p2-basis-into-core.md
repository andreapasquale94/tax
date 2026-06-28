# tax Reorg — Phase 2: BasisPolicy + TaylorBasis into core; rename concept; Series arity — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the `Basis` policy concept (renamed `BasisPolicy`) and the default `TaylorBasis` policy out of `series/` into `core/`, so the carrier `core/expansion.hpp` no longer depends on the `series/` module; and regularize the `Series<>` alias arity.

**Architecture:** Phase 2 of the reorg (spec `docs/superpowers/specs/2026-06-28-tax-library-reorganization-design.md`, F3 + the naming items). Builds on Phase 1 (branch `claude/tax-library-reorg`, at `c03d78a`). Still using the current directory names (`core/`, `series/`, …); the directory-tree move to `expansion/`/`bases/` is Phase 6. After P2 the `core→series` include edge is gone (the carrier needs only the `BasisPolicy` concept + `TaylorBasis`, both now in `core/`); orthogonal bases stay in `series/` and include `core/basis.hpp`.

**Tech Stack:** Header-only C++23; Eigen3; GoogleTest; mamba `tax` env.

## Global Constraints

- C++23; `constexpr` everywhere in the dense core; no heap in the dense core; graded-lex ordering sacred; kernel config macros in-header (ODR); `M ≥ 1`.
- Build/test (repo root, mamba `tax` env active):
  `source /Users/andrea/miniforge3/etc/profile.d/conda.sh && conda activate tax`
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
  The full suite is **56 tests**, 100% passing, through every task (no count change — this phase moves/renames, it doesn't add tests).
- `clang-format` touched files but preserve indented `#    define` PP directives (clang-format 21 de-indents; none expected in P2 files).
- After P2: the **concept** is spelled `BasisPolicy` everywhere (the type names `TaylorBasis`/`ChebyshevBasis`/`LegendreBasis`/`HermiteBasis`/`OrthogonalBasis`/`ChebyshevBasisOn`, the `typename Basis` template parameters, the `using basis = Basis` member alias, and `is_tax_basis`/`NonTaylorBasis`/`ChebyshevLike` concept *names* are UNCHANGED — only the bare `Basis` concept is renamed). `core/basis.hpp` and `core/taylor_basis.hpp` exist; `series/basis.hpp`/`series/taylor_basis.hpp` are gone (no back-compat shim — pre-1.0). `core/expansion.hpp` includes no `tax/series/*` header.

**TDD note:** all three tasks are mechanical refactors; the 56-test suite is the regression net. Each task's verification is build + `ctest` green plus targeted greps proving the rename/move is complete and didn't touch the wrong tokens.

---

### Task 1: Rename the `Basis` concept → `BasisPolicy` (in place, repo-wide)

**Files (concept definition + every concept use site):**
- Modify: `include/tax/series/basis.hpp:61` (the `concept Basis` definition + its doc comments)
- Modify: `include/tax/core/expansion.hpp:29,38,369` and `include/tax/core/named.hpp:272,299` (the `tax::Basis< Basis >` shadow-workaround → `BasisPolicy< Basis >`)
- Modify: `include/tax/operators/arithmetic.hpp` (every `Basis B` template parameter → `BasisPolicy B`)
- Modify: `include/tax/series/operators.hpp:22-23` (`NonTaylorBasis` uses `Basis< B >`)
- Modify: `include/tax/series/chebyshev_math.hpp:30` (`ChebyshevLike` uses `Basis< B >`)
- Modify: `include/tax/series/taylor_basis.hpp`, `chebyshev_basis.hpp`, `legendre_basis.hpp`, `hermite_basis.hpp` (the `static_assert( Basis< … > )` footer in each)

**Interfaces:**
- Consumes: nothing new.
- Produces: the policy concept is named `tax::BasisPolicy`. The `tax::Basis< Basis >` shadow workaround disappears (`BasisPolicy< Basis >` no longer shadows the `typename Basis` parameter, so the `tax::` qualifier is dropped).

- [ ] **Step 1: Inventory every occurrence of the bare `Basis` concept**

Run:
```bash
grep -rn "\bBasis\b" include/ | grep -vE "TaylorBasis|ChebyshevBasis|LegendreBasis|HermiteBasis|OrthogonalBasis|NonTaylorBasis|ChebyshevBasisOn|is_tax_basis|typename Basis|class Basis|using basis|::basis|basis =|Basis_|BasisPolicy"
```
This is your worklist. Every remaining hit is either (a) the concept definition `concept Basis`, (b) a concept *use* `Basis<…>` or `Basis B`, or (c) a doc comment mentioning the concept. (Sanity: it should align with the file list above — `basis.hpp` def, the 5 `tax::Basis<Basis>` sites, the ~18 `Basis B` params in `arithmetic.hpp`, `operators.hpp`/`chebyshev_math.hpp` concept uses, and the 4 `static_assert(Basis<…>)` footers.)

- [ ] **Step 2: Rename the definition**

In `include/tax/series/basis.hpp`, change the concept definition (line ~61):
```cpp
template < typename B >
concept Basis =
```
to:
```cpp
template < typename B >
concept BasisPolicy =
```
Also update the surrounding doc comment that calls it "the `Basis` policy concept" / "A conforming policy" to name `BasisPolicy` (the prose at the top of the file). Do NOT rename `is_tax_basis`, `term`, etc.

- [ ] **Step 3: Rename the shadow-workaround sites (drop the `tax::` qualifier)**

In `include/tax/core/expansion.hpp` (lines 29, 38, 369) and `include/tax/core/named.hpp` (lines 272, 299), change:
```cpp
    requires Scalar< T > && tax::Basis< Basis > && IndexScheme< Scheme >
```
→
```cpp
    requires Scalar< T > && BasisPolicy< Basis > && IndexScheme< Scheme >
```
(and the `named.hpp` form `requires Scalar< T > && tax::Basis< Basis >` → `requires Scalar< T > && BasisPolicy< Basis >`). The `tax::` qualifier was a shadow workaround — now unnecessary since `BasisPolicy` does not collide with the `typename Basis` parameter.

- [ ] **Step 4: Rename the `Basis B` template parameters in arithmetic.hpp**

In `include/tax/operators/arithmetic.hpp`, every operator template is `template < typename T, Basis B, IndexScheme Scheme >`. Change each `Basis B` → `BasisPolicy B` (≈18 occurrences). Also update the two comment lines (`:15`, `:19`) that say "over `Basis B`" → "over `BasisPolicy B`". A safe scoped command:
```bash
perl -pi -e 's/\bBasis B\b/BasisPolicy B/g; s/`Basis B`/`BasisPolicy B`/g' include/tax/operators/arithmetic.hpp
```
(`\bBasis B\b` cannot match `TaylorBasis`/`OrthogonalBasis` etc., which are not followed by ` B`.)

- [ ] **Step 5: Rename the remaining concept-use sites**

- `include/tax/series/operators.hpp` (`NonTaylorBasis` definition, ~line 22-23): `Basis< B >` → `BasisPolicy< B >`.
- `include/tax/series/chebyshev_math.hpp:30` (`ChebyshevLike`): `Basis< B >` → `BasisPolicy< B >`.
- `include/tax/series/taylor_basis.hpp`, `chebyshev_basis.hpp`, `legendre_basis.hpp`, `hermite_basis.hpp`: the footer `static_assert( Basis< TaylorBasis > );` / `static_assert( Basis< ChebyshevBasis > );` etc. → `static_assert( BasisPolicy< … > );`. Use a scoped `perl -pi -e 's/\bBasis< /BasisPolicy< /g'` ONLY on these specific files, then grep each to confirm no `TaylorBasis`/`OrthogonalBasis`/`ChebyshevBasisOn` token was mangled (those have no space before `<` in the pattern `\bBasis< ` and are not preceded by a word boundary at `Basis`, so they are safe — but verify).

- [ ] **Step 6: Verify the rename is complete and surgical**

Run:
```bash
grep -rn "\bBasis\b" include/ | grep -vE "TaylorBasis|ChebyshevBasis|LegendreBasis|HermiteBasis|OrthogonalBasis|NonTaylorBasis|ChebyshevBasisOn|is_tax_basis|typename Basis|class Basis|using basis|::basis|basis =|BasisPolicy"
grep -rn "OrthogonalBasisPolicy\|TaylorBasisPolicy\|ChebyshevBasisPolicy" include/   # must be EMPTY (mangling check)
grep -rn "tax::Basis<" include/                                                      # must be EMPTY (shadow workaround gone)
```
Expected: the first grep returns nothing (every bare-concept use renamed); the second and third return nothing. If the mangling check is non-empty, you over-replaced a type name — fix it.

- [ ] **Step 7: Build + full suite**

Run: `cmake --build build -j && ctest --test-dir build -j`
Expected: build EXIT 0; `100% tests passed … out of 56`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor!: rename the Basis policy concept to BasisPolicy\n\nThe concept name Basis shadowed the ubiquitous typename Basis template\nparameter (forcing the tax::Basis<Basis> workaround). Rename the concept to\nBasisPolicy repo-wide and drop the qualifier. Type names (TaylorBasis,\nOrthogonalBasis, ...) and the typename Basis parameter are unchanged.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Move basis.hpp + taylor_basis.hpp into core/ (sever core→series)

**Files:**
- Rename: `include/tax/series/basis.hpp` → `include/tax/core/basis.hpp` (via `git mv`)
- Rename: `include/tax/series/taylor_basis.hpp` → `include/tax/core/taylor_basis.hpp` (via `git mv`)
- Modify includers of `series/basis.hpp`: `series.hpp`, `series/hermite_basis.hpp`, `series/chebyshev_basis.hpp`, `series/taylor_basis.hpp` (now `core/taylor_basis.hpp`), `series/aliases.hpp`, `series/legendre_basis.hpp`, `core/expansion.hpp`
- Modify includers of `series/taylor_basis.hpp`: `series.hpp`, `series/io.hpp`, `series/aliases.hpp`, `series/operators.hpp`, `core/expansion.hpp`

**Interfaces:**
- Consumes: the renamed concept from Task 1.
- Produces: `<tax/core/basis.hpp>` (the `BasisPolicy` concept) and `<tax/core/taylor_basis.hpp>` (the `TaylorBasis` policy) exist; `core/expansion.hpp` includes no `tax/series/*` header.

- [ ] **Step 1: git mv the two files**

```bash
git mv include/tax/series/basis.hpp include/tax/core/basis.hpp
git mv include/tax/series/taylor_basis.hpp include/tax/core/taylor_basis.hpp
```

- [ ] **Step 2: Update every include path**

Repoint all includers (the `taylor_basis.hpp` file itself includes `series/basis.hpp` → `core/basis.hpp`; and `core/taylor_basis.hpp` includes `kernels/cauchy.hpp` already — that stays, a legitimate core→kernels edge):
```bash
for f in $(grep -rln "tax/series/basis.hpp\|tax/series/taylor_basis.hpp" include/ tests/); do
  perl -pi -e 's{tax/series/basis\.hpp}{tax/core/basis.hpp}g; s{tax/series/taylor_basis\.hpp}{tax/core/taylor_basis.hpp}g' "$f"
done
grep -rn "tax/series/basis.hpp\|tax/series/taylor_basis.hpp" include/ tests/ || echo "no stale paths"
```
Expected: "no stale paths".

- [ ] **Step 3: Confirm core no longer includes series**

Run:
```bash
grep -n "tax/series" include/tax/core/expansion.hpp
```
Expected: nothing (the carrier's only former `series/` includes were `basis.hpp`+`taylor_basis.hpp`, now `core/`). If anything remains, repoint it (it must resolve to a `core/` or `kernels/` header — `core` may not depend on `series`).

- [ ] **Step 4: Build + full suite**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
Expected: build EXIT 0; `100% tests passed … out of 56`. (Header content unchanged — only locations + include paths.)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor!: move BasisPolicy + TaylorBasis from series/ into core/\n\nThe carrier is basis-generic and TaylorBasis-defaulted, so the BasisPolicy\nconcept and the default TaylorBasis policy belong with the carrier, not in\nseries/. git mv series/{basis,taylor_basis}.hpp -> core/ and repoint includers.\ncore/expansion.hpp no longer includes any tax/series header (core->series edge\nsevered); orthogonal bases include core/basis.hpp.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: Regularize the `Series<>` alias arity

**Files:**
- Modify: `include/tax/series/aliases.hpp:18-20` (the generic `Series` alias)

**Interfaces:**
- Consumes: the moved headers from Task 2.
- Produces: `tax::Series< Basis, N, M = 1, T = double >` — the generic univariate-or-multivariate basis-generic alias (previously univariate-only, with `T` in the `M` slot).

- [ ] **Step 1: Inspect current uses of `Series<>`**

Run: `grep -rn "Series<\|Series< " include/ tests/ | grep -vE "TaylorSeries|ChebyshevSeries|LegendreSeries|HermiteSeries|NamedSeries"`
Note every `tax::Series<...>` use (likely few/none). The change adds an `M` parameter with a default, so existing 2-arg `Series<Basis, N>` and 3-arg `Series<Basis, N, T>` uses must still resolve — see Step 2's ordering note.

- [ ] **Step 2: Regularize the alias**

In `include/tax/series/aliases.hpp`, replace:
```cpp
/// Univariate basis-generic series.
template < typename Basis, int N, typename T = double >
using Series = Expansion< T, Basis, IsotropicScheme< N, 1 > >;
```
with:
```cpp
/// Basis-generic expansion (univariate by default; set M for multivariate).
template < typename Basis, int N, int M = 1, typename T = double >
using Series = Expansion< T, Basis, IsotropicScheme< N, M > >;
```
> Ordering note: the old form had `T` in the 3rd slot; the new form has `M` (an `int`) there. If Step 1 found any `Series< Basis, N, SomeScalarType >` use (T in the 3rd slot), it would now bind `M` to a type and fail — rewrite such a use to `Series< Basis, N, 1, SomeScalarType >`. If Step 1 found no 3-arg type-in-3rd-slot uses (the expected case — the per-basis `XxxSeries<N,M,T>` aliases are the multivariate spelling people use, not the generic `Series<>`), no call-site change is needed.

- [ ] **Step 3: Build + full suite**

Run: `cmake --build build -j && ctest --test-dir build -j`
Expected: build EXIT 0; `100% tests passed … out of 56`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor: regularize the generic Series<> alias arity to <Basis,N,M=1,T>\n\nSeries<Basis,N,T> was univariate-only with T in the M slot, inconsistent with\nthe per-basis XxxSeries<N,M=1,T> aliases. Add the M parameter (default 1) so\nSeries<Cheb,3> stays univariate and Series<Cheb,3,2> expresses multivariate.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Phase exit criteria

- `grep -rn "\bBasis\b" include/ | grep -v <type-names/param/BasisPolicy>` returns nothing — the concept is uniformly `BasisPolicy`; no type name was mangled.
- `include/tax/core/basis.hpp` and `include/tax/core/taylor_basis.hpp` exist; `series/basis.hpp`/`series/taylor_basis.hpp` are gone; `grep -n "tax/series" include/tax/core/expansion.hpp` is empty.
- `Series<Basis,N,M=1,T>` arity regularized.
- `ctest` → `100% tests passed … out of 56`. Three commits landed.

## Self-review (completed)

- **Spec coverage (P2 slice):** rename concept `Basis`→`BasisPolicy` → Task 1; move `BasisPolicy`+`TaylorBasis` into core (F3, sever core→series) → Task 2; regularize `Series` arity → Task 3. (The spec also mentions "demote aliases.hpp to a pure-alias file" — that S8 include-hygiene cleanup is low-value and deferred to the P6 facade pass, where the include graph is reworked anyway; noted here so it isn't lost.)
- **Placeholders:** none — every site enumerated, exact greps with mangling-guard, exact `git mv` + perl include-repoint, exact alias replacement with the arity-ordering caveat called out.
- **Type/name consistency:** the rename touches only the concept `Basis` (guarded against `TaylorBasis`/`OrthogonalBasis`/`ChebyshevBasisOn`/`typename Basis`/`using basis`); the moved headers keep identical content; `Series<Basis,N,M=1,T>` matches the per-basis alias arity.
