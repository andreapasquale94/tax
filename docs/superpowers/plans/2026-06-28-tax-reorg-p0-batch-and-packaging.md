# tax Reorg — Phase 0: Remove Batch + Packaging Safety Net — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the first phase of the reorganization — remove the `Batch` SIMD-coefficient capability and add the packaging/CI safety net — leaving the tree building and all tests green.

**Architecture:** Phase 0 of the 10-phase migration in `docs/superpowers/specs/2026-06-28-tax-library-reorganization-design.md` (decisions D8/F9 + F7). It is intentionally first so every later phase works against the simpler `TE<N,M>` and a `find_package` smoke test guards the install layout. No structural moves yet.

**Tech Stack:** Header-only C++23; CMake ≥ 3.28; Eigen3; GoogleTest; built/tested in the mamba `tax` env.

## Global Constraints

- C++23, `cxx_std_23`; `CMAKE_CXX_EXTENSIONS OFF`.
- `constexpr` everywhere in the dense core; **no heap** in the dense core (`std::array` only; `std::vector` only in Sparse storage).
- Graded-lex `flatIndex` coefficient ordering is sacred — never change it.
- Kernel config macros (`TAX_USE_UNROLL`/`TAX_USE_STENCIL`/`TAX_STENCIL_MAX_BYTES`) stay in-header, identical project-wide (ODR); never inject from the build system.
- `M ≥ 1` always.
- Build/test command (run from repo root, mamba `tax` env active):
  `source /Users/andrea/miniforge3/etc/profile.d/conda.sh && conda activate tax`
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
- `clang-format` touched files, but **preserve the repo's indented-PP convention** (`#    define` inside `#ifndef`); clang-format 21 de-indents these — restore them after formatting.
- Consumer contract: users `#include <tax/tax.hpp>`.

**TDD note for this phase:** Tasks 1, 2, 4 are *removals/config*, for which "write a failing test first" does not apply — the existing 55-test suite is the regression net, and each task's verification is an explicit build + `ctest` + `grep`-absence check with expected output. Task 3 (adding `version.hpp`) is a true addition and uses the failing-test-first cycle.

---

### Task 1: Remove the `Batch` capability

**Files:**
- Delete: `include/tax/core/batch.hpp`
- Delete: `tests/core/test_batch.cpp`
- Delete: `docs/guide/batch.md`
- Modify: `include/tax/tax.hpp` (drop the `batch.hpp` include)
- Modify: `include/tax/core/expansion.hpp:21-24` (drop `Batch` forward-decl) and `:393-396` (drop `K` lane on `TE`)
- Modify: `include/tax/operators/named_math_unary.hpp:61` (drop "and Batch" from the comment)
- Modify: `tests/CMakeLists.txt:40` (drop the `test_batch` target)
- Modify: `mkdocs.yml:117` (drop the Batch nav entry)
- Modify: `CLAUDE.md` (lines 5, 30, 123, 131 — scrub Batch) and `README.md:43` (drop the Batch bullet)

**Interfaces:**
- Consumes: nothing (start of plan).
- Produces: `tax::TE<N, M = 1>` (no `K`); `tax::Batch`, `tax::Batchd`, `tax::Batchf`, and `Eigen::NumTraits<tax::Batch<…>>` no longer exist.

- [ ] **Step 1: Confirm `Batch`/`TE<…,K>` are used nowhere but the files above**

Run:
```bash
grep -rn "Batch\|TE< *[0-9A-Za-z_]\+, *[0-9A-Za-z_]\+, *[0-9A-Za-z_]" include/ tests/ benchmarks/ \
  | grep -v "core/batch.hpp\|tests/core/test_batch.cpp"
```
Expected: only the comment in `core/expansion.hpp:21-24`, the `TE` alias in `expansion.hpp:393-396`, and the comment in `operators/named_math_unary.hpp:61`. If any *other* hit references `Batch` or a 3-arg `TE<N,M,K>`, stop and report it (it must be migrated before deleting).

- [ ] **Step 2: Delete the Batch source, test, and doc**

```bash
git rm include/tax/core/batch.hpp tests/core/test_batch.cpp docs/guide/batch.md
```

- [ ] **Step 3: Drop the umbrella include**

In `include/tax/tax.hpp`, delete the line:
```cpp
#include <tax/core/batch.hpp>
```

- [ ] **Step 4: Simplify the `TE` alias and drop the `Batch` forward-decl**

In `include/tax/core/expansion.hpp`, delete the forward declaration (lines 21-24):
```cpp
// Forward declaration so the public `TE` alias can name the batched coefficient
// type without including <tax/core/batch.hpp> (which includes this header).
template < typename T, int K >
struct Batch;
```
and replace the `TE` alias (lines 393-396):
```cpp
/// `TE<N, M, K>` — order-N, M-variate dense `double` Taylor expansion.
template < int N, int M = 1, int K = 1 >
using TE = Expansion< std::conditional_t< K == 1, double, Batch< double, K > >, TaylorBasis,
                      IsotropicScheme< N, M >, storage::Dense >;
```
with:
```cpp
/// `TE<N, M>` — order-N, M-variate dense `double` Taylor expansion.
template < int N, int M = 1 >
using TE = Expansion< double, TaylorBasis, IsotropicScheme< N, M >, storage::Dense >;
```

- [ ] **Step 5: Fix the stale comment in `named_math_unary.hpp`**

In `include/tax/operators/named_math_unary.hpp:61`, change:
```cpp
// TaylorExpansion and Batch overloads already live directly in `tax`.
```
to:
```cpp
// TaylorExpansion overloads already live directly in `tax`.
```

- [ ] **Step 6: Drop the `test_batch` target**

In `tests/CMakeLists.txt`, delete the line:
```cmake
tax_add_test(test_batch SOURCES core/test_batch.cpp)
```

- [ ] **Step 7: Drop the Batch docs nav entry**

In `mkdocs.yml:117`, delete the line:
```yaml
      - Batch (SIMD) Coefficients: guide/batch.md
```

- [ ] **Step 8: Scrub Batch from CLAUDE.md and README.md**

In `CLAUDE.md`:
- Line 5: change `…including *mixed-order* axes, optional *batch* (SIMD-style) coefficients, and Eigen integration (`tax::la`).` → `…including *mixed-order* axes, and Eigen integration (`tax::la`).`
- Line 30: delete the tree line `│   │   ├── batch.hpp         #   Batch<T,K>: K expansions evaluated in lock-step (TE<N,M,K>)`
- Line 123: change `// T       = coefficient type (double, float, or Batch<double,K> for K lock-step expansions)` → `// T       = coefficient type (double or float)`
- Line 131: change `tax::TE<N, M = 1, K = 1>  // dense; K>1 → Batch<double,K> coefficients` → `tax::TE<N, M = 1>        // dense Taylor expansion`

In `README.md:43`, delete the bullet:
```markdown
- **Batch coefficients** — `TE<N, M, K>` makes each coefficient a `Batch<double,
```
(remove the full bullet, including its continuation line).

- [ ] **Step 9: Build and run the full suite**

Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j
```
Expected: configure OK; build EXIT 0; `100% tests passed, 0 tests failed out of 54` (one fewer than before — `test_batch` is gone).

- [ ] **Step 10: Verify Batch is fully gone**

Run:
```bash
grep -rn "Batch" include/ tests/ mkdocs.yml README.md CLAUDE.md
```
Expected: no matches (historical files under `docs/superpowers/` are out of scope and may still mention it).

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor(core)!: remove the Batch SIMD-coefficient capability\n\nD8/F9: delete Batch<T,K>, NumTraits<Batch>, Batchd/Batchf, the K lane on TE\n(-> TE<N,M>), the test, and the docs page. Not needed; resolves the\nthree-batched-spellings naming incoherence by elimination.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Housekeeping — delete dead artifacts, scrub Python refs, fix `basis.hpp` doc drift

**Files:**
- Delete: `Doxyfile`
- Delete: `include/tax/core/taylor_expansion.hpp` (the dead back-compat shim, included by nothing)
- Modify: `CLAUDE.md` (drop the `pyproject.toml` tree line + the "pyproject.toml is forward-looking" pitfall)
- Modify: `README.md` (drop any Python-bindings / `pyproject` mention if present)
- Modify: `include/tax/series/basis.hpp:38-44` (fix the 2-arg-vs-3-arg `derivative`/`integral` doc drift)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: no API change (deletions of dead files + comment fixes only).

- [ ] **Step 1: Confirm the shim and Doxyfile are referenced by nothing buildable**

Run:
```bash
grep -rn "taylor_expansion.hpp\|Doxyfile" include/ tests/ benchmarks/ CMakeLists.txt mkdocs.yml
```
Expected: no matches (the shim's former includers were repointed to `expansion.hpp` in a prior commit; `Doxyfile` is referenced by nothing).

- [ ] **Step 2: Delete the dead artifacts**

```bash
git rm Doxyfile include/tax/core/taylor_expansion.hpp
```

- [ ] **Step 3: Scrub `pyproject`/Python-binding references**

Run `grep -n "pyproject\|Python\|python" CLAUDE.md README.md` and remove the lines that describe a non-existent `pyproject.toml` / Python bindings:
- In `CLAUDE.md`: delete the `├── pyproject.toml …` tree line and the "**`pyproject.toml` is forward-looking:** there are no Python binding sources in the tree yet" pitfall bullet.
- In `README.md`: delete any "Python bindings (planned)" / `pyproject` line if present.

(Verify there is genuinely no `pyproject.toml`: `ls pyproject.toml` → "No such file or directory".)

- [ ] **Step 4: Fix the `basis.hpp` concept doc drift**

In `include/tax/series/basis.hpp`, the documentation comment (around lines 38-44) shows `derivative`/`integral` as 2-arg `(out, c)`, but the concept (`:71-72`) checks and every policy implements the **3-arg** form `(out, c, axis)`. Update the doc comment block to match:
```cpp
//   template< typename T, typename Scheme >
//   static constexpr void derivative( std::array< T, Scheme::nCoeff >& out,
//                                     const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept;
//
//   template< typename T, typename Scheme >
//   static constexpr void integral( std::array< T, Scheme::nCoeff >& out,
//                                   const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept;
```
(Match the surrounding comment's exact indentation/style; this is a comment-only change.)

- [ ] **Step 5: Build and test (no behavior change expected)**

Run:
```bash
cmake --build build -j && ctest --test-dir build -j
```
Expected: build EXIT 0; `100% tests passed … out of 54`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(printf 'chore: delete dead artifacts; scrub Python refs; fix basis doc drift\n\nF7: remove the unused Doxyfile and the dead core/taylor_expansion.hpp shim,\nstrike the phantom pyproject/Python-binding references, and correct the\nBasisPolicy concept comment to the 3-arg derivative/integral form the\nconcept actually checks.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: Add `version.hpp`

**Files:**
- Create: `include/tax/version.hpp`
- Create: `tests/core/test_version.cpp`
- Modify: `include/tax/tax.hpp` (include `version.hpp`)
- Modify: `tests/CMakeLists.txt` (register `test_version`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: macros `TAX_VERSION_MAJOR` (0), `TAX_VERSION_MINOR` (1), `TAX_VERSION_PATCH` (0), and `TAX_VERSION_STRING` ("0.1.0"), available via `<tax/tax.hpp>`.

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_version.cpp`:
```cpp
#include <gtest/gtest.h>

#include <tax/tax.hpp>

TEST( Version, MacrosMatchProject )
{
    EXPECT_EQ( TAX_VERSION_MAJOR, 0 );
    EXPECT_EQ( TAX_VERSION_MINOR, 1 );
    EXPECT_EQ( TAX_VERSION_PATCH, 0 );
    EXPECT_STREQ( TAX_VERSION_STRING, "0.1.0" );
}
```
Register it in `tests/CMakeLists.txt` next to the other core tests:
```cmake
tax_add_test(test_version SOURCES core/test_version.cpp)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target test_version -j
```
Expected: FAIL — compile error, `use of undeclared identifier 'TAX_VERSION_MAJOR'` (the macros don't exist yet).

- [ ] **Step 3: Create `version.hpp`**

Create `include/tax/version.hpp`:
```cpp
#pragma once

// Library version. Keep in sync with project(VERSION ...) in CMakeLists.txt.
// (A reorg phase may later generate this via configure_file; for now it is a
// hand-maintained header so the library stays usable header-only from source.)

#define TAX_VERSION_MAJOR 0
#define TAX_VERSION_MINOR 1
#define TAX_VERSION_PATCH 0
#define TAX_VERSION_STRING "0.1.0"
```

- [ ] **Step 4: Expose it from the umbrella**

In `include/tax/tax.hpp`, add as the first include after the file's opening comment:
```cpp
#include <tax/version.hpp>
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
cmake --build build --target test_version -j && ./build/tests/test_version
```
Expected: PASS (`[  PASSED  ] 1 test.`).

- [ ] **Step 6: Full suite still green**

Run: `ctest --test-dir build -j`
Expected: `100% tests passed … out of 55` (back to 55: `test_version` added).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(printf 'feat(packaging): add tax/version.hpp with TAX_VERSION_* macros\n\nF7: the headers exposed no version surface. Add TAX_VERSION_MAJOR/MINOR/PATCH\n+ TAX_VERSION_STRING (kept in sync with project(VERSION 0.1.0)); include it\nfrom the umbrella; cover it with a test.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: Install / `find_package` smoke test + `SameMinorVersion`

**Files:**
- Modify: `CMakeLists.txt:104` (`SameMajorVersion` → `SameMinorVersion`)
- Create: `tests/install/CMakeLists.txt` (standalone downstream consumer project)
- Create: `tests/install/consumer.cpp`
- Create: `.github/workflows/install.yml` (CI job: install + consume)

**Interfaces:**
- Consumes: `tax::tax` install/export from `CMakeLists.txt`; `<tax/tax.hpp>`, `TAX_VERSION_STRING` from Task 3.
- Produces: a CI guarantee that `cmake --install` + `find_package(tax CONFIG)` + a 5-line consumer compiles and links.

- [ ] **Step 1: Tighten package version compatibility (pre-1.0)**

In `CMakeLists.txt`, change the `write_basic_package_version_file` call (line 104):
```cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
```
to:
```cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
```
(Rationale: at `0.x`, `SameMajorVersion` lets any `0.y` satisfy a `0.x` request — too loose while the API churns; `SameMinorVersion` requires the minor to match.)

- [ ] **Step 2: Write the downstream consumer (the "test")**

Create `tests/install/consumer.cpp`:
```cpp
#include <tax/tax.hpp>

#include <cstdio>

int main()
{
    auto x = tax::TE< 5 >::variable( 1.0 );
    auto f = sin( x ) * exp( x );
    std::printf( "tax %s: f.value() = %g\n", TAX_VERSION_STRING, f.value() );
    return 0;
}
```
Create `tests/install/CMakeLists.txt` (a *standalone* project — it is configured separately, against the installed package, not as a subdirectory of the main build):
```cmake
cmake_minimum_required(VERSION 3.28)
project(tax_install_consumer LANGUAGES CXX)

find_package(tax CONFIG REQUIRED)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE tax::tax)
```

- [ ] **Step 3: Verify locally — install then consume**

Run from the repo root (mamba `tax` env active):
```bash
rm -rf build /tmp/tax-prefix /tmp/tax-consumer-build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_UNITTESTS=OFF
cmake --install build --prefix /tmp/tax-prefix
cmake -S tests/install -B /tmp/tax-consumer-build -DCMAKE_PREFIX_PATH=/tmp/tax-prefix
cmake --build /tmp/tax-consumer-build -j
/tmp/tax-consumer-build/consumer
```
Expected: install copies headers + `lib/cmake/tax/tax{Config,ConfigVersion,Targets}.cmake`; the consumer configures (finds `tax::tax`), builds, and prints `tax 0.1.0: f.value() = …`.

- [ ] **Step 4: Add the CI job**

Create `.github/workflows/install.yml`:
```yaml
name: install
on: [push, pull_request]
jobs:
  install-consume:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Eigen
        run: sudo apt-get update && sudo apt-get install -y libeigen3-dev
      - name: Configure + install tax
        run: |
          cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_UNITTESTS=OFF
          cmake --install build --prefix "$PWD/_prefix"
      - name: Build downstream consumer against the installed package
        run: |
          cmake -S tests/install -B _consumer -DCMAKE_PREFIX_PATH="$PWD/_prefix"
          cmake --build _consumer -j
          ./_consumer/consumer
```

- [ ] **Step 5: Restore the unit-test build dir**

The Step-3 verification reconfigured `build` with `-DTAX_BUILD_UNITTESTS=OFF`. Restore it:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j
```
Expected: `100% tests passed … out of 55`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(printf 'ci(packaging): smoke-test install + find_package; SameMinorVersion\n\nF7: the install/export machinery shipped but CI never exercised it. Add a\nstandalone downstream consumer (tests/install) and an install.yml job that\ninstalls tax then find_package()s + compiles it. Tighten the package version\ncompatibility from SameMajorVersion to SameMinorVersion for the 0.x series.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Phase exit criteria

- `cmake --build build -j` EXIT 0; `ctest` → `100% tests passed … out of 55`.
- `grep -rn Batch include/ tests/` → no matches; `Batchd`/`Batchf`/`NumTraits<Batch>` gone; `TE` is 2-param.
- Local install + downstream `find_package(tax CONFIG)` consumer compiles and runs.
- `Doxyfile`, `core/taylor_expansion.hpp`, and phantom Python refs removed.
- All four commits landed; tree clean.

## Self-review (completed)

- **Spec coverage (P0 slice):** D8/F9 Batch removal → Task 1; F7 dead artifacts + doc drift → Task 2; F7 version surface → Task 3; F7 install CI + `SameMinorVersion` → Task 4. ✔
- **Placeholders:** none — every step has exact paths, code, commands, and expected output.
- **Type/name consistency:** `TE<N,M>` (no `K`) used consistently; `TAX_VERSION_*`/`TAX_VERSION_STRING` defined in Task 3 and consumed in Task 4; `tax::tax`/`find_package(tax CONFIG)` match `CMakeLists.txt`. ✔
- **Correction vs spec:** the spec/blueprint said the current package compatibility was `AnyNewerVersion`; the actual value is `SameMajorVersion` (`CMakeLists.txt:104`). Task 4 changes the real value to `SameMinorVersion`.
