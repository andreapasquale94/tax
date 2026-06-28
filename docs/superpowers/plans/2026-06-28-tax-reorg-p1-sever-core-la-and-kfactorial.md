# tax Reorg — Phase 1: Sever core→la + k!-member constraint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the carrier's `gradient()`/`hessian()` members (re-homing their bodies into the free `tax::la::` functions) so `<tax/expansion.hpp>` no longer depends on the `tax::la` module, and constrain the `k!`-scaled value accessors to `TaylorBasis` so they can't silently return wrong numbers on orthogonal bases.

**Architecture:** Phase 1 of the reorg (spec `docs/superpowers/specs/2026-06-28-tax-library-reorganization-design.md`, decisions D9/F2 + F1). Builds on Phase 0 (branch `claude/tax-library-reorg`, currently at `e16f337`). No directory moves yet — this is the layering/correctness pass that the later structural moves ride on.

**Tech Stack:** Header-only C++23; Eigen3; GoogleTest; mamba `tax` env.

## Global Constraints

- C++23; `constexpr` everywhere in the dense core; no heap in the dense core; graded-lex coefficient ordering is sacred; kernel config macros stay in-header (ODR); `M ≥ 1`.
- Build/test (repo root, mamba `tax` env active):
  `source /Users/andrea/miniforge3/etc/profile.d/conda.sh && conda activate tax`
  `cmake --build build -j && ctest --test-dir build -j` — the full suite is **55 tests** and must stay **100% passing** through every task of this phase (no test count change expected: tests are rewritten, not added/removed, except Task 2 may add one).
- `clang-format` touched files but preserve the repo's indented-PP convention (`#    define` inside `#ifndef`); clang-format 21 de-indents these — restore them. (P1's touched files have no nested PP.)
- After P1: `<tax/core/expansion.hpp>` must NOT `#include` any `tax/la/*` header; it includes `<Eigen/Core>` directly. The carrier `gradient()`/`hessian()` members are gone; `tax::la::gradient(f)`/`hessian(f)` are the only form. The carrier keeps `eval(const Eigen::MatrixBase<…>&)`, `eval(const Input&)`, `eval(T)`. The value-form `derivative(MultiIndex)` / `derivative<Alpha...>()` members and the public `tax::la::invert` are constrained to `TaylorBasis`.

**TDD note:** Task 1 is a refactor (move + rewrite call sites); the 55-test suite is its regression net — verification is build + `ctest` green + a grep that `expansion.hpp` no longer includes `tax/la`. Task 2 adds a real compile-time test (a `requires`-expression) proving the constraint, then applies it.

---

### Task 1: Re-home gradient()/hessian() into free functions; sever the core→la include

**Files:**
- Modify: `include/tax/la/derivatives.hpp:37-54` (move the loop bodies into the free `gradient`/`hessian`)
- Modify: `include/tax/core/expansion.hpp:12` (swap include) and `:327-360` (delete the two members)
- Modify: `tests/mixed/test_mixed_la.cpp` (9 call sites), `tests/eigen/test_eigen_gradient.cpp:22`, `tests/eigen/test_eigen_hessian.cpp:23`

**Interfaces:**
- Consumes: nothing new.
- Produces: `tax::la::gradient(const TaylorExpansion<T,Scheme,S>&) -> Eigen::Matrix<T,Scheme::vars,1>` and `tax::la::hessian(...) -> Eigen::Matrix<T,Scheme::vars,Scheme::vars>` are now self-contained (no longer delegate to a member). The carrier has no `gradient()`/`hessian()` members.

- [ ] **Step 1: Move the gradient body into the free function**

In `include/tax/la/derivatives.hpp`, replace the delegating free `gradient` (lines 36-42):
```cpp
/// Compute the gradient of a scalar `TaylorExpansion` at its expansion point.
template < typename T, typename Scheme, typename S >
[[nodiscard]] Eigen::Matrix< T, Scheme::vars, 1 > gradient(
    const TaylorExpansion< T, Scheme, S >& f ) noexcept
{
    return f.gradient();
}
```
with the re-homed body:
```cpp
/// Gradient `[df/dx_0, …, df/dx_{M-1}]` of a scalar `TaylorExpansion` at its expansion point.
template < typename T, typename Scheme, typename S >
[[nodiscard]] Eigen::Matrix< T, Scheme::vars, 1 > gradient(
    const TaylorExpansion< T, Scheme, S >& f ) noexcept
{
    Eigen::Matrix< T, Scheme::vars, 1 > g;
    MultiIndex< Scheme::vars > alpha{};
    for ( int i = 0; i < Scheme::vars; ++i )
    {
        alpha[std::size_t( i )] = 1;
        g( i ) = f.derivative( alpha );
        alpha[std::size_t( i )] = 0;
    }
    return g;
}
```

- [ ] **Step 2: Move the hessian body into the free function**

In the same file, replace the delegating free `hessian` (lines 48-54):
```cpp
/// Compute the Hessian matrix of a scalar `TaylorExpansion` at its expansion point.
template < typename T, typename Scheme, typename S >
[[nodiscard]] Eigen::Matrix< T, Scheme::vars, Scheme::vars > hessian(
    const TaylorExpansion< T, Scheme, S >& f ) noexcept
{
    return f.hessian();
}
```
with:
```cpp
/// Hessian `H(i,j) = d^2 f / (dx_i dx_j)` of a scalar `TaylorExpansion` at its expansion point.
template < typename T, typename Scheme, typename S >
[[nodiscard]] Eigen::Matrix< T, Scheme::vars, Scheme::vars > hessian(
    const TaylorExpansion< T, Scheme, S >& f ) noexcept
{
    Eigen::Matrix< T, Scheme::vars, Scheme::vars > H;
    for ( int i = 0; i < Scheme::vars; ++i )
    {
        for ( int j = 0; j < Scheme::vars; ++j )
        {
            MultiIndex< Scheme::vars > alpha{};
            alpha[std::size_t( i )] += 1;
            alpha[std::size_t( j )] += 1;
            H( i, j ) = f.derivative( alpha );
        }
    }
    return H;
}
```

- [ ] **Step 3: Delete the carrier members and swap the core include**

In `include/tax/core/expansion.hpp`, delete the entire Gradient/Hessian member block (the comment header at line 327 `// Gradient and Hessian (Taylor value semantics)` through the closing of `hessian()` at line 360, i.e. both the `gradient()` and `hessian()` member functions and their section comment).

Then change the include at line 12:
```cpp
#include <tax/la/types.hpp>
```
to:
```cpp
#include <Eigen/Core>
```
(The carrier no longer needs `tax::la::VecNT`/`MatNT` — only `gradient`/`hessian` used them — but the retained `eval(const Eigen::MatrixBase<…>&)` member at line ~243 needs `Eigen::MatrixBase`, which `<Eigen/Core>` provides.)

- [ ] **Step 4: Verify the core no longer depends on the la module**

Run:
```bash
grep -n "tax/la" include/tax/core/expansion.hpp
grep -n "VecNT\|MatNT\|tax::la" include/tax/core/expansion.hpp
```
Expected: zero matches for both (no `tax/la/*` include, no `VecNT`/`MatNT`/`tax::la` use remain in the carrier). If any remain, find and resolve before continuing.

- [ ] **Step 5: Rewrite the member call sites in tests**

Replace `f.gradient()` → `tax::la::gradient(f)` and `f.hessian()` → `tax::la::hessian(f)` at every call site. The exact sites (use the receiver expression in place of `f`):
- `tests/eigen/test_eigen_gradient.cpp:22` — `auto g2 = f.gradient();` → `auto g2 = tax::la::gradient(f);`
- `tests/eigen/test_eigen_hessian.cpp:23` — `auto H2 = f.hessian();` → `auto H2 = tax::la::hessian(f);`
- `tests/mixed/test_mixed_la.cpp` lines 88, 89, 111 (`.gradient()`), 135, 136, 167 (`.hessian()`), 262, 268 (`v(0).gradient()`, `v(1).gradient()`), 297 (`dot.gradient()`):
  e.g. `const auto g_me = f_me.gradient();` → `const auto g_me = tax::la::gradient( f_me );`;
  `const auto gv0 = v( 0 ).gradient();` → `const auto gv0 = tax::la::gradient( v( 0 ) );`;
  `const auto g = dot.gradient();` → `const auto g = tax::la::gradient( dot );`.
  Grep to confirm none remain: `grep -rn "\.gradient()\|\.hessian()" tests/` → only matches inside `tax::la::gradient(...)`/`tax::la::hessian(...)` calls (there should be none of the bare `.gradient()`/`.hessian()` member form left).

- [ ] **Step 6: Build and run the full suite**

Run:
```bash
cmake --build build -j && ctest --test-dir build -j
```
Expected: build EXIT 0; `100% tests passed … out of 55`. (The gradient/hessian numerics are unchanged — same loop, now in the free function.)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor(core)!: move gradient()/hessian() off the carrier; sever core->la\n\nD9/F2: re-home the carrier gradient()/hessian() member bodies into the free\ntax::la::gradient/hessian (which previously just delegated), delete the members,\nand swap expansion.hpp s <tax/la/types.hpp> include for <Eigen/Core> (kept for\nthe retained eval(Eigen) member). The expansion core no longer depends on the\ntax::la module. Call sites use tax::la::gradient(f)/hessian(f).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Constrain the k!-value accessors and invert to TaylorBasis

**Files:**
- Modify: `include/tax/core/expansion.hpp:198` and `:209` (constrain the two value-form `derivative` members)
- Modify: `include/tax/la/invert.hpp:90` (`is_te_v` → `is_taylor_te_v`)
- Create: `tests/core/test_kfactorial_constraint.cpp` (compile-time proof the constraint holds)
- Modify: `tests/CMakeLists.txt` (register the new test)

**Interfaces:**
- Consumes: the carrier from Task 1.
- Produces: `Expansion<T,Basis,Scheme>::derivative(MultiIndex)` and `::derivative<Alpha...>()` are SFINAE-disabled unless `Basis` is `TaylorBasis`; `tax::la::invert` only accepts Taylor maps.

- [ ] **Step 1: Guard — confirm nothing calls the value-form derivative on a non-Taylor expansion**

Run:
```bash
grep -rn "\.derivative\(\|\.template derivative<\|\.derivative<" include/ tests/
```
Review each hit: every call must be on a `TaylorExpansion`/`TE`/`NamedTaylorExpansion`/sparse-`TE` (Taylor) receiver, OR inside a context already constrained to `is_taylor_te_v` (e.g. `la/derivatives.hpp:28,77`, `la/axis_diff.hpp`). If ANY hit is on a Chebyshev/Legendre/Hermite (orthogonal) expansion, STOP and report it — that call is exercising the k!-bug and must be removed/redirected, not silently broken by the constraint.

- [ ] **Step 2: Write the failing compile-time test**

Create `tests/core/test_kfactorial_constraint.cpp`:
```cpp
#include <gtest/gtest.h>

#include <tax/tax.hpp>

// The k!-scaled value accessors apply Taylor semantics and are wrong for
// orthogonal bases, so they must be callable on a Taylor expansion and
// NON-callable (SFINAE-disabled) on Chebyshev/Legendre/Hermite.

template < typename E >
concept HasValueDerivative = requires( const E e, tax::MultiIndex< E::scheme::vars > a ) {
    e.derivative( a );
};

using TaylorE = tax::TE< 3, 2 >;
using ChebE = tax::ChebyshevSeries< 3, 2 >;
using LegE = tax::LegendreSeries< 3, 2 >;

static_assert( HasValueDerivative< TaylorE >,
               "Taylor expansion must expose the k!-scaled derivative()" );
static_assert( !HasValueDerivative< ChebE >,
               "Chebyshev expansion must NOT expose the k!-scaled derivative()" );
static_assert( !HasValueDerivative< LegE >,
               "Legendre expansion must NOT expose the k!-scaled derivative()" );

TEST( KFactorialConstraint, TaylorOnly )
{
    // The static_asserts above are the real test; this keeps a runtime hook so
    // the file is a normal gtest TU.
    SUCCEED();
}
```
Register it in `tests/CMakeLists.txt` next to the other core tests:
```cmake
tax_add_test(test_kfactorial_constraint SOURCES core/test_kfactorial_constraint.cpp)
```

> Note on the test: `E::scheme` is the carrier's scheme alias and `E::scheme::vars` the variable count (the same `Scheme::vars` used throughout `expansion.hpp`). If `E::scheme` is not the correct spelling of the carrier's scheme type alias, inspect `include/tax/core/expansion.hpp` for the public alias (e.g. `using scheme = Scheme;`) and use the actual name; do NOT invent one — report if absent.

- [ ] **Step 3: Run the test — verify the negative `static_assert`s FAIL to compile now**

Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target test_kfactorial_constraint -j
```
Expected: FAIL — the two `!HasValueDerivative<ChebE/LegE>` static_asserts fire (`static assertion failed: Chebyshev expansion must NOT expose…`), because the unconstrained `derivative()` member is currently callable on every basis. This is the RED state: the k!-bug is present.

- [ ] **Step 4: Constrain the two value-form `derivative` members**

In `include/tax/core/expansion.hpp`, add a `requires` clause to each value-form `derivative` member.

The runtime-index form (line 198):
```cpp
    [[nodiscard]] constexpr T derivative( const MultiIndex< Scheme::vars >& alpha ) const noexcept
```
→
```cpp
    [[nodiscard]] constexpr T derivative( const MultiIndex< Scheme::vars >& alpha ) const noexcept
        requires( std::is_same_v< Basis, TaylorBasis > )
```

The compile-time form (line 209, the `template < int... Alpha >` overload):
```cpp
    template < int... Alpha >
    [[nodiscard]] constexpr T derivative() const noexcept
```
→
```cpp
    template < int... Alpha >
    [[nodiscard]] constexpr T derivative() const noexcept
        requires( std::is_same_v< Basis, TaylorBasis > )
```
(`std::is_same_v` is available — `<type_traits>` is already included by `expansion.hpp`. `TaylorBasis` is in scope.)

- [ ] **Step 5: Gate `invert` on the Taylor trait**

In `include/tax/la/invert.hpp`, line 90, change the constraint on the public `invert`:
```cpp
    requires( detail::is_te_v< typename Derived::Scalar > )
```
→
```cpp
    requires( detail::is_taylor_te_v< typename Derived::Scalar > )
```
(The `detail::identityMap`/`composeOne`/`composeMap`/`linear` helpers are reachable only via `invert`, so this one gate is sufficient to keep the Picard inversion Taylor-only.)

- [ ] **Step 6: GREEN — the constraint test passes**

Run:
```bash
cmake --build build --target test_kfactorial_constraint -j && ./build/tests/test_kfactorial_constraint
```
Expected: compiles (all three `static_assert`s now hold — Taylor has the member, orthogonal don't) and `[  PASSED  ] 1 test.`

- [ ] **Step 7: Full suite green**

Run:
```bash
cmake --build build -j && ctest --test-dir build -j
```
Expected: `100% tests passed … out of 56` (55 + `test_kfactorial_constraint`). If any existing test fails to compile, it was calling a now-constrained member on a non-Taylor expansion — that is a real bug surfaced by the constraint (Step 1 should have caught it); STOP and report rather than relaxing the constraint.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(printf 'fix(core)!: constrain k!-value accessors + invert to TaylorBasis\n\nF1: Expansion::derivative(MultiIndex) and derivative<Alpha...>() apply k!\nTaylor scaling and silently returned wrong numbers on orthogonal bases; gate\nboth members on Basis == TaylorBasis. Gate tax::la::invert on is_taylor_te_v\n(the Picard inversion is Taylor-only). Add a compile-time test that the value\naccessors are callable on Taylor and ill-formed on Chebyshev/Legendre.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Phase exit criteria

- `<tax/core/expansion.hpp>` includes no `tax/la/*` header (grep clean); includes `<Eigen/Core>`; has no `gradient()`/`hessian()` members; keeps `eval(Eigen)`/`eval(Input)`/`eval(T)`.
- `tax::la::gradient(f)`/`hessian(f)` are self-contained and return the same values as before (existing eigen/mixed la tests pass).
- The value-form `derivative` members and `tax::la::invert` are TaylorBasis-only; the new compile-time test proves it.
- `ctest` → `100% tests passed … out of 56`. Both commits landed.

## Self-review (completed)

- **Spec coverage (P1 slice):** F2/D9 (sever core→la, keep eval(Eigen)) → Task 1; F1 (k!-member + invert constraint) → Task 2. ✔
- **Placeholders:** none — exact code for every edit, exact call-site list, exact commands + expected output. The one judgment point (the `E::scheme::vars` spelling in the test) is called out with a concrete fallback instruction.
- **Type/name consistency:** the free `gradient`/`hessian` keep their existing return types `Eigen::Matrix<T,Scheme::vars,1>` / `<…,Scheme::vars>`; `is_taylor_te_v` is the trait already used at `derivatives.hpp:20,62`; `TaylorBasis`/`std::is_same_v` are in scope in `expansion.hpp`. ✔
- **Ordering:** Task 1 re-homes the body (calling `f.derivative(alpha)` on a Taylor `f`) before Task 2 constrains `derivative` — order-independent since `f` is always TaylorBasis at those sites. ✔
