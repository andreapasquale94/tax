# tax Reorg — Phase 4: mixed operators → operators/; binary-math parity — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the mixed-order operator surface out of the `core/mixed_named.hpp` type header into `operators/mixed_*.hpp` (symmetry with the named layer; severs the core→operators include edge), and bring named + mixed binary math (`pow`/`atan2`) to full parity with the dense surface.

**Architecture:** Phase 4 of the reorg (spec `docs/superpowers/specs/2026-06-28-tax-library-reorganization-design.md`, H-C + H-D; the M-B namespace move is **deferred to P6**). Builds on Phase 3 (branch `claude/tax-library-reorg`, at `ad5e29b`). The mixed operators stay in `namespace tax::named` for now (M-B deferred). Directories stay current (`operators/`, `core/`); the tree move is Phase 6.

**Tech Stack:** Header-only C++23; Eigen3; GoogleTest; mamba `tax` env.

## Global Constraints

- C++23; `constexpr` everywhere in the dense core; no heap in the dense core; graded-lex ordering sacred; kernel macros in-header (ODR); `M ≥ 1`.
- Build/test (repo root, mamba `tax` env active):
  `source /Users/andrea/miniforge3/etc/profile.d/conda.sh && conda activate tax`
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
  Suite is **56 tests** entering P4; Task 2 adds one test file → **57** at phase end.
- `clang-format` touched files; preserve indented `#    define` PP directives (the operator macros use top-level `#define`, which clang-format keeps).
- After P4: `core/mixed_named.hpp` includes NO `tax/operators/*` header; the mixed operator surface lives in `operators/mixed_arithmetic.hpp` + `operators/mixed_math_unary.hpp` + `operators/mixed_math_binary.hpp` (all `namespace tax::named`); named & mixed binary math match the dense `pow`/`atan2` surface.

**TDD note:** Task 1 is a mechanical move (the 56-test suite + the existing mixed tests are the net). Task 2 ADDS binary forms — write the new-form test first (RED: forms don't resolve), then add the forms (GREEN).

---

### Task 1: Move the mixed operator surface into operators/ (sever core→operators)

**Files:**
- Create: `include/tax/operators/mixed_arithmetic.hpp` — the binary (`TAX_MIXED_BINARY_OP` +,-,*,/), scalar (`TAX_MIXED_SCALAR_OP` +,-,*,/), scalar-lhs (`operator+`/`*`/`-`/`/` with scalar on the left), and unary-negate operators, moved verbatim from `core/mixed_named.hpp` lines 316-386. In `namespace tax::named`.
- Create: `include/tax/operators/mixed_math_unary.hpp` — the `TAX_MIXED_UNARY_FN` macro + its 20 instantiations, moved verbatim from `core/mixed_named.hpp` lines 388-419. In `namespace tax::named`.
- Modify: `include/tax/core/mixed_named.hpp` — delete lines 316-419 (the moved operators); delete the `#include <tax/operators/{arithmetic,math_binary,math_unary}.hpp>` lines (15-17); keep the class, the `MTE` alias, the `tax::` re-exports, and the `tax::mixed` factories.
- Modify: `include/tax/tax.hpp` — add `#include <tax/operators/mixed_arithmetic.hpp>` and `#include <tax/operators/mixed_math_unary.hpp>` immediately after `#include <tax/core/mixed_named.hpp>` (line 15).

**Interfaces:**
- Consumes: `core/mixed_named.hpp` (the `MixedTaylorExpansion` type + `detail::MergedMixedTaylorExpansion`/`detail::TypeList`).
- Produces: the mixed operator free-functions in `tax::named` (unchanged signatures), now in `operators/`. `core/mixed_named.hpp` no longer includes any `operators/` header.

- [ ] **Step 1: Create operators/mixed_arithmetic.hpp**

`#pragma once`; file-header comment ("Free-function arithmetic surface for MixedTaylorExpansion: operands embed into the union (max-order per shared axis) before delegating to the inner TaylorExpansion operators. Mirrors operators/named_arithmetic.hpp."); includes `#include <tax/core/mixed_named.hpp>`, `#include <tax/operators/arithmetic.hpp>`, `#include <type_traits>`; open `namespace tax::named { ... }`; paste **verbatim** the operator code from `core/mixed_named.hpp` lines 316-386 (the `TAX_MIXED_BINARY_OP` macro+4 instantiations+undef, `TAX_MIXED_SCALAR_OP` macro+4+undef, the four scalar-lhs operators `+`/`*`/`-`/`/`, and the unary `operator-`). Do not alter the bodies.

- [ ] **Step 2: Create operators/mixed_math_unary.hpp**

`#pragma once`; file-header comment ("Unary math surface for MixedTaylorExpansion: applies tax::FN to the inner expansion, rewrapping with the same axes. Mirrors operators/named_math_unary.hpp."); includes `#include <cmath>`, `#include <tax/core/mixed_named.hpp>`, `#include <tax/operators/math_unary.hpp>`; open `namespace tax::named { ... }`; paste **verbatim** the `TAX_MIXED_UNARY_FN` macro + its 20 instantiations + `#undef` from `core/mixed_named.hpp` lines 388-419.

- [ ] **Step 3: Trim core/mixed_named.hpp**

Delete the operator block (lines 316-419) and the three `#include <tax/operators/...>` lines (15-17) from `core/mixed_named.hpp`. Keep: the includes it still needs (`<array>`, `<cstddef>`, `tax/core/concepts.hpp`, `tax/core/multi_index.hpp`, `tax/core/axis.hpp`, `tax/core/scheme/mixed.hpp`, `tax/core/expansion.hpp`, `<utility>`); the `MixedTaylorExpansion` class; the `MTE` alias; the `tax::` re-exports (`using named::MixedTaylorExpansion/MTE/OrderedAxis`); the `tax::mixed` factories.

- [ ] **Step 4: Wire the umbrella**

In `include/tax/tax.hpp`, after the `#include <tax/core/mixed_named.hpp>` line, add:
```cpp
#include <tax/operators/mixed_arithmetic.hpp>
#include <tax/operators/mixed_math_unary.hpp>
```
(They self-include the dense operators + the mixed type, so position after `core/mixed_named.hpp` is sufficient; the dense-operator includes later in the umbrella are idempotent via `#pragma once`.)

- [ ] **Step 5: Verify core is decoupled from operators**

Run:
```bash
grep -n "tax/operators" include/tax/core/mixed_named.hpp || echo "core/mixed_named.hpp free of tax/operators"
grep -nE "TAX_MIXED_BINARY_OP|TAX_MIXED_SCALAR_OP|TAX_MIXED_UNARY_FN" include/tax/core/mixed_named.hpp || echo "operator macros gone from core"
```
Expected: both echo the success message.

- [ ] **Step 6: Build + full suite**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build -j`
Expected: build EXIT 0; `100% tests passed … out of 56`. (The operators are the same free functions, found by ADL; behavior unchanged.)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(printf 'refactor!: move the mixed operator surface into operators/mixed_*.hpp\n\nH-C: the MixedTaylorExpansion arithmetic + scalar + unary-math operators were\ninlined in the type header (forcing core/mixed_named.hpp -> operators/...).\nMove them verbatim into operators/mixed_arithmetic.hpp + mixed_math_unary.hpp\n(namespace tax::named, mirroring the named layer) and drop the operators\nincludes from core. core/mixed_named.hpp no longer depends on operators/.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Binary-math parity — mixed_math_binary.hpp + backfill named

**Files:**
- Create: `include/tax/operators/mixed_math_binary.hpp` — the full mixed binary surface.
- Modify: `include/tax/operators/named_math_binary.hpp` — add the missing forms.
- Modify: `include/tax/tax.hpp` — add `#include <tax/operators/mixed_math_binary.hpp>` after `mixed_math_unary.hpp`.
- Create: `tests/operators/test_named_mixed_binary.cpp` — exercises the new forms.
- Modify: `tests/CMakeLists.txt` — register the new test.

**Interfaces:**
- Consumes: `MixedTaylorExpansion`/`MTE`, `NamedExpansion`/`NE`, `detail::MergedMixedTaylorExpansion`, `detail::MergedNamedExpansion`, and the dense `tax::pow`/`tax::atan2`.
- Produces: full `pow`/`atan2` parity for both named and mixed expansions.

- [ ] **Step 1: Write the failing parity test**

Create `tests/operators/test_named_mixed_binary.cpp`:
```cpp
#include <gtest/gtest.h>

#include <tax/tax.hpp>

// Binary-math parity for named + mixed expansions: pow(x,int/real),
// pow(x,x), pow(scalar,x), and the three atan2 forms must all resolve and
// match the inner (anonymous) expansion's result.

TEST( NamedMixedBinary, NamedPowAtan2Parity )
{
    auto x = tax::variable< "x", 1 >( 1.3 );  // NE<1, Axis<"x",1>> (TaylorBasis)
    auto y = tax::variable< "y", 1 >( 0.7 );

    // pow(NE, NE) and pow(scalar, NE)
    auto p_nn = pow( x, y );
    auto p_sn = pow( 2.0, x );
    EXPECT_NEAR( p_nn.value(), std::pow( 1.3, 0.7 ), 1e-10 );
    EXPECT_NEAR( p_sn.value(), std::pow( 2.0, 1.3 ), 1e-10 );

    // atan2(NE, const) and atan2(const, NE)
    auto a_nc = atan2( x, 2.0 );
    auto a_cn = atan2( 2.0, x );
    EXPECT_NEAR( a_nc.value(), std::atan2( 1.3, 2.0 ), 1e-10 );
    EXPECT_NEAR( a_cn.value(), std::atan2( 2.0, 1.3 ), 1e-10 );
}

TEST( NamedMixedBinary, MixedPowAtan2 )
{
    auto x = tax::mixed::variable< "x", 4 >( 1.3 );  // MTE
    auto y = tax::mixed::variable< "y", 4 >( 0.7 );

    auto p_int = pow( x, 2 );
    auto p_mm  = pow( x, y );
    auto p_sm  = pow( 2.0, x );
    auto a_mm  = atan2( x, y );
    auto a_mc  = atan2( x, 2.0 );
    auto a_cm  = atan2( 2.0, x );

    EXPECT_NEAR( p_int.value(), 1.3 * 1.3, 1e-10 );
    EXPECT_NEAR( p_mm.value(), std::pow( 1.3, 0.7 ), 1e-10 );
    EXPECT_NEAR( p_sm.value(), std::pow( 2.0, 1.3 ), 1e-10 );
    EXPECT_NEAR( a_mm.value(), std::atan2( 1.3, 0.7 ), 1e-10 );
    EXPECT_NEAR( a_mc.value(), std::atan2( 1.3, 2.0 ), 1e-10 );
    EXPECT_NEAR( a_cm.value(), std::atan2( 2.0, 1.3 ), 1e-10 );
}
```
Register in `tests/CMakeLists.txt` (next to the operator tests): `tax_add_test(test_named_mixed_binary SOURCES operators/test_named_mixed_binary.cpp)`. (Add `#include <cmath>` to the test if `std::pow`/`std::atan2` need it — they do.)

- [ ] **Step 2: RED — confirm the new forms don't resolve yet**

Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target test_named_mixed_binary -j
```
Expected: FAIL to compile — `pow(2.0, x)` / `pow(x, y)` / `atan2(x, 2.0)` / the entire `MixedPowAtan2` block do not resolve (named is missing `pow(NE,NE)`/`pow(scalar,NE)`/const `atan2`; mixed has no binary math). Record the errors.

- [ ] **Step 3: Backfill named_math_binary.hpp**

In `include/tax/operators/named_math_binary.hpp`, inside `namespace tax::named`, add (mirroring the existing `pow`/`atan2` there, using `using tax::pow;`/`using tax::atan2;`):
```cpp
/// `x^p` for a Taylor-valued exponent over the union of the operands' axes.
template < typename T, typename Basis, int N, typename... A, typename... B >
[[nodiscard]] auto pow( const NamedExpansion< T, Basis, N, A... >& x,
                        const NamedExpansion< T, Basis, N, B... >& p ) noexcept
{
    using R = detail::MergedNamedExpansion< T, Basis, N, detail::TypeList< A... >,
                                            detail::TypeList< B... > >;
    using tax::pow;
    return R{ pow( x.template embed< R >().inner(), p.template embed< R >().inner() ) };
}

/// `s^x` for a scalar base (axis set unchanged).
template < typename T, typename Basis, int N, typename... A >
[[nodiscard]] NamedExpansion< T, Basis, N, A... > pow(
    std::type_identity_t< T > s, const NamedExpansion< T, Basis, N, A... >& x ) noexcept
{
    using tax::pow;
    return NamedExpansion< T, Basis, N, A... >{ pow( s, x.inner() ) };
}

/// `atan2(y, x)` with a constant `x` (axis set unchanged).
template < typename T, typename Basis, int N, typename... A >
[[nodiscard]] NamedExpansion< T, Basis, N, A... > atan2(
    const NamedExpansion< T, Basis, N, A... >& y, std::type_identity_t< T > x ) noexcept
{
    using tax::atan2;
    return NamedExpansion< T, Basis, N, A... >{ atan2( y.inner(), x ) };
}

/// `atan2(y, x)` with a constant `y` (axis set unchanged).
template < typename T, typename Basis, int N, typename... A >
[[nodiscard]] NamedExpansion< T, Basis, N, A... > atan2(
    std::type_identity_t< T > y, const NamedExpansion< T, Basis, N, A... >& x ) noexcept
{
    using tax::atan2;
    return NamedExpansion< T, Basis, N, A... >{ atan2( y, x.inner() ) };
}
```

- [ ] **Step 4: Create operators/mixed_math_binary.hpp**

`#pragma once`; file-header comment ("Binary math surface for MixedTaylorExpansion: pow, atan2. Mirrors operators/named_math_binary.hpp; the inner call is unqualified after `using tax::FN` so ADL reaches the dense math."); includes `#include <cmath>`, `#include <tax/core/mixed_named.hpp>`, `#include <tax/operators/math_binary.hpp>`, `#include <type_traits>`; `namespace tax::named { ... }` with the full surface:
```cpp
template < typename T, typename... A >
[[nodiscard]] MixedTaylorExpansion< T, A... > pow( const MixedTaylorExpansion< T, A... >& x,
                                                   int n ) noexcept
{
    using tax::pow;
    return MixedTaylorExpansion< T, A... >{ pow( x.inner(), n ) };
}

template < typename T, typename... A >
[[nodiscard]] MixedTaylorExpansion< T, A... > pow( const MixedTaylorExpansion< T, A... >& x,
                                                   std::type_identity_t< T > p ) noexcept
{
    using tax::pow;
    return MixedTaylorExpansion< T, A... >{ pow( x.inner(), p ) };
}

template < typename T, typename... A, typename... B >
[[nodiscard]] auto pow( const MixedTaylorExpansion< T, A... >& x,
                        const MixedTaylorExpansion< T, B... >& p ) noexcept
{
    using R = detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    using tax::pow;
    return R{ pow( x.template embed< R >().inner(), p.template embed< R >().inner() ) };
}

template < typename T, typename... A >
[[nodiscard]] MixedTaylorExpansion< T, A... > pow( std::type_identity_t< T > s,
                                                   const MixedTaylorExpansion< T, A... >& x ) noexcept
{
    using tax::pow;
    return MixedTaylorExpansion< T, A... >{ pow( s, x.inner() ) };
}

template < typename T, typename... A, typename... B >
[[nodiscard]] auto atan2( const MixedTaylorExpansion< T, A... >& y,
                          const MixedTaylorExpansion< T, B... >& x ) noexcept
{
    using R = detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    using tax::atan2;
    return R{ atan2( y.template embed< R >().inner(), x.template embed< R >().inner() ) };
}

template < typename T, typename... A >
[[nodiscard]] MixedTaylorExpansion< T, A... > atan2( const MixedTaylorExpansion< T, A... >& y,
                                                     std::type_identity_t< T > x ) noexcept
{
    using tax::atan2;
    return MixedTaylorExpansion< T, A... >{ atan2( y.inner(), x ) };
}

template < typename T, typename... A >
[[nodiscard]] MixedTaylorExpansion< T, A... > atan2( std::type_identity_t< T > y,
                                                     const MixedTaylorExpansion< T, A... >& x ) noexcept
{
    using tax::atan2;
    return MixedTaylorExpansion< T, A... >{ atan2( y, x.inner() ) };
}
```
Then in `include/tax/tax.hpp` add `#include <tax/operators/mixed_math_binary.hpp>` after `mixed_math_unary.hpp`.

> If `pow`/`atan2` need to be surfaced as qualified `tax::pow`/`tax::atan2` for mixed (the named layer re-exports them via `using named::pow;` in named_math_binary.hpp), confirm `tax::pow(mte, …)` resolves; ADL on a `tax::named::MixedTaylorExpansion` argument finds `tax::named::pow`, and the existing `tax::pow`/`tax::atan2` re-exports in `named_math_binary.hpp` already pull `named::pow`/`named::atan2` (which now include the mixed overloads, same namespace). No extra re-export needed — but verify with the test's unqualified calls (which use ADL) and add a `tax::pow(x,2)` qualified call to the mixed test if you want to assert the qualified path too.

- [ ] **Step 5: GREEN — build the test + full suite**

Run:
```bash
cmake --build build --target test_named_mixed_binary -j && ./build/tests/test_named_mixed_binary
cmake --build build -j && ctest --test-dir build -j
```
Expected: the target compiles and `[ PASSED ] 2 tests`; full suite `100% tests passed … out of 57`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(printf 'feat(operators): binary-math parity for named + mixed expansions\n\nH-D: add operators/mixed_math_binary.hpp (pow(MTE,int/real/MTE/scalar),\natan2(MTE,MTE/const)) -- mixed had no binary math -- and backfill the missing\nforms in named_math_binary (pow(NE,NE), pow(scalar,NE), const-arg atan2). Both\nnow match the dense pow/atan2 surface. New parity test (57 tests).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Phase exit criteria

- `core/mixed_named.hpp` includes no `tax/operators/*` header and defines no operator macros; the mixed operators live in `operators/mixed_{arithmetic,math_unary,math_binary}.hpp`.
- Named + mixed expose the full `pow`/`atan2` surface (matching dense); the parity test passes.
- `ctest` → `100% tests passed … out of 57`. Two commits landed.

## Self-review (completed)

- **Spec coverage (P4 slice):** move mixed operators out + sever core→operators (H-C) → Task 1; mixed binary math + named backfill (H-D) → Task 2; M-B (namespace move) explicitly deferred to P6 (spec §8 updated). ✔
- **Placeholders:** none — Task 1 cites exact line ranges to move verbatim + umbrella wiring + verification greps; Task 2 gives the full new-form code for both named and mixed + a concrete RED/GREEN test.
- **Type/name consistency:** moved operators stay in `tax::named`; new binary forms mirror the existing `named_math_binary` pattern (`using tax::FN;` + `MergedNamedExpansion`/`MergedMixedTaylorExpansion` for the two-operand case); umbrella include order places mixed operator headers after `core/mixed_named.hpp`.
- **Risk note:** Task 1 is a verbatim move (low risk); the only subtlety is umbrella include ordering (mixed operator headers self-include the dense ops, so order is forgiving). Task 2's `pow(MTE,MTE)`/`pow(NE,NE)` require `a.value() > 0` at runtime (Taylor-valued exponent = `exp(b·log a)`), matching the dense contract — the test uses positive bases.
