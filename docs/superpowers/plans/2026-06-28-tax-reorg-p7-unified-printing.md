# Phase 7 — Unified Printing (F6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the two separate printing facilities into one (`io/series.hpp`), extend the basis render hook to carry a variable symbol so multivariate-orthogonal and named-over-orthogonal expansions print for the first time, and constrain the previously-unconstrained `to_string`.

**Architecture:** Today `io/series.hpp` owns the rich Taylor printer (Unicode subscripts/superscripts, implicit multiplication, tabular style, Eigen-vector rendering) plus an *unconstrained* `to_string(const F&)`; `bases/io.hpp` separately owns univariate-only orthogonal printing (`1 + 3*T_2`) via a 1-arg `B::term(k)` hook. Phase 7 folds the orthogonal printer into `io/series.hpp`, generalizes it to the full multivariate tensor-product form by widening the hook to `term(int k, std::string_view var)`, adds the named-over-orthogonal path, and routes everything through one set of `streamScalar` overloads behind `operator<<` / `series()` / `to_string()`.

**Tech Stack:** Header-only C++23, GoogleTest, CMake, the mamba `tax` env (conda clang++ + Eigen). Build/test:
`mamba run -n tax cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure`

## Global Constraints

- **Orthogonal output format (maintainer decision):** function-style with the variable **always shown**.
  Univariate `1 + 3*T_2(x₀)`; multivariate `2 + 5*T_2(x₀)*T_1(x₁)`; named-over-orthogonal `1 + 3*T_2(u)`.
  The basis label (`T_k` / `P_k` / `He_k`) stays **ASCII**; the variable symbol reuses the Taylor printer's
  naming — Unicode `x₀`/`x₁` for unnamed (via `detail::defaultVarName`), the bare axis name for named (via
  `detail::namedVarName`). The coefficient is **always** shown with a `*` separator (no `1`-elision), matching
  the pre-fold orthogonal style. The constant (total-degree-0) term prints as just the coefficient.
- **Accepted churn:** the only existing orthogonal-print assertion (`tests/series/test_series_unify.cpp`
  `IoStreamsInBasis`) changes from `"1 + 3*T_2"` to `"1 + 3*T_2(x₀)"`. The zero series still prints `"0"`.
  No other existing test string may change. The 58 currently-passing tests stay green (modulo this one string).
- **Mixed-expansion printing is OUT OF SCOPE.** No `streamScalar` overload for `MixedTaylorExpansion`;
  `to_string(mixed)` must become a clean *constraint* error (Task 3), not a deep instantiation error.
- **Behavior preservation:** the rich Taylor printer (subscripts, superscripts, tabular, threshold, Eigen
  vectors) is unchanged. Do not alter `writeSeries`, the Taylor `streamScalar` overloads, the proxies,
  `subscriptOf`/`superscriptOf`/`formatMagnitude`/`defaultVarName`/`namedVarName`, or the `series()` Eigen
  overload, except where a step explicitly says so.
- **No heap in the dense core, graded-lex ordering sacred, constexpr/noexcept preserved.** Printing already
  uses `std::string`/`std::ostringstream` (that is fine — it is the io layer, not the dense core). Do not
  touch coefficient layout or kernel math.
- **`clang-format` touched files**, preserving the repo's indented-PP convention (do not mass-reformat).

---

## File Structure

- `include/tax/io/series.hpp` — gains the orthogonal printer (Task 1 moves it here; Task 2 rewrites it to the
  unified multivariate form; Task 3 adds the `to_string`/`series` constraints).
- `include/tax/bases/io.hpp` — **deleted** in Task 1 (its two functions move into `io/series.hpp`).
- `include/tax/bases.hpp` — drops the `#include <tax/bases/io.hpp>` line (Task 1).
- `include/tax/expansion/taylor_basis.hpp`, `include/tax/bases/chebyshev_basis.hpp`,
  `include/tax/bases/hermite_basis.hpp`, `include/tax/bases/legendre_basis.hpp` — `term(int k)` →
  `term(int k, std::string_view var)` (Task 2).
- `include/tax/expansion/basis.hpp` — update the policy-contract doc comment for `term` (Task 2).
- `tests/series/test_series_unify.cpp` — churned assertion + new multivariate/named-orthogonal tests (Task 2).
- `tests/io/test_series.cpp` — new `to_string` constraint static-checks (Task 3).

---

### Task 1: Fold the orthogonal printer into `io/series.hpp` (verbatim move)

A pure relocation: the two functions in `bases/io.hpp` move byte-for-byte into `io/series.hpp`, the old file
is deleted, and the facade include is dropped. No behavior change — `IoStreamsInBasis` still asserts
`"1 + 3*T_2"` and still passes. This isolates "the move is clean" from the feature work in Task 2.

**Files:**
- Modify: `include/tax/io/series.hpp` (add the two functions; add one include)
- Delete: `include/tax/bases/io.hpp`
- Modify: `include/tax/bases.hpp:11` (remove the `bases/io.hpp` include)
- Test: `tests/series/test_series_unify.cpp` (unchanged; must still pass)

**Interfaces:**
- Consumes: `tax::Expansion<T,B,Scheme>`, `tax::TaylorBasis`, the `B::term(int k)` hook, `Scheme::isUnivariate`,
  `Scheme::order` (all already available transitively in `io/series.hpp` via `<tax/expansion/expansion.hpp>`).
- Produces: `tax::to_string(const Expansion<T,B,Scheme>&)` and `tax::operator<<(ostream&, const Expansion<T,B,Scheme>&)`,
  both constrained `Scheme::isUnivariate && !std::is_same_v<B, TaylorBasis>` — now defined in `io/series.hpp`.

- [ ] **Step 1: Confirm the current orthogonal-print test passes (baseline)**

Run: `mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build -R series_unify --output-on-failure`
Expected: PASS (`IoStreamsInBasis` asserts `tax::to_string(g) == "1 + 3*T_2"` and the zero case `"0"`).

- [ ] **Step 2: Add the explicit TaylorBasis include to `io/series.hpp`**

The moved code does `std::is_same_v<B, TaylorBasis>`; make the dependency explicit (it is already transitive).
In `include/tax/io/series.hpp`, in the include block (after `#include <tax/expansion/named.hpp>`), add:

```cpp
#include <tax/expansion/taylor_basis.hpp>
```

- [ ] **Step 3: Move the two orthogonal functions into `io/series.hpp`**

In `include/tax/io/series.hpp`, immediately AFTER the public Taylor `operator<<` block (the
`operator<<(std::ostream&, const TaylorExpansion<T, Scheme, S>&)` that ends around line 283) and BEFORE the
`// --- series() manipulator ---` comment, insert these two functions verbatim from `bases/io.hpp`:

```cpp
// --- Orthogonal-basis series (univariate; folded from bases/io.hpp) ----------

/// Human-readable univariate series in its own basis, e.g. "1 + 2*x + 3*x^2" or
/// "1 + 2*T_1 + 3*T_2". Zero coefficients are omitted; the zero series prints
/// as "0".
template < typename T, typename B, typename Scheme >
    requires( Scheme::isUnivariate && !std::is_same_v< B, TaylorBasis > )
[[nodiscard]] std::string to_string( const Expansion< T, B, Scheme >& f )
{
    std::ostringstream os;
    bool first = true;
    for ( int k = 0; k <= Scheme::order; ++k )
    {
        const T ck = f[std::size_t( k )];
        if ( ck == T{ 0 } ) continue;
        if ( !first ) os << " + ";
        first = false;
        if ( k == 0 )
            os << ck;
        else
            os << ck << "*" << B::term( k );
    }
    if ( first ) os << T{ 0 };
    return os.str();
}

template < typename T, typename B, typename Scheme >
    requires( Scheme::isUnivariate && !std::is_same_v< B, TaylorBasis > )
std::ostream& operator<<( std::ostream& os, const Expansion< T, B, Scheme >& f )
{
    return os << to_string( f );
}
```

- [ ] **Step 4: Delete `bases/io.hpp`**

Run: `git rm include/tax/bases/io.hpp`

- [ ] **Step 5: Drop the facade include**

In `include/tax/bases.hpp`, remove the line:

```cpp
#include <tax/bases/io.hpp>
```

- [ ] **Step 6: Build and run the full suite**

Run: `mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure`
Expected: clean build; **58/58 pass** (no test string changed — this is a pure move).

- [ ] **Step 7: Verify no dangling reference to the deleted header**

Run: `grep -rn "bases/io.hpp" include/ tests/`
Expected: no output.

- [ ] **Step 8: clang-format and commit**

```bash
clang-format -i include/tax/io/series.hpp include/tax/bases.hpp
git add include/tax/io/series.hpp include/tax/bases.hpp include/tax/bases/io.hpp
git commit -m "refactor(io): fold orthogonal printing into io/series.hpp"
```

---

### Task 2: Extend the render hook; multivariate + named orthogonal printing; new format

The feature work. Widen `term(int k)` → `term(int k, std::string_view var)` across all four bases and the
concept doc, then replace the univariate orthogonal printer (just moved in Task 1) with a unified
`writeOrthoSeries` core that renders the full multivariate tensor-product form and routes through
`streamScalar` (so `operator<<`, `series()`, and the generic `to_string` all reach it). Add the
named-over-orthogonal overloads. Churn the one affected test string; add coverage for the new cases.

**Files:**
- Modify: `include/tax/expansion/taylor_basis.hpp` (term signature + body; add `<string_view>`)
- Modify: `include/tax/bases/chebyshev_basis.hpp` (term signature + body; add `<string_view>`)
- Modify: `include/tax/bases/legendre_basis.hpp` (term signature + body; add `<string_view>`)
- Modify: `include/tax/bases/hermite_basis.hpp` (term signature + body; add `<string_view>`)
- Modify: `include/tax/expansion/basis.hpp` (policy-contract doc comment)
- Modify: `include/tax/io/series.hpp` (replace the Task-1 orthogonal functions with the unified machinery)
- Test: `tests/series/test_series_unify.cpp` (churn one assertion; add multivariate + named tests)

**Interfaces:**
- Consumes: `detail::defaultVarName(int, const SeriesOptions&)`, `detail::namedVarName<Axes...>(int, const SeriesOptions&)`,
  `detail::formatMagnitude(T, int)`, `totalDegree(MultiIndex)`, `Scheme::vars`, `Scheme::nCoeff`,
  `Scheme::multiOf(std::size_t)`, `IsotropicScheme<N,M>`, `named::NamedExpansion<T,B,N,Axes...>::vars_v`,
  `named::NamedExpansion<...>::inner()`, the new `B::term(int, std::string_view)`.
- Produces: `detail::writeOrthoSeries<Scheme,T,B>(os, coeffAt, nameOf, opts)`; `detail::streamScalar` overloads
  for unnamed orthogonal `Expansion<T,B,Scheme,storage::Dense>` and named orthogonal
  `named::NamedExpansion<T,B,N,Axes...>` (both `requires !std::is_same_v<B,TaylorBasis>`); the matching
  `operator<<` for each. These supersede the Task-1 standalone orthogonal `to_string`/`operator<<`.

- [ ] **Step 1: Update the `term` hook in `expansion/taylor_basis.hpp`**

Ensure `#include <string_view>` is present (add it near the other includes if missing). Replace the existing
`term(int k)` (around lines 33-38) with:

```cpp
    [[nodiscard]] static std::string term( int k, std::string_view var )
    {
        if ( k == 0 ) return "1";
        if ( k == 1 ) return std::string( var );
        return std::string( var ) + "^" + std::to_string( k );
    }
```

- [ ] **Step 2: Update `term` in `bases/chebyshev_basis.hpp`**

Add `#include <string_view>` if missing. Replace the existing `term(int k)` (around lines 44-48) with:

```cpp
    [[nodiscard]] static std::string term( int k, std::string_view var )
    {
        if ( k == 0 ) return "1";
        return "T_" + std::to_string( k ) + "(" + std::string( var ) + ")";
    }
```

- [ ] **Step 3: Update `term` in `bases/legendre_basis.hpp`**

Add `#include <string_view>` if missing. Replace the existing `term(int k)` (around lines 29-33) with:

```cpp
    [[nodiscard]] static std::string term( int k, std::string_view var )
    {
        if ( k == 0 ) return "1";
        return "P_" + std::to_string( k ) + "(" + std::string( var ) + ")";
    }
```

- [ ] **Step 4: Update `term` in `bases/hermite_basis.hpp`**

Add `#include <string_view>` if missing. Replace the existing `term(int k)` (around lines 29-33) with:

```cpp
    [[nodiscard]] static std::string term( int k, std::string_view var )
    {
        if ( k == 0 ) return "1";
        return "He_" + std::to_string( k ) + "(" + std::string( var ) + ")";
    }
```

- [ ] **Step 5: Update the policy-contract doc in `expansion/basis.hpp`**

Find the line documenting the term hook (around line 28):

```cpp
//   static std::string                term( int k );    // pretty basis label, e.g. "x^2" / "T_2"
```

Replace it with:

```cpp
//   static std::string  term( int k, std::string_view var );  // basis label, e.g. "x^2" / "T_2(x)"
```

- [ ] **Step 6: Replace the Task-1 orthogonal functions with the unified writer in `io/series.hpp`**

Delete the two functions added in Task 1 (the `// --- Orthogonal-basis series (univariate; folded ...` block:
the constrained `to_string(const Expansion<T,B,Scheme>&)` and its `operator<<`). In their place — but the
`writeOrthoSeries` core belongs in the `detail` namespace next to `writeSeries`. Add, inside
`namespace tax { namespace detail { ... } }`, immediately AFTER `writeSeries` (after its closing brace,
around line 199):

```cpp
/// Core writer for orthogonal-basis series: renders Σ coeff · Π_v B::term(deg_v, name_v).
/// Mirrors writeSeries' coefficient iteration / sign / threshold handling, but each
/// non-constant monomial is a product of per-variable basis labels, e.g. "T_2(x₀)*T_1(x₁)".
/// The coefficient is always shown (no 1-elision); the constant term prints as just the coefficient.
template < typename Scheme, typename T, typename B, typename CoeffAt, typename NameOf >
void writeOrthoSeries( std::ostream& os, CoeffAt&& coeffAt, NameOf&& nameOf,
                       const SeriesOptions& opts )
{
    constexpr int M = Scheme::vars;
    const std::size_t n = Scheme::nCoeff;
    const T thr = T( opts.threshold );
    bool any = false;
    for ( std::size_t k = 0; k < n; ++k )
    {
        const T c = coeffAt( k );
        if ( c == T{ 0 } ) continue;
        const bool neg = c < T{ 0 };
        const T mag = neg ? -c : c;
        if ( mag <= thr ) continue;
        const auto alpha = Scheme::multiOf( k );
        const bool is_const = totalDegree( alpha ) == 0;

        if ( !any )
        {
            if ( neg ) os << "-";
        } else
            os << ( neg ? " - " : " + " );
        any = true;

        os << formatMagnitude( mag, opts.precision );
        if ( !is_const )
            for ( int v = 0; v < M; ++v )
            {
                const int e = alpha[std::size_t( v )];
                if ( e == 0 ) continue;
                os << "*" << B::term( e, nameOf( v ) );
            }
    }
    if ( !any ) os << "0";
}
```

- [ ] **Step 7: Add the orthogonal `streamScalar` overloads in `io/series.hpp`**

Still inside `namespace tax::detail`, immediately after the named Taylor `streamScalar` overload (the one on
`named::NamedTaylorExpansion<T, N, Axes...>`, around line 227), add the unnamed and named orthogonal
dispatchers:

```cpp
// Unnamed orthogonal expansion (any non-Taylor basis), dense storage.
template < typename T, typename B, typename Scheme >
    requires( !std::is_same_v< B, TaylorBasis > )
void streamScalar( std::ostream& os, const Expansion< T, B, Scheme, storage::Dense >& f,
                   const SeriesOptions& opts )
{
    writeOrthoSeries< Scheme, T, B >(
        os, [&]( std::size_t k ) { return f[k]; },
        [&]( int v ) { return defaultVarName( v, opts ); }, opts );
}

// Named expansion over a non-Taylor basis (named-over-orthogonal).
template < typename T, typename B, int N, typename... Axes >
    requires( !std::is_same_v< B, TaylorBasis > )
void streamScalar( std::ostream& os, const named::NamedExpansion< T, B, N, Axes... >& f,
                   const SeriesOptions& opts )
{
    constexpr int M = named::NamedExpansion< T, B, N, Axes... >::vars_v;
    writeOrthoSeries< IsotropicScheme< N, M >, T, B >(
        os, [&]( std::size_t k ) { return f.inner()[k]; },
        [&]( int v ) { return namedVarName< Axes... >( v, opts ); }, opts );
}
```

- [ ] **Step 8: Add the public orthogonal `operator<<` (unnamed) in `io/series.hpp`**

After the public Taylor `operator<<` (the `TaylorExpansion<T, Scheme, S>` one, ~line 283) — i.e. where the
Task-1 functions were removed — add:

```cpp
// --- Orthogonal-basis streaming (univariate + multivariate) -----------------

template < typename T, typename B, typename Scheme >
    requires( !std::is_same_v< B, TaylorBasis > )
std::ostream& operator<<( std::ostream& os, const Expansion< T, B, Scheme, storage::Dense >& f )
{
    detail::streamScalar( os, f, SeriesOptions{} );
    return os;
}
```

- [ ] **Step 9: Add the named-orthogonal `operator<<` in `namespace tax::named`**

In the existing `namespace named { ... }` block near the end of `io/series.hpp` (the one already holding the
named Taylor `operator<<`), add a sibling for the non-Taylor basis:

```cpp
/// ADL hook so `os << namedExpansion` resolves for named-over-orthogonal expansions.
template < typename T, typename B, int N, typename... Axes >
    requires( !std::is_same_v< B, tax::TaylorBasis > )
std::ostream& operator<<( std::ostream& os, const NamedExpansion< T, B, N, Axes... >& f )
{
    tax::detail::streamScalar( os, f, tax::SeriesOptions{} );
    return os;
}
```

- [ ] **Step 10: Churn the affected test string**

In `tests/series/test_series_unify.cpp`, in `TEST( SeriesUnify, IoStreamsInBasis )`, change the univariate
assertion (the zero case is unchanged):

```cpp
    EXPECT_EQ( tax::to_string( g ), "1 + 3*T_2(x₀)" );
```

- [ ] **Step 11: Add multivariate-orthogonal and named-over-orthogonal tests**

Append to `tests/series/test_series_unify.cpp`:

```cpp
// Multivariate orthogonal printing: each variable carries its own basis label,
// joined by '*'. Build the coefficient array directly and pin one cross term.
TEST( SeriesUnify, MultivariateOrthogonalPrints )
{
    using E = tax::ChebyshevSeries< 3, 2 >;  // order 3, 2 vars
    std::array< double, E::nCoefficients > a{};
    a[0] = 2.0;  // constant
    tax::MultiIndex< 2 > m21{};
    m21[0] = 2;  // x₀ degree 2
    m21[1] = 1;  // x₁ degree 1
    a[tax::flatIndex< 2 >( m21 )] = 5.0;
    E g{ a };
    EXPECT_EQ( tax::to_string( g ), "2 + 5*T_2(x₀)*T_1(x₁)" );
}

// Named-over-orthogonal printing: the variable symbol is the axis name.
TEST( SeriesUnify, NamedOrthogonalUsesAxisName )
{
    auto u = tax::variable< "u", 4, tax::ChebyshevBasis >( 0.0 );
    const auto s = tax::to_string( tax::exp( u ) );
    EXPECT_NE( s.find( "(u)" ), std::string::npos );
}
```

If `tests/series/test_series_unify.cpp` does not already `#include <string>`, add it to the includes.

- [ ] **Step 12: Build and run the suite**

Run: `mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure`
Expected: clean build; all tests pass including the two new ones and the churned `IoStreamsInBasis`.
(Test count rises by 2 from the prior 58.)

- [ ] **Step 13: clang-format and commit**

```bash
clang-format -i include/tax/io/series.hpp include/tax/expansion/taylor_basis.hpp \
  include/tax/bases/chebyshev_basis.hpp include/tax/bases/legendre_basis.hpp \
  include/tax/bases/hermite_basis.hpp include/tax/expansion/basis.hpp
git add include/tax/io/series.hpp include/tax/expansion/taylor_basis.hpp \
  include/tax/bases/chebyshev_basis.hpp include/tax/bases/legendre_basis.hpp \
  include/tax/bases/hermite_basis.hpp include/tax/expansion/basis.hpp \
  tests/series/test_series_unify.cpp
git commit -m "feat(io): multivariate + named orthogonal printing via term(k, var)"
```

---

### Task 3: Constrain `to_string` and the scalar `series()`

The generic `to_string(const F&)` and scalar `series(const F&)` currently accept *any* non-Eigen `F` and only
fail deep inside instantiation if no `streamScalar` overload exists. Add an `is_printable` trait covering the
expansion kinds that actually have a `streamScalar` path (`Expansion` and `NamedExpansion`, any basis), and
constrain both functions so misuse — including `to_string(mixed)` — becomes a clean, immediate constraint
error. The Eigen vector/matrix path is preserved.

**Files:**
- Modify: `include/tax/io/series.hpp` (add `detail::is_printable`; constrain `series()` scalar + `to_string`)
- Test: `tests/io/test_series.cpp` (static constraint checks)

**Interfaces:**
- Consumes: `tax::Expansion`, `tax::named::NamedExpansion`, `Eigen::EigenBase`.
- Produces: `detail::is_printable_v<F>`; constrained `tax::series(const F&, SeriesOptions)` (scalar) and
  `tax::to_string(const F&, SeriesOptions)`.

- [ ] **Step 1: Write the failing static checks**

Append to `tests/io/test_series.cpp` (which already `#include`s `<tax/tax.hpp>`, `<string>`, `<sstream>`):

```cpp
// ---------------------------------------------------------------------------
// to_string is constrained: only printable expansions (any basis) + Eigen
// matrices of them resolve; arbitrary types are a clean constraint error.
// ---------------------------------------------------------------------------

namespace
{
template < typename F, typename = void >
struct HasToString : std::false_type
{
};
template < typename F >
struct HasToString< F, std::void_t< decltype( tax::to_string( std::declval< const F& >() ) ) > >
    : std::true_type
{
};
}  // namespace

static_assert( HasToString< tax::TE< 3 > >::value, "Taylor expansion must print" );
static_assert( HasToString< tax::ChebyshevSeries< 2 > >::value, "orthogonal expansion must print" );
static_assert( HasToString< tax::NE< 3, tax::Axis< "x", 1 > > >::value, "named expansion must print" );
static_assert( !HasToString< int >::value, "scalars are not printable expansions" );
static_assert( !HasToString< std::string >::value, "std::string is not a printable expansion" );
static_assert( !HasToString< tax::MTE< tax::OrderedAxis< "x", 1, 3 > > >::value,
               "mixed-expansion printing is out of scope" );
```

- [ ] **Step 2: Run to verify it fails to compile**

Run: `mamba run -n tax cmake --build build -j 2>&1 | tail -20`
Expected: FAIL — the negative `static_assert`s fire (`to_string` is still unconstrained, so
`HasToString<int>` / `HasToString<std::string>` / `HasToString<MTE>` are all `true`).

- [ ] **Step 3: Add the `is_printable` trait**

In `include/tax/io/series.hpp`, inside `namespace tax::detail` (place it near the top of the `detail` block,
after the helper declarations and before `writeSeries`), add:

```cpp
/// Types with a streamScalar path: any-basis Expansion and any-basis NamedExpansion.
/// Used to constrain to_string()/series() so misuse is a clean constraint error.
/// (Mixed expansions are intentionally excluded — they have no streamScalar overload.)
template < typename >
struct is_printable : std::false_type
{
};
template < typename T, typename B, typename Scheme, typename Storage >
struct is_printable< Expansion< T, B, Scheme, Storage > > : std::true_type
{
};
template < typename T, typename B, int N, typename... Axes >
struct is_printable< named::NamedExpansion< T, B, N, Axes... > > : std::true_type
{
};
template < typename F >
inline constexpr bool is_printable_v = is_printable< F >::value;
```

- [ ] **Step 4: Constrain the scalar `series()`**

In `include/tax/io/series.hpp`, replace the scalar `series()` constraint (currently
`requires( !std::is_base_of_v< Eigen::EigenBase< F >, F > )`) with the printable-expansion constraint:

```cpp
template < typename F >
    requires( detail::is_printable_v< F > )
[[nodiscard]] detail::ScalarSeriesProxy< F > series( const F& f, SeriesOptions opts = {} )
{
    return { f, opts };
}
```

- [ ] **Step 5: Constrain `to_string`**

Replace the generic `to_string` template with the constrained form (scalar printable OR Eigen matrix):

```cpp
template < typename F >
    requires( detail::is_printable_v< F > || std::is_base_of_v< Eigen::EigenBase< F >, F > )
[[nodiscard]] std::string to_string( const F& f, SeriesOptions opts = {} )
{
    std::ostringstream os;
    os << series( f, opts );
    return os.str();
}
```

- [ ] **Step 6: Build and run the full suite**

Run: `mamba run -n tax cmake --build build -j && mamba run -n tax ctest --test-dir build --output-on-failure`
Expected: clean build (all `static_assert`s now hold); all tests pass.

- [ ] **Step 7: clang-format and commit**

```bash
clang-format -i include/tax/io/series.hpp tests/io/test_series.cpp
git add include/tax/io/series.hpp tests/io/test_series.cpp
git commit -m "fix(io): constrain to_string/series to printable expansions"
```

---

## Self-Review Notes (controller)

- **Spec coverage (F6):** fold `bases/io.hpp` → `io/series.hpp` (Task 1); `term(int k, std::string_view var)`
  + multivariate-orthogonal + named-over-orthogonal `operator<<` (Task 2); constrain `to_string` (Task 3). All
  covered.
- **Type consistency:** `term(int, std::string_view)` is updated in all four bases AND the concept doc in the
  same task (Task 2) — no intermediate signature mismatch. `streamScalar` overloads are disjoint by
  `requires(!is_same_v<B,TaylorBasis>)` vs the Taylor/alias overloads (which fix `B == TaylorBasis`), so no
  ambiguity. The unnamed orthogonal `operator<<`/`streamScalar` are dense-only (no sparse orthogonal alias
  exists — YAGNI).
- **Churn budget:** exactly one existing assertion changes (`IoStreamsInBasis`), per the Global Constraints.
- **Out-of-scope:** mixed printing stays unsupported and is now a clean constraint error (locked by a negative
  `static_assert` in Task 3).
