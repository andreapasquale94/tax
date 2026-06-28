# tax Library Reorganization — Design

**Date:** 2026-06-28
**Branch base:** `claude/orthogonal-polynomials-redesign-c0ki66` (post basis-generic redesign + the merged cleanup)
**Status:** Design — approved decisions captured; pending maintainer review of this spec before planning.

---

## 1. Goal & scope

Reorganize the header-only C++23 library **tax** for better maintainability, naming coherence, and
integration — across all four maintainer-prioritized dimensions: public API naming/ergonomics, internal
module boundaries, Eigen/LA integration, and headers/packaging.

The library is pre-1.0; **breaking changes are permitted** (namespaces, public include paths, public API
names/aliases, header tree, back-compat shims). The companion `ode/ads` plugin builds on tax and will adapt;
compatibility is not preserved for its sake, with one explicit exception: the `tax::la` namespace spelling is
retained (see §4) to avoid gratuitous churn in downstream code.

This effort grew out of a multi-agent deep review (6 areas, 43 proposals, adversarially critiqued). The full
review blueprint is the input to this spec; this document records the **decided** design.

### Non-goals (explicitly out of scope)
- Reworking `Batch` to remove its Eigen dependency (so a truly "Eigen-free core" is **not** promised).
- New capability beyond reorg: generic `convertBasis<Target>`, orthogonal transcendental policy, sparse
  support for orthogonal/Mixed schemes. These are recorded as future feature work, not part of the reorg.
- Re-litigating the basis-generic `Expansion<T,Basis,Scheme,Storage>` redesign — it is the foundation.

---

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Reorganization approach | **B — feature-module**: group by capability (`expansion/ bases/ eigen/ io/`); kernels become a private `expansion/detail/`. |
| D2 | Carrier noun | **`Expansion` everywhere**: rename the `*Series` aliases to `*Expansion`; reserve "series" for the printing facility (`tax::series()`, `io/`). |
| D3 | `derivative()` accessor | **Keep** `derivative()`/`derivative<>()` (the `k!`-scaled value form). No `partial()` rename. |
| D4 | Alias families | **Keep both**: terse Taylor shorthands `TE/STE/NE/MTE` + basis-generic `*Expansion`. Delete redundant `TEn`; rename the `MixedTE` trap → `BoxTE`. |
| D5 | Eigen layer | Directory becomes **`eigen/`**; namespace **stays `tax::la`** (intentional dir/namespace divergence, zero API churn). |
| D6 | Namespaces | Make **`tax::la` and `tax::named` inline**; **`tax::mixed` non-inline** (factory disambiguation). Delete the repeated `using`-re-export blocks. |
| D7 | Umbrella | One umbrella `<tax/tax.hpp>` + three self-complete sub-facades (`expansion.hpp`, `bases.hpp`, `eigen.hpp`). |

---

## 3. Target layout (feature-module)

Top-level directories map to capabilities; dependencies point strictly downward
(`io → eigen → bases → expansion → expansion/detail`).

```
include/tax/
  tax.hpp              # umbrella = expansion.hpp + bases.hpp + eigen.hpp + io/series.hpp
  version.hpp          # NEW: TAX_VERSION_MAJOR/MINOR/PATCH + TAX_VERSION string
  expansion.hpp        # FACADE: core capability (carrier + named/mixed + ops + default Taylor basis)
  bases.hpp            # FACADE: orthogonal basis policies + spectral extras  (was series.hpp)
  eigen.hpp            # FACADE: Eigen integration, self-contained  (was la.hpp)

  expansion/                       # THE core capability
    concepts.hpp                   # Scalar, TaylorPolynomial, DensePolynomial (+ carrier concept)
    meta.hpp                       # NEW: FixedString, compareFixed, TypeList, Prepend, generic fold
    multi_index.hpp  enumeration.hpp
    basis.hpp                      # MOVED from series/: BasisPolicy concept (renamed from Basis)
    taylor_basis.hpp               # MOVED from series/: TaylorBasis default monomial policy
    scheme.hpp  scheme/{concept,isotropic,mixed}.hpp
    storage/{dense,sparse}.hpp
    expansion.hpp                  # carrier Expansion<T,Basis,Scheme,Storage>; NO eigen/ or bases/ includes
    batch.hpp                      # Batch<T,K>; self-contained NumTraits<Batch>
    axis.hpp                       # NEW: Axis, OrderedAxis, axis meta (Offset/Dim/IsCanonical/IsSubsetOf/
                                   #      buildAxisMap), ONE policy-parameterised Merge, embed/slice remap helpers
    named.hpp                      # NamedExpansion (tax::named)
    mixed.hpp                      # MixedExpansion (tax::mixed)  (was mixed_named.hpp)
    promote.hpp
    detail/                        # PRIVATE kernels (was kernels/)
      stencil_config.hpp           # NEW: TAX_USE_UNROLL/STENCIL/STENCIL_MAX_BYTES single home
      cauchy.hpp  cauchy_unroll.hpp  cauchy_stencil.hpp  recurrence_stencil.hpp  mixed_stencils.hpp
      algebra.hpp  trigonometric.hpp  transcendental.hpp  sparse_cauchy.hpp  sparse_subs.hpp
    ops/                           # public free-function surface (was operators/)
      arithmetic.hpp  math_unary.hpp  math_binary.hpp  sparse.hpp
      named_arithmetic.hpp  named_math_unary.hpp  named_math_binary.hpp
      mixed_arithmetic.hpp  mixed_math_unary.hpp  mixed_math_binary.hpp   # NEW: lifted out of the type header
      unary_functions.def          # NEW: centralized unary-fn table

  bases/   (was series/)
    ortho.hpp                      # OrthogonalBasis<Derived> CRTP + 3-term-recurrence engine (namespace fixed)
    chebyshev.hpp  legendre.hpp  hermite.hpp   (was *_basis.hpp)
    chebyshev_interp.hpp  chebyshev_math.hpp  convert.hpp   # spectral extras
    ops.hpp                        # integer pow for non-Taylor bases  (was series/operators.hpp)
    aliases.hpp                    # orthogonal aliases: ChebyshevExpansion/LegendreExpansion/HermiteExpansion
                                   #   + generic Series<Basis,N,M=1,T>.  (TaylorExpansion lives in expansion/)

  eigen/   (was la/, namespace stays tax::la)
    types.hpp                      # Vec/Mat/VecNT/MatNT/MatNMT  (no longer included by the carrier)
    traits.hpp                     # NEW: unified te_traits + is_te/is_named/is_mixed + rebind
    num_traits.hpp                 # the 3 Expansion NumTraits via one shared factory base (Batch excluded)
    expansion_vectors.hpp          # TEVec/NEVec/MTEVec
    values.hpp  derivatives.hpp  axis_diff.hpp
    named_diff.hpp                 # MERGED named + mixed per-axis differential surface
    truncate.hpp  invert.hpp

  io/
    series.hpp                     # ONE basis-generic streamer (folds in the old series/io.hpp)
```

Deleted: `kernels/` (→ `expansion/detail/`), `operators/` (→ `expansion/ops/`), `la/` (→ `eigen/`),
`core/` (→ `expansion/`), `series/io.hpp` (→ `io/series.hpp`), `core/taylor_expansion.hpp` shim, `Doxyfile`.

### Public include story
- `<tax/tax.hpp>` — everything (unchanged contract: the recommended include).
- `<tax/expansion.hpp>` — carrier + named/mixed + operators + Taylor basis, no orthogonal bases, no Eigen helpers.
- `<tax/bases.hpp>` — adds the orthogonal basis policies + spectral extras.
- `<tax/eigen.hpp>` — the Eigen integration; self-complete (includes mixed; fixes today's `la.hpp` gap).

Honest caveat: `expansion.hpp` still transitively pulls Eigen (via `batch.hpp` and io), so it is "core types
without orthogonal bases," **not** "Eigen-free." A no-Eigen core is out of scope (Non-goals).

---

## 4. Namespace scheme

- **`tax`** — all public types and the carrier.
- **`tax::la`** *(inline)* — the Eigen helpers (`gradient`/`hessian`/`jacobian`/`value`/`eval`/`invert`/
  `variables`/`truncate`). Being inline, every helper is also a first-class `tax::` name with ADL intact, so
  `tax::gradient(te)` and `tax::la::gradient(te)` are the **same** entity. The directory is `eigen/` but the
  namespace remains `tax::la` (D5) — a deliberate, documented divergence to keep downstream `tax::la::` code
  compiling unchanged.
- **`tax::named`** *(inline)* — `NamedExpansion`, named factories, named differential ops. Inline ⇒ first-class
  `tax::` names; the repeated `using named::…;` re-export blocks are deleted.
- **`tax::mixed`** *(non-inline)* — `MixedExpansion`, `OrderedAxis`, and the mixed `variable`/`variables`
  factories. **Must stay non-inline**: `named::variable<"x",N>(T)` and `mixed::variable<"x",Order>(double)`
  have identical leading-template + call shapes, so inlining both into `tax::` makes `tax::variable<"x",4>(1.0)`
  genuinely ambiguous. The mixed *types* (not factories) are surfaced into `tax::` via explicit
  `using mixed::MixedExpansion; using mixed::MTE;` (type names don't collide).
- **`tax::detail`, `tax::detail::kernels`** — internals (the orthogonal recurrence engine's namespace is
  corrected to `tax::detail::kernels`).

Implementation note: with `tax::la` inline, the scalar `tax::value` overload must be *defined inside* `tax::la`
so `tax::value` and `tax::la::value` are one entity (not a redefinition).

---

## 5. Naming changes (rename table)

| From | To | Rationale |
|------|----|-----------|
| `Basis` concept (`series/basis.hpp`) | `BasisPolicy` | stop shadowing the ubiquitous `typename Basis` param (the `tax::Basis<Basis>` workaround disappears) |
| orthogonal `*Series` (`ChebyshevSeries`, `LegendreSeries`, `HermiteSeries`) | `*Expansion` (`ChebyshevExpansion`, …) | D2 — one carrier noun; reserve "series" for printing |
| `TaylorSeries`, `NamedSeries` | *(delete — redundant)* | `TaylorExpansion` and `NamedExpansion` already exist for these |
| `Series<Basis,N,T>` (univariate-only) | `Series<Basis,N,M=1,T>` | regularize arity to match the per-basis aliases (generic spelling retained) |
| `TEn<N,M>` | *(delete)* | fully redundant with `TE<N,M>` |
| `MixedTE<Groups...>` (unnamed anisotropic alias) | `BoxTE<Groups...>` | removes the `MixedTE`/`MTE`-name-trap; not `MixedTaylorSeries` (recreates the trap) |
| `core/` dir | `expansion/` | feature-module (D1) |
| `kernels/` dir | `expansion/detail/` | kernels become private (D1) |
| `operators/` dir | `expansion/ops/` | operators belong to the expansion capability |
| `series/` dir, `series.hpp` | `bases/`, `bases.hpp` | "series" freed for the printing facility |
| `la/` dir, `la.hpp` | `eigen/`, `eigen.hpp` | capability-named dir (namespace stays `tax::la`, D5) |
| `*_basis.hpp` (chebyshev/legendre/hermite) | `chebyshev.hpp` / `legendre.hpp` / `hermite.hpp` | dir already says "basis" |
| `mixed_named.hpp` / `MixedTaylorExpansion` | `mixed.hpp` / `MixedExpansion` | shorter, matches `tax::mixed`; `MTE` retained as terse alias |

**Not renamed** (critique-rejected): `coefficients()` (collides with `std::array::data()` semantics if shortened);
`derivative()` (D3); the terse `TE/STE/NE/MTE` family is retained (D4). `NamedTaylorExpansion` is kept as a
thin Taylor convenience alias of `NamedExpansion` only if still used after the sweep; `NamedSeries` is dropped.

**Terse-alias placement:** each terse alias is co-located with its type — `TE`/`TEn`(→deleted)/`STE`/`BoxTE` in
`expansion/expansion.hpp`, `NE` in `expansion/named.hpp`, `MTE` in `expansion/mixed.hpp`. `TaylorExpansion`
(carrier's Taylor alias) also lives in `expansion/`; only the orthogonal/generic aliases live in `bases/aliases.hpp`.

---

## 6. Correctness & edge fixes (independent of the tree)

These are carried regardless of layout and several precede the move so the safety net lands first.

- **F1 — `k!`-member correctness.** `Expansion::gradient()`, `hessian()`, and the value-form
  `derivative()`/`derivative<>()` apply `k!` scaling but are unconstrained on `Basis`, so they silently return
  wrong numbers for orthogonal bases. **Constrain all to `requires std::is_same_v<Basis, TaylorBasis>`.** Keep
  them (15+ test call sites). Also gate `invert`/`identityMap`/`composeOne` on the Taylor trait.
- **F2 — sever `core→eigen`.** Spell the carrier members' return types as plain
  `Eigen::Matrix<T, Scheme::vars, 1>` / `<…, Scheme::vars>` and drop `#include <eigen/types.hpp>` from the
  carrier (Eigen is already a hard core dep via `batch.hpp`). No API change (identical to today's `VecNT`/`MatNT`).
- **F3 — sever `core→bases`.** Move `BasisPolicy` and `TaylorBasis` into `expansion/`; the carrier needs the
  Taylor policy + Cauchy product, which is a legitimate downward `expansion → expansion/detail` edge.
- **F4 — mixed operators out of the type header.** Move the inlined mixed operator surface into
  `expansion/ops/mixed_*.hpp` (symmetry with named); bring `mixed_math_binary` to full parity
  (`pow(x,int/real)`, `pow(MTE,MTE)`, `pow(scalar,MTE)`, all `atan2` forms) and **backfill the same forms into
  `named_math_binary`** (it is itself incomplete today).
- **F5 — re-export dance.** Inline `tax::la` and `tax::named` (D6); delete the repeated `using`-blocks; this
  also fixes the asymmetry where `tax::gradient<"x">(ne)` resolves but `tax::gradient(plain_te)` did not.
- **F6 — unified printing.** Fold `series/io.hpp` into `io/series.hpp`; extend the basis render hook to carry a
  variable symbol (`term(int k, std::string_view var)`) so multivariate orthogonal and `NamedExpansion`-over-
  orthogonal (which print nowhere today) get `operator<<`; constrain the unconstrained `to_string(const F&)`.
- **F7 — packaging.** Add `version.hpp`; add a CI job that `cmake --install`s then `find_package(tax CONFIG)`
  + compiles a 5-line consumer; set package compatibility to `SameMinorVersion`; delete dead artifacts
  (`Doxyfile`, the `core/taylor_expansion.hpp` shim, phantom `pyproject`/Python-binding references); fix the
  `basis.hpp` 2-arg-vs-3-arg `derivative/integral` doc drift.
- **F8 — kernel/meta tidies.** Single `expansion/detail/stencil_config.hpp` for the `TAX_USE_*` macros (delete
  the factually-wrong "headers include each other" comment); delete the two dead `<N,M>` shims in `algebra.hpp`;
  consolidate sparse operators into `expansion/ops/sparse.hpp`; central `unary_functions.def` (now that mixed is
  the 5th consumer); collapse the `Merge`/`MergeOrdered` families into one policy-parameterised template.
  **Drop** the `AxisCarrier` CRTP idea (the named/mixed twins are not real twins — named is basis-generic,
  mixed Taylor-only, different public surfaces); share only the *meta* and the embed/slice remap helpers.

---

## 7. WONTFIX / capability matrix (ratify; do not "fix" later)

Document these as an explicit internals capability matrix rather than leaving latent surprises:
- **Kernel↔Scheme product-dispatch round-trip** (`Scheme::cauchyProduct`): the deliberate customization seam of
  the basis-generic redesign. Keep.
- **`MixedScheme`↔`mixed_stencils` forward-decl cycle:** documented, benign, within-template. Re-keying on a
  per-group descriptor array would duplicate the sacred graded-box ordering (invariant hazard). Keep.
- **Sparse storage is `IsotropicScheme`-Taylor-only:** no sparse path for orthogonal or Mixed schemes. Record.
- **Multivariate orthogonal is intentionally second-class:** for M≥2, orthogonal bases support only
  `{+,-,*,pow,deriv,integ,eval}` (no division, transcendentals, convert; printing added by F6). Record.

Hard invariants preserved throughout: graded-lex `flatIndex` layout untouched; all moved code stays `constexpr`
and `std::array`-only in the dense core; kernel config macros stay in-header and identical project-wide (ODR);
`M ≥ 1` asserts unchanged.

---

## 8. Phased migration

Each phase builds + passes the full `ctest` suite in the mamba `tax` env. "Mechanical" = compile-verifiable
path/namespace moves; "Delicate" = semantics or test-string assertions can shift. Ordered so the safety net and
correctness fixes land before the wide tree move.

- **P0 — Packaging + CI safety net (mechanical).** `version.hpp`; install/`find_package` CI smoke test;
  `SameMinorVersion`; delete dead artifacts; fix `basis.hpp` doc drift. *(F7)*
- **P1 — `core→eigen` cut + `k!`-constraint (mechanical edit + correctness gate).** *(F1, F2)* — turns prior
  silent misuse into a compile error; verify orthogonal tests don't rely on it.
- **P2 — `BasisPolicy` + `TaylorBasis` into the core capability; rename concept; regularize `Series` arity;
  demote `aliases.hpp` to a pure-alias file (mechanical, wide; one atomic commit).** *(F3, naming)*
- **P3 — extract `meta.hpp` + `axis.hpp`; collapse `Merge` (3a mechanical, 3b delicate compile-time meta).** *(F8)*
- **P4 — mixed operators → `ops/`; full binary parity (named + mixed); move mixed type into `tax::mixed`.** *(F4)*
- **P5 — eigen consolidation + inline namespaces.** Merge named+mixed la surface → `named_diff.hpp`; one shared
  `ExpansionNumTraits` factory (Batch excluded); unify `traits.hpp`; inline `tax::la`/`tax::named`, delete the
  `using`-blocks; self-complete `eigen.hpp`. *(F5)*
- **P6 — THE tree move (mechanical, wide; one atomic compile-verified commit).** `core/→expansion/`,
  `kernels/→expansion/detail/`, `operators/→expansion/ops/`, `series/→bases/`, `la/→eigen/`; add the three
  facades + shrink `tax.hpp`. The ODR-sensitive `stencil_config.hpp` keeps the macro order intact.
- **P7 — unified printing (delicate; test-string churn).** *(F6)*
- **P8 — kernel/meta tidies (low-risk).** `stencil_config.hpp`, dead-shim deletions, `ops/sparse.hpp`,
  `unary_functions.def`. *(F8 remainder)*
- **P9 — naming sweep (mechanical, wide blast radius; one rename per commit).** `*Series→*Expansion`,
  delete `TEn`, `MixedTE→BoxTE`, `mixed_named→mixed`/`MixedTaylorExpansion→MixedExpansion`. Land last.

**Deferred (feature work, not reorg):** generic `convertBasis<Target>`; Chebyshev-as-`OrthogonalBasis` via a
`toCanonical` hook; orthogonal transcendental policy. Scheduled independently.

---

## 9. Risks & open items

- **Wide mechanical sweeps (P2, P6, P9)** touch many files; mitigated by doing each as a single atomic,
  compile-verified commit and keeping `ctest` green per phase. P6 is the highest-churn step (the cost the
  critics flagged for approach B); accepted deliberately for the cleaner capability-grouped tree.
- **Inline-namespace subtlety (P5):** `tax::value` must be defined inside `tax::la`; ADL behavior of the now-
  `tax::`-level helpers must be re-verified against existing call sites.
- **Q-batching (minor, deferred):** `TE<N,M,K>`, `TaylorExpansion<…Batch<double,K>>`, and `Batchd<K>` are three
  spellings of a batched Taylor type. Pick one recommended spelling and document the others as equivalents —
  low urgency, folded into P9 docs.
- **Companion `ode/ads` plugin** must adapt to moved include paths and the `*Series→*Expansion` renames;
  `tax::la::` and `derivative()` are preserved to limit its churn.

---

## 10. Verification

- Each phase: `cmake --build build -j` + `ctest --test-dir build -j` green in the mamba `tax` env.
- New tests: orthogonal-basis `k!`-member misuse is now ill-formed (compile-fail test or static check);
  multivariate-orthogonal and named-over-orthogonal printing (F6); install/`find_package` consumer (F7);
  mixed `pow`/`atan2` parity (F4).
- `clang-format` on touched files, preserving the repo's indented-PP convention.
- No new heap in the dense core; `constexpr` surface intact.
