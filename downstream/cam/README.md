# cam — Collision Avoidance Maneuver (tax port)

A C++23 port of the C++ propagation core of
[arma1978/multi-impulsive-cam](https://github.com/arma1978/multi-impulsive-cam),
replacing DACE with [tax](../..) for differential-algebra polynomial
propagation.

## Scope

The original repository combines MATLAB optimisation (SCVX / NLP / MILP via
MOSEK and `fmincon`/`intlinprog`) with a C++ DA-based propagator. This port
replaces the C++ DACE propagator with a header-only tax implementation and
keeps a compatible command-line interface so the MATLAB layer could still
orchestrate it (or be re-implemented).

What this port includes:

| Original                                 | Port (tax)                              |
|------------------------------------------|-----------------------------------------|
| `cpp/astroRoutines.h`                    | `include/cam/{anomaly,elements,frames,propagator,collision}.hpp` |
| `cpp/stateProp.cpp`                      | `apps/state_prop.cpp`                   |
| `cpp/stateBackProp.cpp`                  | `apps/state_back_prop.cpp`              |
| `cpp/statePropMultiMapsFullP.cpp`        | `include/cam/multi_impulse.hpp` (`buildLinearMaps`) |
| `src/routines/linearMaps.m`              | `include/cam/multi_impulse.hpp`         |
| `src/routines/linConvexProblem.m`<br/>`src/routines/linConvexSolve.m` (MOSEK SOCP) | `include/cam/optim/socp.hpp` (smoothed projected-gradient SOCP) |
| `src/routines/refConvex.m`<br/>`main/mainSCVX.m`                  | `include/cam/optim/scvx.hpp` + `apps/scvx_main.cpp` |
| `main/mainNLP.m` (`fmincon`)             | `include/cam/optim/nlp.hpp`             |
| `DA::invert` (TCA refinement)            | `cam::findTCA` using `tax::invert`      |
| `AlgebraicVector<DA>`                    | `Eigen::Matrix<tax::TEn<N,M>, 6, 1>`    |
| `AlgebraicMatrix<DA>`                    | `Eigen::Matrix<tax::TEn<N,M>, 3, 3>`    |

What is **not** in this port:

* `mainMILP.m`. The MILP formulation requires `intlinprog`/Gurobi and we
  intentionally keep the C++ port solver-free. Adding it would mean
  fetching Coin-OR/Cbc.
* MATLAB plotting (`mainPlot.m`, `mainBoxPlot.m`). The C++ apps write
  results to plain text in `runtime/`; plot them with whatever tool you
  prefer.
* `mainLargeSimSCVX.m` campaign batch driver. The single-conjunction
  pipeline is functional; loop it externally over a catalogue.
* `statePropMultiMapsFullPRefine.cpp` (true SCVX outer-iteration
  re-propagation). The current C++ SCVX does a single linearisation pass
  around the unperturbed reference — for short manoeuvre windows this is
  often sufficient. Re-propagating around the current solution would
  recover the full SCVX outer loop.
* RK78 numerical propagator. The averaged-J2 analytical propagator is
  provided instead. If you need a numerical integrator, tax already ships
  one (`tax::ode::integrate`).

## Layout

```
downstream/cam/
├── include/cam/
│   ├── cam.hpp              # umbrella header
│   ├── constants.hpp        # mu, rE, J2/J3/J4, scaling
│   ├── linalg.hpp           # Eigen type aliases + vnorm/cons helpers
│   ├── anomaly.hpp          # true <-> ecc <-> mean anomaly conversions
│   ├── elements.hpp         # kep <-> cart, kep <-> delaunay, kep <-> hill, osc <-> mean
│   ├── propagator.hpp       # propKepAn, propJ2An, averagedJ2rhs, rhsJ234
│   ├── frames.hpp           # RTN, encounter plane, B-plane
│   ├── tca.hpp              # findTCA via tax::invert
│   ├── collision.hpp        # Chan and Alfano collision-probability formulas
│   ├── multi_impulse.hpp    # N-impulse propagation + chained STMs
│   └── optim/
│       ├── optim.hpp        # umbrella for the optimiser headers
│       ├── socp.hpp         # smoothed projected-gradient SOCP solver
│       ├── scvx.hpp         # Sequential Convex Optimisation driver
│       └── nlp.hpp          # smoothed-penalty NLP driver
├── apps/
│   ├── state_prop.cpp       # port of stateProp.cpp
│   ├── state_back_prop.cpp  # port of stateBackProp.cpp
│   └── scvx_main.cpp        # port of mainSCVX.m + mainNLP.m
├── tests/
│   ├── test_kepler.cpp      # round-trip sanity (doubles)
│   ├── test_tca.cpp         # DA-based TCA finding
│   └── test_scvx.cpp        # multi-impulse + NLP solver
└── runtime/                 # inputs/outputs for the apps
```

## Build

From the `downstream/cam/` directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or from the repository root, as a subdirectory, by adding to the root
`CMakeLists.txt`:

```cmake
add_subdirectory(downstream/cam)
```

The CMake script auto-detects the sibling tax repository (two directories
up). If it is not found it falls back to `find_package(tax)` and then to
`FetchContent` from GitHub.

## Usage as a library

```cpp
#include <cam/cam.hpp>
#include <Eigen/Dense>
#include <tax/tax.hpp>

using DA = tax::TEn<2, 7>;  // order 2, 6 state + 1 time perturbation

// Build state with DA dependence on the 7 variables
cam::Vec6<DA> x0 = ...;
DA tof = ...;

cam::Vec6<DA> xf = cam::propJ2An(x0, tof, muSc, rESc, cam::EarthPhysics::J2);
cam::Vec6<DA> xfKep = cam::propKepAn(x0, tof, muSc);

// Refine TCA via polynomial map inversion
cam::Vec6<DA> rel;
for (int i = 0; i < 6; ++i) rel[i] = xfPrimary[i] - xfSecondary[i];
DA dt = cam::findTCA<2, 7>(rel);
```

### Optimisation pipeline

```cpp
cam::MultiImpulseConfig cfg;
cfg.nImpulses     = 5;
cfg.dtSeconds     = 60.0;
cfg.t2TCAseconds  = 600.0;

cam::LinearMaps<double> L =
    cam::buildLinearMaps(xs0, xd0, PsRTN, PdRTN, cfg);  // tax-driven STM extraction

cam::optim::SCVXConfig opt;
opt.sqrMahalaTarget = 25.0;     // 5σ avoidance
opt.dvMax           = 0.001;     // 1 m/s per component
auto sol = cam::optim::solveSCVX(xs0, xd0, PsRTN, PdRTN, cfg, opt);
// sol.dv (3N), sol.totalDeltaV, sol.sqrMahalanobis, sol.missDistance
```

## Runtime protocol (apps)

Apps read from and write to the `runtime/` directory. `state_prop` expects
`xd0.dat` and `xs0.dat`; `state_back_prop` expects `xdTCA.dat` and
`xsTCA.dat`. Each file has seven lines: `rx ry rz vx vy vz t` in kilometres,
km/s, and seconds (unscaled).

## License

BSD 3-Clause (same as tax).
