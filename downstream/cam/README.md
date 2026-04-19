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

| Original (DACE)                 | Port (tax)                              |
|---------------------------------|-----------------------------------------|
| `cpp/astroRoutines.h`           | `include/cam/{anomaly,elements,frames,propagator,collision}.hpp` |
| `cpp/stateProp.cpp`             | `apps/state_prop.cpp`                   |
| `cpp/stateBackProp.cpp`         | `apps/state_back_prop.cpp`              |
| `DA::invert` (TCA refinement)   | `cam::findTCA` using `tax::invert`      |
| `AlgebraicVector<DA>`           | `Eigen::Matrix<tax::TEn<N,M>, 6, 1>`    |
| `AlgebraicMatrix<DA>`           | `Eigen::Matrix<tax::TEn<N,M>, 3, 3>`    |

What is **not** in this port:

* MATLAB drivers (`mainSCVX.m`, `mainNLP.m`, `mainMILP.m`, ...). These depend
  on MOSEK 11 and MATLAB R2025b and are out of scope for a header-only C++
  port. The port exposes the same `runtime/` file protocol so an external
  driver could still call the executables.
* `statePropMultiMapsFull*.cpp` and Keplerian-only variants — they are close
  variants of `stateProp.cpp`; once you have one of them you have them all.
  Trivial to add on top of this library.
* RK78 numerical propagator. The averaged-J2 analytical propagator is
  provided instead (the same one the original uses as `propJ2An`). If you
  need a numerical integrator, tax already ships one (`tax::ode::integrate`).

## Layout

```
downstream/cam/
├── include/cam/
│   ├── cam.hpp          # umbrella header
│   ├── constants.hpp    # mu, rE, J2/J3/J4, scaling
│   ├── linalg.hpp       # Eigen type aliases + vnorm/cons helpers
│   ├── anomaly.hpp      # true <-> ecc <-> mean anomaly conversions
│   ├── elements.hpp     # kep <-> cart, kep <-> delaunay, kep <-> hill, osc <-> mean
│   ├── propagator.hpp   # propKepAn, propJ2An, averagedJ2rhs, rhsJ234
│   ├── frames.hpp       # RTN, encounter plane, B-plane
│   ├── tca.hpp          # findTCA via tax::invert
│   └── collision.hpp    # Chan and Alfano collision-probability formulas
├── apps/
│   ├── state_prop.cpp       # port of stateProp.cpp
│   └── state_back_prop.cpp  # port of stateBackProp.cpp
├── tests/
│   ├── test_kepler.cpp  # round-trip sanity (doubles)
│   └── test_tca.cpp     # DA-based TCA finding
└── runtime/             # inputs/outputs for the apps
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

## Runtime protocol (apps)

Apps read from and write to the `runtime/` directory. `state_prop` expects
`xd0.dat` and `xs0.dat`; `state_back_prop` expects `xdTCA.dat` and
`xsTCA.dat`. Each file has seven lines: `rx ry rz vx vy vz t` in kilometres,
km/s, and seconds (unscaled).

## License

BSD 3-Clause (same as tax).
