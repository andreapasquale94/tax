# Taylor ODE Integrator

The `tax::ode` module provides an adaptive, high-order Taylor-method integrator for
ordinary differential equations. It supports both scalar and vector ODEs, dense
output, DA-expanded integration over sets of initial conditions, and automatic
domain splitting (ADS) for large initial-condition domains.

## Key Features

- **High-order time stepping.** Taylor orders of \(N = 15\)--\(25\) are typical,
  enabling very large time steps compared to classical Runge--Kutta methods. A
  single step captures the local solution to machine precision over a wide
  interval.

- **Adaptive step-size control.** The Jorba--Zou (2005) criterion estimates the
  convergence radius of the Taylor series from its last two coefficients, giving
  reliable and near-optimal step sizes without trial evaluations.

- **Dense output.** Every integration step produces a Taylor polynomial that can
  be evaluated at any intermediate time, providing continuous output at no extra
  cost.

- **DA-expanded integration.** Instead of propagating a single initial condition,
  the state can be a multivariate polynomial (a `TEn<P,D>`) representing a
  neighbourhood of initial conditions. The result is a polynomial flow map that
  maps any initial condition in the domain to its final state.

- **ADS-integrated propagation.** For large initial-condition domains where a
  single polynomial cannot accurately represent the flow map, the integrator
  automatically splits the domain using Automatic Domain Splitting (Wittig et al.
  2015). The result is a tree of piecewise polynomial flow maps.

## Including the Module

```cpp
#include <tax/ode/taylor_integrator.hpp>
```

Or via the umbrella header:

```cpp
#include <tax/tax.hpp>
```

## Module Headers

| Header | Purpose |
|--------|---------|
| `tax/ode/taylor_integrator.hpp` | Umbrella include for the entire ODE module |
| `tax/ode/step.hpp` | Single Taylor step (scalar and vector) |
| `tax/ode/stepsize.hpp` | Jorba--Zou adaptive step-size control |
| `tax/ode/integrate.hpp` | Full integration loop with adaptive stepping |
| `tax/ode/solution.hpp` | `TaylorSolution` container with dense output via `operator()` |
| `tax/ode/integrate_ads.hpp` | DA-expanded step, box propagation, and ADS-ODE integration |

## Dependencies

The scalar ODE interface has no dependencies beyond the core `tax` library.
The vector ODE interface and all DA/ADS functionality require **Eigen 3.4+**.

## Quick Example

```cpp
#include <tax/tax.hpp>
#include <cmath>
#include <iostream>

int main()
{
    // Scalar ODE: dx/dt = x, x(0) = 1  -->  x(t) = exp(t)
    auto f = [](const auto& x, const auto& t) { return x; };
    auto sol = tax::ode::integrate<25>(f, 1.0, 0.0, 2.0, 1e-16);

    std::cout << "x(2) = " << sol.x.back()
              << "  (exact: " << std::exp(2.0) << ")\n";

    // Dense output at an arbitrary time
    std::cout << "x(1.5) = " << sol(1.5)
              << "  (exact: " << std::exp(1.5) << ")\n";
}
```

## What's Next

- [Mathematical Foundations](math.md) -- the theory behind the Taylor method,
  step-size control, DA expansion, and ADS-ODE integration.
- [API Reference](api.md) -- complete reference for all public functions and
  types in `tax::ode`.
- [Examples](examples.md) -- worked examples covering scalar ODEs, vector
  systems, dense output, Kepler orbits, DA propagation, and ADS splitting.
