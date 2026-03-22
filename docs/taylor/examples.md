# Examples

Worked examples demonstrating the Taylor ODE integrator for scalar ODEs, vector
systems, dense output, DA-expanded propagation, and ADS splitting.

All examples assume the following includes:

```cpp
#include <tax/tax.hpp>
#include <cmath>
#include <iostream>
```

## Scalar ODE: Exponential Growth

The simplest test case: \(\dot{x} = x\), \(x(0) = 1\), whose exact solution is
\(x(t) = e^t\).

```cpp
auto f = [](const auto& x, [[maybe_unused]] const auto& t) { return x; };

auto sol = tax::ode::integrate<25>(f, 1.0, 0.0, 2.0, 1e-16);

std::cout << "x(2)  = " << sol.x.back() << "\n";           // ≈ 7.389056099
std::cout << "exact = " << std::exp(2.0) << "\n";
std::cout << "steps = " << sol.t.size() - 1 << "\n";       // typically 2--3
```

With order \(N = 25\) and tolerance \(10^{-16}\), the integrator reaches
\(t = 2\) in just a few steps while matching `std::exp` to full double
precision.

## Harmonic Oscillator

A 2D vector ODE: \(\dot{x} = v\), \(\dot{v} = -x\), with initial conditions
\(x(0) = 1\), \(v(0) = 0\). The exact solution is \(x(t) = \cos t\),
\(v(t) = -\sin t\).

```cpp
auto f = [](auto& dx, const auto& x, [[maybe_unused]] const auto& t) {
    dx(0) = x(1);
    dx(1) = -x(0);
};

Eigen::Vector2d x0{1.0, 0.0};
auto sol = tax::ode::integrate<25>(f, x0, 0.0, 2 * M_PI, 1e-16);

// After one full period, the state should return to (1, 0)
std::cout << "x(2pi) = " << sol.x.back()(0) << "\n";   // ≈ 1.0
std::cout << "v(2pi) = " << sol.x.back()(1) << "\n";   // ≈ 0.0
```

## Dense Output

The `TaylorSolution` returned by `integrate` supports evaluation at arbitrary
times via `operator()`. The stored Taylor polynomials provide continuous output
without re-integration.

```cpp
auto f = [](const auto& x, [[maybe_unused]] const auto& t) { return x; };
auto sol = tax::ode::integrate<25>(f, 1.0, 0.0, 5.0, 1e-16);

// Evaluate at 100 evenly spaced points
for (int i = 0; i <= 100; ++i)
{
    double t = 5.0 * i / 100.0;
    double x = sol(t);
    double exact = std::exp(t);
    std::cout << "t=" << t << "  x=" << x << "  err=" << std::abs(x - exact) << "\n";
}
```

The dense output is exact to order \(N\) within each step's convergence radius,
so the error is typically at machine precision for well-resolved solutions.

## Output at Specified Times

When you need the solution at specific times (rather than at the integrator's
adaptive step points), use the `trange` overload. This avoids storing Taylor
polynomials and instead interpolates on the fly.

```cpp
auto f = [](const auto& x, [[maybe_unused]] const auto& t) { return x; };

// Output at 11 evenly spaced times from 0 to 2
std::vector<double> trange(11);
for (int i = 0; i <= 10; ++i)
    trange[i] = 0.2 * i;

auto sol = tax::ode::integrate<25>(f, 1.0, trange, 1e-16);

for (std::size_t i = 0; i < sol.t.size(); ++i)
    std::cout << "t=" << sol.t[i] << "  x=" << sol.x[i] << "\n";
```

## Kepler Problem

The two-body problem in 2D: a unit mass orbiting a central body with
gravitational parameter \(\mu = 1\). The state is
\((x, y, \dot{x}, \dot{y})\) and the equations of motion are:

\[
\ddot{x} = -\frac{x}{r^3}, \qquad \ddot{y} = -\frac{y}{r^3}, \qquad r = \sqrt{x^2 + y^2}.
\]

For an initial circular orbit at radius 1, the period is \(2\pi\).

```cpp
auto kepler = [](auto& dx, const auto& x, [[maybe_unused]] const auto& t) {
    auto r2 = x(0) * x(0) + x(1) * x(1);
    auto r3 = r2 * sqrt(r2);
    dx(0) = x(2);
    dx(1) = x(3);
    dx(2) = -x(0) / r3;
    dx(3) = -x(1) / r3;
};

// Circular orbit: x(0)=1, y(0)=0, vx(0)=0, vy(0)=1
Eigen::Vector4d x0{1.0, 0.0, 0.0, 1.0};
auto sol = tax::ode::integrate<25>(kepler, x0, 0.0, 2 * M_PI, 1e-16);

std::cout << "x(2pi) = " << sol.x.back()(0) << "\n";   // ≈ 1.0
std::cout << "y(2pi) = " << sol.x.back()(1) << "\n";   // ≈ 0.0
std::cout << "steps  = " << sol.t.size() - 1 << "\n";
```

The high-order Taylor method typically completes one Kepler orbit in around
10--15 steps at machine precision.

## DA-Expanded Propagation

Instead of propagating a single initial condition, propagate a box of initial
conditions simultaneously. The result is a polynomial flow map: evaluate it at
any normalised deviation \(\boldsymbol{\delta} \in [-1, 1]^D\) to obtain the
final state for the corresponding initial condition.

```cpp
auto f = [](auto& dx, const auto& x, [[maybe_unused]] const auto& t) {
    dx(0) = x(1);
    dx(1) = -x(0);
};

// Box of initial conditions: center (1, 0), half-widths (0.1, 0.1)
tax::Box<double, 2> box{
    .center = {1.0, 0.0},
    .halfWidth = {0.1, 0.1}
};

// Propagate from t=0 to t=1 with DA order P=3
auto xf = tax::ode::propagateBox<20, 3, 2>(f, box, 0.0, 1.0, 1e-16);

// Evaluate at the box centre (delta = 0, 0)
std::cout << "x_centre = " << xf(0).eval({0.0, 0.0}) << "\n";
std::cout << "v_centre = " << xf(1).eval({0.0, 0.0}) << "\n";

// Evaluate at a corner (delta = 0.5, -0.3)
// This corresponds to x0 = 1.0 + 0.1*0.5 = 1.05, v0 = 0.0 + 0.1*(-0.3) = -0.03
std::cout << "x_corner = " << xf(0).eval({0.5, -0.3}) << "\n";
std::cout << "v_corner = " << xf(1).eval({0.5, -0.3}) << "\n";
```

The DA polynomial captures the dependence of the final state on the initial
conditions to order \(P = 3\). This is far cheaper than propagating many
individual trajectories when the domain is moderate in size.

## ADS-ODE Integration

For large initial-condition domains, a single DA polynomial may not be accurate
enough. The ADS-ODE integrator automatically splits the domain where the
polynomial approximation degrades, producing a tree of piecewise flow maps.

```cpp
auto f = [](auto& dx, const auto& x, [[maybe_unused]] const auto& t) {
    dx(0) = x(1);
    dx(1) = -x(0);
};

// Large box of initial conditions
tax::Box<double, 2> box{
    .center = {1.0, 0.0},
    .halfWidth = {0.5, 0.5}
};

// Integrate with ADS: N=15 (time order), P=3 (DA order)
// step_tol=1e-12 (time stepping), ads_tol=1e-4 (splitting threshold)
// max depth=6
auto tree = tax::ode::integrateAds<15, 3>(
    f, box, 0.0, 3.0, 1e-12, 1e-4, 6
);

// Iterate over all leaves
std::cout << "Number of subdomains: " << tree.doneLeaves().size() << "\n";

for (int i : tree.doneLeaves())
{
    const auto& leaf = tree.node(i).leaf();
    const auto& sub_box = leaf.box;
    const auto& flow = leaf.tte.state;

    std::cout << "Subdomain centre: ("
              << sub_box.center[0] << ", " << sub_box.center[1]
              << "), half-widths: ("
              << sub_box.halfWidth[0] << ", " << sub_box.halfWidth[1]
              << ")\n";

    // Evaluate at the subdomain centre
    std::cout << "  x(3) = " << flow(0).eval({0.0, 0.0})
              << ", v(3) = " << flow(1).eval({0.0, 0.0}) << "\n";
}
```

The ADS algorithm ensures that every subdomain's polynomial meets the requested
truncation-error tolerance. Subdomains near the nominal trajectory (where the
dynamics are nearly linear) tend to remain large, while subdomains with stronger
nonlinearity are split more finely.

### Point Evaluation on the ADS Tree

To evaluate the propagated state for a specific initial condition, find the leaf
that contains it, compute the local normalised deviation, and evaluate the
polynomial:

```cpp
// Query: what is x(3) for initial condition (1.2, 0.1)?
std::array<double, 2> query_ic = {1.2, 0.1};

int leaf_idx = tree.findLeaf(query_ic);
const auto& leaf = tree.node(leaf_idx).leaf();

// Compute normalised deviation: delta_i = (x0_i - center_i) / halfWidth_i
std::array<double, 2> delta;
for (int k = 0; k < 2; ++k)
    delta[k] = (query_ic[k] - leaf.box.center[k]) / leaf.box.halfWidth[k];

std::cout << "x(3) at (1.2, 0.1) = " << leaf.tte.state(0).eval(delta) << "\n";
std::cout << "v(3) at (1.2, 0.1) = " << leaf.tte.state(1).eval(delta) << "\n";
```

This evaluation is \(O(\text{depth})\) for the tree walk plus \(O(\binom{P+D}{D})\)
for the polynomial evaluation -- negligible compared to re-integrating the ODE.
