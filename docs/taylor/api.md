# API Reference

Complete reference for the public functions and types in the `tax::ode` module.

## Template Parameters

The following template parameters appear throughout the API:

| Parameter | Meaning | Typical values |
|-----------|---------|----------------|
| `N` | Taylor expansion order in time | 15--25 |
| `P` | DA expansion order in initial-condition variables | 1--4 |
| `D` | State-space dimension (number of state variables / DA variables) | 1--6 |
| `T` | Scalar coefficient type | `double` |
| `F` | Right-hand side callable type | lambda or function object |

## Step Functions

**Header:** `tax/ode/step.hpp`

These functions compute a single adaptive Taylor step and return the solution
polynomial together with a recommended step size.

### StepResult

```cpp
template <typename Poly, typename T>
struct StepResult
{
    Poly p;   // Taylor polynomial of the solution, centred at the current time.
    T h;      // Recommended step-size magnitude (positive).
};
```

### Scalar step

```cpp
template <int N, typename F, typename T = double>
StepResult<TruncatedTaylorExpansionT<T, N, 1>, T>
step(F&& f, T x0, T tc, T abstol);
```

Compute one Taylor step for a scalar ODE \(\dot{x} = f(x, t)\).

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `f` | callable | Right-hand side with signature `f(x, t) -> dx/dt` |
| `x0` | `T` | Current state value |
| `tc` | `T` | Current time |
| `abstol` | `T` | Absolute tolerance for step-size control |

**Returns:** `StepResult` containing the Taylor polynomial `p` (evaluate at
displacement \(\tau\) via `p.eval(tau)`) and the recommended step size `h`.

### Vector step

```cpp
template <int N, typename F, typename T, int D>
StepResult<Eigen::Matrix<TruncatedTaylorExpansionT<T, N, 1>, D, 1>, T>
step(F&& f, const Eigen::Matrix<T, D, 1>& x0, T tc, T abstol);
```

Compute one Taylor step for a vector ODE where `f` writes derivatives into its
first argument.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `f` | callable | Right-hand side with signature `f(dx, x, t)` (writes into `dx`) |
| `x0` | `Eigen::Matrix<T,D,1>` | Current state vector |
| `tc` | `T` | Current time |
| `abstol` | `T` | Absolute tolerance for step-size control |

**Returns:** `StepResult` containing the polynomial vector `p` (evaluate at
displacement \(\tau\) via `tax::eval(p, tau)`) and the recommended step size `h`.

## Step-Size Control

**Header:** `tax/ode/stepsize.hpp`

### Scalar step size

```cpp
template <typename T, int N>
T stepsize(const TruncatedTaylorExpansionT<T, N, 1>& x, T abstol) noexcept;
```

Compute the adaptive step size from the last two Taylor coefficients using the
Jorba--Zou (2005) criterion. Returns the recommended step-size magnitude
(always positive). If a trailing coefficient is exactly zero, that term does not
constrain the step size.

### Vector step size

```cpp
template <typename T, int N, int D>
T stepsize(
    const Eigen::Matrix<TruncatedTaylorExpansionT<T, N, 1>, D, 1>& x,
    T abstol) noexcept;
```

Minimum step size across all state components.

## Integration

**Header:** `tax/ode/integrate.hpp`

All `integrate` overloads return a `TaylorSolution` (see below). They support
both forward integration (\(t_0 < t_{\max}\)) and backward integration
(\(t_0 > t_{\max}\)).

### Scalar ODE, adaptive stepping

```cpp
template <int N, typename F, typename T = double>
TaylorSolution<N, T, T>
integrate(F&& f, T x0, T t0, T tmax, T abstol, int maxsteps = 500);
```

Integrate the scalar ODE \(\dot{x} = f(x, t)\) from \(t_0\) to \(t_{\max}\)
with adaptive step-size control.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `f` | callable | `f(x, t) -> dx/dt` |
| `x0` | `T` | Initial state |
| `t0` | `T` | Initial time |
| `tmax` | `T` | Final time |
| `abstol` | `T` | Absolute tolerance |
| `maxsteps` | `int` | Maximum number of steps (default 500) |

### Scalar ODE, specified output times

```cpp
template <int N, typename F, typename T = double>
TaylorSolution<N, T, T>
integrate(F&& f, T x0, const std::vector<T>& trange, T abstol,
          int maxsteps = 500);
```

Integrate and record the solution at the times listed in `trange`. The first
element of `trange` is the initial time; the sequence must be monotonic. The
returned solution has `sol.t == trange` and `sol.x[i]` is the state at
`trange[i]`.

### Vector ODE, adaptive stepping

```cpp
template <int N, typename F, typename T, int D>
TaylorSolution<N, Eigen::Matrix<T, D, 1>, T>
integrate(F&& f, const Eigen::Matrix<T, D, 1>& x0, T t0, T tmax, T abstol,
          int maxsteps = 500);
```

Integrate the vector ODE where `f` has signature `f(dx, x, t)` and writes the
derivatives into its first argument.

### Vector ODE, specified output times

```cpp
template <int N, typename F, typename T, int D>
TaylorSolution<N, Eigen::Matrix<T, D, 1>, T>
integrate(F&& f, const Eigen::Matrix<T, D, 1>& x0,
          const std::vector<T>& trange, T abstol, int maxsteps = 500);
```

Vector equivalent of the specified-output-times overload.

## TaylorSolution

**Header:** `tax/ode/solution.hpp`

```cpp
template <int N, typename State, typename T = double>
struct TaylorSolution
{
    using Poly = /* TE<N> for scalar, Eigen::Matrix<TE<N>,D,1> for vector */;

    std::vector<T> t;          // Step times (monotonic).
    std::vector<State> x;      // State at each step time.
    std::vector<Poly> p;       // Taylor polynomials centred at each step time.

    State operator()(T time) const;   // Dense-output evaluation.
};
```

### Members

| Member | Type | Description |
|--------|------|-------------|
| `t` | `std::vector<T>` | Monotonic sequence of step times |
| `x` | `std::vector<State>` | State values at each step time |
| `p` | `std::vector<Poly>` | Taylor polynomials for dense output |

### Dense output: `operator()`

```cpp
State operator()(T time) const;
```

Evaluate the solution at an arbitrary time within \([t_0, t_{\max}]\). Uses
binary search to locate the step interval containing `time`, then evaluates
the corresponding Taylor polynomial via Horner's method.

**Throws:** `std::out_of_range` if no polynomials are stored (i.e. when using
the specified-output-times overload, which does not store polynomials).

## DA Step

**Header:** `tax/ode/integrate_ads.hpp`

### stepDa

```cpp
template <int N, int P, int D, typename F>
StepResult<Eigen::Matrix<TruncatedTaylorExpansionT<TEn<P, D>, N, 1>, D, 1>, double>
stepDa(F&& f, const Eigen::Matrix<TEn<P, D>, D, 1>& x0, double tc, double abstol);
```

Compute one Taylor step for a vector ODE with DA-expanded state. The state
components are multivariate polynomials (`TEn<P,D>`) representing a
neighbourhood of initial conditions. Time remains a plain scalar.

The step-size control uses the generalised Jorba--Zou criterion with the infinity
norm of the DA polynomial coefficients (see
[DA Step-Size Control](math.md#da-step-size-control)).

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `f` | callable | `f(dx, x, t)` with DA-valued state vectors |
| `x0` | `Eigen::Matrix<TEn<P,D>, D, 1>` | Current DA state vector |
| `tc` | `double` | Current time (scalar) |
| `abstol` | `double` | Absolute tolerance |

### makeDaState

```cpp
template <int P, int D>
Eigen::Matrix<TEn<P, D>, D, 1>
makeDaState(const Box<double, D>& box);
```

Build DA-expanded initial conditions from a `Box`. Component \(i\) becomes the
polynomial \(c_i + h_i \, \delta_i\) where \(\boldsymbol{\delta} \in [-1,1]^D\)
is the normalised deviation.

### propagateBox

```cpp
template <int N, int P, int D, typename F>
Eigen::Matrix<TEn<P, D>, D, 1>
propagateBox(F&& f, const Box<double, D>& box, double t0, double tmax,
             double abstol, int maxsteps = 500);
```

Integrate a vector ODE with DA-expanded state over a single subdomain.
Constructs DA initial conditions from `box` via `makeDaState`, then advances
from \(t_0\) to \(t_{\max}\) with adaptive Taylor stepping. Returns the final
DA state vector (the polynomial flow map).

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `f` | callable | `f(dx, x, t)` |
| `box` | `Box<double, D>` | Initial-condition domain (centre + half-widths) |
| `t0` | `double` | Initial time |
| `tmax` | `double` | Final time |
| `abstol` | `double` | Absolute tolerance for time stepping |
| `maxsteps` | `int` | Maximum integration steps (default 500) |

## FlowMap

**Header:** `tax/ode/integrate_ads.hpp`

```cpp
template <int P, int D>
struct FlowMap
{
    using DA    = TEn<P, D>;
    using Input = std::array<double, D>;

    Eigen::Matrix<DA, D, 1> state;
};
```

Polynomial flow map stored in each ADS leaf. The `state` member is a vector of
multivariate polynomials in the normalised initial-condition deviations
\(\boldsymbol{\delta} \in [-1, 1]^D\).

## ADS-ODE Integration

**Header:** `tax/ode/integrate_ads.hpp`

### integrateAds

```cpp
template <int N, int P, typename F, int D>
AdsTree<FlowMap<P, D>>
integrateAds(F&& f, const Box<double, D>& x0_box, double t0, double tmax,
             double step_tol, double ads_tol, int ads_max_depth = 30,
             int maxsteps = 500);
```

Integrate a vector ODE with Automatic Domain Splitting over the initial-condition
domain `x0_box`. If the DA flow map's truncation error exceeds `ads_tol`, the
domain is bisected along the variable that contributes most to the degree-\(P\)
coefficients, and each half is re-propagated independently.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `f` | callable | `f(dx, x, t)` |
| `x0_box` | `Box<double, D>` | Initial-condition domain |
| `t0` | `double` | Initial time |
| `tmax` | `double` | Final time |
| `step_tol` | `double` | Absolute tolerance for adaptive time stepping |
| `ads_tol` | `double` | Truncation-error tolerance for ADS splitting |
| `ads_max_depth` | `int` | Maximum recursive bisection depth (default 30) |
| `maxsteps` | `int` | Maximum integration steps per subdomain (default 500) |

**Returns:** `AdsTree<FlowMap<P,D>>` whose done leaves contain the piecewise
polynomial flow maps. Iterate over results with:

```cpp
for (int i : tree.doneLeaves())
{
    const auto& leaf = tree.node(i).leaf();
    // leaf.tte.state -- DA polynomial flow map
    // leaf.box       -- subdomain of initial conditions
}
```

!!! note "Template parameter order"
    The template parameters are `<N, P, F, D>` where `D` is deduced from the
    `Box` argument. In practice you specify only `N` and `P`:
    `integrateAds<15, 3>(f, box, ...)`.
