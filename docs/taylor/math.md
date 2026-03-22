# Mathematical Foundations

This page describes the mathematical theory behind the Taylor ODE integrator
implemented in the `tax::ode` module.

## Taylor Method for ODEs

Consider an initial-value problem (IVP):

\[
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, t), \qquad \mathbf{x}(t_0) = \mathbf{x}_0.
\]

The Taylor method constructs the time-Taylor polynomial of the solution about
\(t_0\):

\[
\mathbf{x}(t_0 + \tau) = \sum_{k=0}^{N} \mathbf{x}^{[k]} \tau^k
\]

where \(\mathbf{x}^{[k]}\) are the **normalised Taylor coefficients**:

\[
\mathbf{x}^{[k]} = \frac{1}{k!} \frac{d^k \mathbf{x}}{dt^k}\bigg|_{t_0}.
\]

The truncation order \(N\) is a compile-time parameter. Typical values are
\(N = 15\)--\(25\); the high order enables very large time steps compared to
classical methods of order 4--8.

## Picard Iteration via Automatic Differentiation

The Taylor coefficients are computed by propagating DA arithmetic through the
right-hand side \(\mathbf{f}\). Starting from \(\mathbf{x}^{[0]} = \mathbf{x}_0\),
the \(k\)-th coefficient is obtained from:

\[
\mathbf{x}^{[k+1]} = \frac{1}{k+1} \mathbf{f}^{[k]}
\]

where \(\mathbf{f}^{[k]}\) is the \(k\)-th Taylor coefficient of
\(\mathbf{f}(\mathbf{x}(t), t)\). Because the library represents time \(t\) as
a univariate Taylor expansion \(t = t_0 + \tau\), evaluating \(\mathbf{f}\)
with DA arithmetic automatically produces the full series
\(\mathbf{f}^{[0]}, \mathbf{f}^{[1]}, \ldots, \mathbf{f}^{[N-1]}\) via the
chain rule. Each iteration extends the known Taylor series by one order.

In the implementation, the loop in `step.hpp` performs exactly \(N\) evaluations
of \(\mathbf{f}\), each time reading off the newly available coefficient and
dividing by \(k+1\).

## Step-Size Control

The integrator uses the **Jorba--Zou (2005)** criterion to select the step size
adaptively. Given the Taylor polynomial of order \(N\) and an absolute tolerance
\(\varepsilon\), the step size is:

\[
h = \min\!\left(
    \left(\frac{\varepsilon}{|\mathbf{x}^{[N-1]}|}\right)^{\!1/(N-1)},\;
    \left(\frac{\varepsilon}{|\mathbf{x}^{[N]}|}\right)^{\!1/N}
\right).
\]

This estimates the radius of convergence of the Taylor series: if the last
coefficients are small, the series converges over a wide interval and a large
step is safe. For **vector systems**, the minimum step size across all state
components is used:

\[
h = \min_{i} \; h_i
\]

where \(h_i\) is the Jorba--Zou estimate for state component \(i\).

The criterion is evaluated in `stepsize.hpp`. If a coefficient is exactly zero,
the corresponding term does not constrain the step size (infinity is returned for
that term).

## Dense Output

The Taylor polynomial computed at each step provides **continuous dense output**
at no extra cost. For any \(t \in [t_n, t_{n+1}]\), the solution is:

\[
\mathbf{x}(t) = \sum_{k=0}^{N} \mathbf{x}_n^{[k]}\, (t - t_n)^k
\]

evaluated using Horner's method for numerical stability. This is exact to order
\(N\) within the convergence radius of the Taylor series.

The `TaylorSolution::operator()` performs a binary search over the stored step
times to locate the correct interval, then evaluates the corresponding polynomial.
Both forward and backward integration are supported.

## DA-Expanded Integration

Instead of propagating a single initial condition \(\mathbf{x}_0 \in \mathbb{R}^D\),
the state can be a **Differential Algebra (DA) polynomial** expanded around a
nominal trajectory. The state becomes:

\[
\mathbf{x}_k = \text{TEn<P, D>}
\]

where \(P\) is the DA order in the initial-condition variables and \(D\) is the
state dimension. Each state component is a multivariate polynomial in normalised
deviations \(\boldsymbol{\delta} \in [-1, 1]^D\).

The time-Taylor coefficients are now **DA-valued**: each \(\mathbf{x}^{[k]}\)
is itself a polynomial in \(\boldsymbol{\delta}\). The Picard iteration proceeds
identically, but all arithmetic (addition, multiplication, division, square root,
etc.) operates on DA polynomials via the `tax` library's truncated algebra.

The initial conditions are constructed from a `Box<double, D>` via `makeDaState`:

\[
x_i = c_i + h_i \, \delta_i, \qquad i = 1, \ldots, D
\]

where \(c_i\) is the centre, \(h_i\) is the half-width, and \(\delta_i\) is the
\(i\)-th DA variable.

## DA Step-Size Control

The generalised Jorba--Zou criterion replaces the scalar absolute value with the
**infinity norm** of the DA polynomial coefficients:

\[
h = \min\!\left(
    \left(\frac{\varepsilon}{\|\mathbf{x}^{[N-1]}\|_\infty}\right)^{\!1/(N-1)},\;
    \left(\frac{\varepsilon}{\|\mathbf{x}^{[N]}\|_\infty}\right)^{\!1/N}
\right)
\]

where

\[
\|p\|_\infty = \max_{|\alpha| \le P} |p_\alpha|
\]

is the maximum absolute coefficient of the DA polynomial \(p\). For vector
systems, the minimum across all state components is used, as in the scalar case.

This ensures that the time step is conservative enough for the entire family of
initial conditions represented by the DA polynomial, not just the nominal
trajectory.

## ADS-ODE Integration

For large initial-condition domains, a single DA polynomial of moderate order
\(P\) may not accurately represent the flow map. The **Automatic Domain Splitting
(ADS)** algorithm (Wittig et al. 2015) addresses this by recursively bisecting
the domain where the polynomial approximation degrades.

The algorithm proceeds as follows:

1. **Propagate** the initial-condition box from \(t_0\) to \(t_{\max}\) using
   DA-expanded integration, yielding a polynomial flow map.

2. **Evaluate the truncation error:**

    \[
    \varepsilon_{\text{trunc}} = \max_{\substack{|\alpha| = P \\ j = 1,\ldots,D}} |(\mathbf{x}_j)_\alpha|
    \]

    This is the largest degree-\(P\) coefficient across all state components. A
    large value indicates that the polynomial is losing accuracy and needs more
    resolution.

3. **If** \(\varepsilon_{\text{trunc}} < \varepsilon_{\text{ADS}}\) (the ADS
   tolerance) **or** the maximum split depth is reached, **accept** the
   polynomial and mark the leaf as done.

4. **Otherwise, split** the domain along the variable that contributes most to
   the degree-\(P\) truncation error. For each variable \(k\), the score is:

    \[
    s_k = \sum_{\substack{|\alpha| = P \\ \alpha_k > 0}} \sum_{j=1}^{D} |(\mathbf{x}_j)_\alpha|
    \]

    The dimension with the highest score is bisected.

5. **Propagate** each half independently and repeat from step 2.

The result is an `AdsTree<FlowMap<P,D>>` whose done leaves contain piecewise
polynomial flow maps covering the entire initial-condition domain.

## Flow Map

The DA propagation yields the **flow map** \(\Phi_{t_0}^{t_f}\). Given initial
conditions parameterised as

\[
\mathbf{x}_0 = \mathbf{c} + H\,\boldsymbol{\delta}, \qquad \boldsymbol{\delta} \in [-1, 1]^D
\]

where \(\mathbf{c}\) is the box centre and \(H = \mathrm{diag}(h_1, \ldots, h_D)\)
is the half-width matrix, the final state is:

\[
\mathbf{x}(t_f) = \sum_{|\alpha| \le P} (\Phi_i)_\alpha \, \boldsymbol{\delta}^\alpha
\]

for each state component \(i\). Evaluating the polynomial at a specific
\(\boldsymbol{\delta}\) recovers the propagated state for the corresponding
initial condition, without re-integrating the ODE.

For the ADS case, each leaf has its own local box and polynomial. To evaluate at
a global initial condition \(\mathbf{x}_0\):

1. Find the leaf whose box contains \(\mathbf{x}_0\) (via `tree.findLeaf`).
2. Compute the local normalised deviation:
   \(\delta_k = (x_{0,k} - c_k^{\text{leaf}}) / h_k^{\text{leaf}}\).
3. Evaluate the leaf's flow map polynomial at \(\boldsymbol{\delta}\).
