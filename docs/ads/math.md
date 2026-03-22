# Mathematical Foundations

This page describes the mathematical theory behind the Automatic Domain
Splitting (ADS) algorithm implemented in the `tax` library, following
Wittig et al. (2015).

## Problem Statement

Given a function \(f : \mathbb{R}^M \to \mathbb{R}\) and an axis-aligned
domain

\[
\Omega = [a_1, b_1] \times [a_2, b_2] \times \cdots \times [a_M, b_M],
\]

approximate \(f\) by a **piecewise polynomial** such that the approximation
error on each subdomain is below a user-specified tolerance
\(\varepsilon\).

A single truncated Taylor polynomial of order \(N\) may not achieve this
accuracy when \(\Omega\) is large. ADS adaptively subdivides \(\Omega\)
into smaller boxes until every piece satisfies the error bound.

## Normalised Variables

Each subdomain is an axis-aligned box with center \(\mathbf{c}\) and
half-widths \(\mathbf{h}\). The physical coordinates \(\mathbf{x}\) are
related to normalised variables \(\boldsymbol{\delta} \in [-1, 1]^M\) by

\[
x_k = c_k + h_k \, \delta_k, \qquad k = 1, \ldots, M.
\]

The polynomial approximation on this subdomain is expressed in the
normalised variables:

\[
f(\mathbf{x}) \approx \sum_{|\alpha| \le N} p_\alpha \, \delta^{\alpha},
\]

where \(\alpha = (\alpha_1, \ldots, \alpha_M)\) is a multi-index,
\(|\alpha| = \alpha_1 + \cdots + \alpha_M\) is its total degree, and
\(\delta^{\alpha} = \delta_1^{\alpha_1} \cdots \delta_M^{\alpha_M}\).

Working in normalised variables ensures that the polynomial coefficients
directly reflect the function's behavior on \([-1, 1]^M\), making the
truncation-error estimate meaningful regardless of the physical scale of
the subdomain.

## Truncation Error Estimate

The quality of the polynomial approximation is estimated from the
**highest-degree coefficients**. If the Taylor series is converging, the
degree-\(N\) coefficients should be small. The truncation error is
estimated as their infinity norm:

\[
\varepsilon_{\text{trunc}} = \max_{|\alpha| = N} |p_\alpha|.
\]

This is the key quantity in the ADS algorithm: if
\(\varepsilon_{\text{trunc}} < \varepsilon\), the subdomain is accepted;
otherwise, it must be split.

**Interpretation:** Large degree-\(N\) coefficients indicate that the
polynomial has not yet converged -- the series still has significant
energy at the truncation boundary. Small coefficients indicate rapid
convergence and a good approximation.

## Split Dimension Selection

When a subdomain must be split, the algorithm chooses the dimension that
contributes most to the truncation error. For each variable \(k\), a
score is computed by summing the absolute values of all degree-\(N\)
coefficients that involve variable \(k\):

\[
s_k = \sum_{\substack{|\alpha| = N \\ \alpha_k > 0}} |p_\alpha|.
\]

The split dimension is then

\[
k^* = \arg\max_k \; s_k.
\]

This is the heuristic proposed by Wittig et al. (2015): splitting along
the variable that "most affects" the highest-order terms is the most
efficient way to reduce the truncation error.

## Domain Bisection

When a subdomain with center \(\mathbf{c}\) and half-widths \(\mathbf{h}\)
fails the tolerance check, it is bisected along dimension \(k^*\) at its
center \(c_{k^*}\). This produces two child boxes:

**Left child** (lower half along dimension \(k^*\)):

\[
\text{center}_{k^*}^{(\text{left})} = c_{k^*} - \frac{h_{k^*}}{2},
\qquad
h_{k^*}^{(\text{left})} = \frac{h_{k^*}}{2}.
\]

The interval covered is \([c_{k^*} - h_{k^*},\; c_{k^*}]\).

**Right child** (upper half along dimension \(k^*\)):

\[
\text{center}_{k^*}^{(\text{right})} = c_{k^*} + \frac{h_{k^*}}{2},
\qquad
h_{k^*}^{(\text{right})} = \frac{h_{k^*}}{2}.
\]

The interval covered is \([c_{k^*},\; c_{k^*} + h_{k^*}]\).

All other dimensions remain unchanged. Each child box is then
re-evaluated with a fresh degree-\(N\) polynomial and tested against the
tolerance.

## The ADS Algorithm

The full algorithm proceeds as follows:

```
1.  Evaluate the degree-N polynomial on the initial domain.
2.  Insert the initial domain into the work queue.
3.  While the work queue is not empty:
    a.  Pop a subdomain from the queue.
    b.  Estimate the truncation error: epsilon_trunc = max |p_alpha| for |alpha| = N.
    c.  If epsilon_trunc < tolerance  OR  depth >= max_depth:
            Mark as done (accept this subdomain).
    d.  Else:
            Compute scores s_k for each dimension k.
            Pick split dimension k* = argmax_k s_k.
            Bisect the box along dimension k*.
            Evaluate degree-N polynomials on both child boxes.
            Push both children into the work queue.
4.  Return the tree of accepted subdomains.
```

The work queue uses BFS ordering (FIFO), so all subdomains at a given
depth are processed before moving to the next level.

The `max_depth` parameter provides a hard limit on the number of
bisections, preventing unbounded recursion for functions that are
difficult to approximate (e.g., functions with singularities near the
domain).

## Convergence

For analytic functions, each bisection halves the domain width in one
dimension. Since the normalised variables \(\delta_k \in [-1, 1]\) are
rescaled to the new half-width, the degree-\(N\) coefficient of a smooth
function on the halved domain is typically reduced by a factor of

\[
2^{-N}.
\]

This means the algorithm converges **geometrically**: after \(d\)
bisections, the truncation error is roughly

\[
\varepsilon_{\text{trunc}} \sim \varepsilon_0 \cdot 2^{-Nd},
\]

where \(\varepsilon_0\) is the initial truncation error. For a
polynomial of order \(N = 10\), each bisection reduces the error by a
factor of approximately \(2^{-10} \approx 10^{-3}\), so only a few
levels of splitting are typically needed.

Functions with limited regularity (e.g., finite differentiability) will
converge more slowly, and the `max_depth` parameter ensures termination
in all cases.

## Point Lookup

Given a query point \(\mathbf{x}\), the containing subdomain is found by
walking the binary tree from the root:

1. At an **internal node** with split dimension \(k^*\) and split value
   \(v\), compare \(x_{k^*}\) against \(v\):
    - If \(x_{k^*} \le v\), descend to the left child.
    - If \(x_{k^*} > v\), descend to the right child.
2. At a **leaf node**, verify that the leaf's box contains \(\mathbf{x}\)
   and return it.

The complexity is \(O(\text{depth})\), where depth is the number of
bisections from the root to the leaf.

For multi-root trees (when multiple independent initial domains are used),
each root is searched in sequence until a containing leaf is found.

## Reference

A. Wittig, P. Di Lizia, R. Armellin, et al., "Propagation of large
uncertainty sets in orbital dynamics by automatic domain splitting,"
*Celestial Mechanics and Dynamical Astronomy*, vol. 122, pp. 239--261, 2015.
