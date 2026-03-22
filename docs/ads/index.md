# Automatic Domain Splitting (ADS)

The ADS module implements the **Automatic Domain Splitting** algorithm from
Wittig et al. (2015) for adaptive piecewise polynomial approximation over
large domains.

## Motivation

A single truncated Taylor polynomial of order \(N\) centered at a point
provides an accurate approximation only in a neighborhood of that point.
When the domain of interest is too large for a single polynomial to meet a
desired accuracy, ADS **recursively bisects** the domain until every
subdomain's polynomial satisfies a truncation-error tolerance.

The result is a binary tree of subdomains, each carrying a degree-\(N\)
polynomial that approximates the target function to the requested precision.

## How It Works

1. Start with the full domain and evaluate a degree-\(N\) polynomial on it.
2. Estimate the truncation error from the highest-degree coefficients.
3. If the error is below the tolerance, accept the subdomain (mark it done).
4. Otherwise, pick the dimension that contributes most to the error, bisect
   the domain along that dimension, evaluate polynomials on both halves, and
   enqueue them for processing.
5. Repeat until the work queue is empty.

The algorithm uses a **BFS work queue** so that all subdomains at a given
depth are processed before moving deeper.

## Data Structure

ADS uses an **arena-based binary tree**:

- All nodes live in a contiguous `std::vector` (the arena), referenced by
  integer index. No pointer invalidation occurs when the arena grows.
- A `std::deque` serves as the BFS work queue of unprocessed leaf indices.
- Leaf removal from the leaf list is \(O(1)\) via swap-and-pop.
- Point lookup (finding which subdomain contains a given point) is
  \(O(\text{depth})\) via a binary tree walk.

## Key Components

| Component | Header | Description |
|-----------|--------|-------------|
| `Box<T, M>` | `tax/ads/box.hpp` | Axis-aligned hyperrectangle in \(M\) dimensions |
| `AdsNode<TTE>` | `tax/ads/ads_node.hpp` | Tree node: either a `Leaf` (polynomial + subdomain) or an `Internal` node (split metadata + child indices) |
| `AdsTree<TTE>` | `tax/ads/ads_tree.hpp` | Arena-based binary tree with work queue, leaf tracking, and point lookup |
| `AdsRunner<N, M, F>` | `tax/ads/ads_runner.hpp` | The ADS algorithm driver: evaluates the function, estimates error, splits as needed |

## Include Header

```cpp
#include <tax/ads.hpp>
```

This facade header pulls in all ADS components: `Box`, `AdsNode`, `AdsTree`,
and `AdsRunner`.

## Quick Example

```cpp
#include <tax/ads.hpp>

// Approximate f(x) = exp(-x^2) on [-3, 3]
auto f = [](const auto& x) { return exp(-x * x); };

tax::Box<double, 1> domain{{0.0}, {3.0}};
tax::AdsRunner<10, 1, decltype(f)> runner(f, 1e-5, 60);
auto tree = runner.run(domain);

std::cout << "Subdomains: " << tree.numDone() << "\n";

// Point evaluation
int idx = tree.findLeaf({1.5});
const auto& leaf = tree.node(idx).leaf();
double delta = (1.5 - leaf.box.center[0]) / leaf.box.halfWidth[0];
double approx = leaf.tte.eval(delta);
```

## Further Reading

- [Mathematical Foundations](math.md) -- the theory behind ADS
- [API Reference](api.md) -- complete class and function documentation
- [Examples](examples.md) -- worked examples with full code

## Reference

A. Wittig, P. Di Lizia, R. Armellin, et al., "Propagation of large
uncertainty sets in orbital dynamics by automatic domain splitting,"
*Celestial Mechanics and Dynamical Astronomy*, vol. 122, pp. 239--261, 2015.
