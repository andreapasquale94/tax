# Examples

Worked examples demonstrating the ADS module in practice.

---

## Approximating a Gaussian

The classic benchmark from Wittig et al. (2015): approximate
\(f(x) = e^{-x^2}\) on \([-3, 3]\) using a 10th-order polynomial with
a truncation-error tolerance of \(10^{-5}\).

The domain \([-3, 3]\) is represented as a `Box` with center 0 and
half-width 3. The function is too oscillatory (in Taylor coefficient terms)
for a single polynomial to achieve the target accuracy, so ADS
automatically splits the domain into several subdomains.

```cpp
#include <tax/ads.hpp>
#include <iostream>

int main()
{
    // The function to approximate.
    // Arguments MUST be taken by const& (expression nodes store references).
    auto gaussian = [](const auto& x) { return exp(-x * x); };

    // Domain: [-3, 3] = center 0, half-width 3
    tax::Box<double, 1> domain{{0.0}, {3.0}};

    // Build and run the ADS runner: order 10, 1 variable, tolerance 1e-5
    tax::AdsRunner<10, 1, decltype(gaussian)> runner(gaussian, 1e-5, 60);
    auto tree = runner.run(domain);

    std::cout << "Total subdomains: " << tree.numDone() << "\n";
    std::cout << "Total tree nodes: " << tree.numNodes() << "\n";
}
```

The `makeAdsRunner` factory provides a shorter alternative:

```cpp
auto runner = tax::makeAdsRunner<10, 1>(gaussian, 1e-5, 60);
auto tree = runner.run(domain);
```

---

## Point Evaluation

After running ADS, you can evaluate the piecewise polynomial approximation
at any point inside the original domain. Use `findLeaf` to locate the
subdomain containing the query point, then convert to normalised
coordinates and evaluate.

```cpp
#include <tax/ads.hpp>
#include <cmath>
#include <iostream>

int main()
{
    auto gaussian = [](const auto& x) { return exp(-x * x); };

    tax::Box<double, 1> domain{{0.0}, {3.0}};
    auto runner = tax::makeAdsRunner<10, 1>(gaussian, 1e-5, 60);
    auto tree = runner.run(domain);

    // Evaluate the approximation at x = 1.5
    double x = 1.5;
    int idx = tree.findLeaf({x});

    if (idx < 0)
    {
        std::cerr << "Point not found in any subdomain\n";
        return 1;
    }

    const auto& leaf = tree.node(idx).leaf();

    // Convert to normalised variable: delta = (x - center) / halfWidth
    double delta = (x - leaf.box.center[0]) / leaf.box.halfWidth[0];

    // Evaluate the local polynomial at delta
    double approx = leaf.tte.eval(delta);
    double exact  = std::exp(-x * x);

    std::cout << "x       = " << x << "\n";
    std::cout << "approx  = " << approx << "\n";
    std::cout << "exact   = " << exact << "\n";
    std::cout << "error   = " << std::abs(approx - exact) << "\n";
}
```

The point lookup is an \(O(\text{depth})\) binary tree walk: at each
internal node, the query coordinate is compared against the split value
to decide whether to descend left or right.

---

## Multivariate ADS

ADS works with any number of variables. Here we approximate a 2D function
\(f(x, y) = \sin(x) \cos(y)\) on the domain \([-\pi, \pi]^2\).

```cpp
#include <tax/ads.hpp>
#include <cmath>
#include <iostream>

int main()
{
    // 2D function: arguments by const&
    auto f = [](const auto& x, const auto& y) { return sin(x) * cos(y); };

    // Domain: [-pi, pi] x [-pi, pi]
    tax::Box<double, 2> domain{{0.0, 0.0}, {M_PI, M_PI}};

    // Order 8, 2 variables, tolerance 1e-6
    auto runner = tax::makeAdsRunner<8, 2>(f, 1e-6, 30);
    auto tree = runner.run(domain);

    std::cout << "Subdomains: " << tree.numDone() << "\n";

    // Evaluate at (1.0, 0.5)
    int idx = tree.findLeaf({1.0, 0.5});
    if (idx >= 0)
    {
        const auto& leaf = tree.node(idx).leaf();
        double dx = (1.0 - leaf.box.center[0]) / leaf.box.halfWidth[0];
        double dy = (0.5 - leaf.box.center[1]) / leaf.box.halfWidth[1];

        double approx = leaf.tte.eval({dx, dy});
        double exact  = std::sin(1.0) * std::cos(0.5);

        std::cout << "approx = " << approx << "\n";
        std::cout << "exact  = " << exact << "\n";
        std::cout << "error  = " << std::abs(approx - exact) << "\n";
    }
}
```

The split-dimension heuristic automatically determines whether to split
along \(x\) or \(y\) at each level based on which variable contributes
more to the truncation error.

---

## Iterating Over Results

The `doneLeaves()` method returns the arena indices of all accepted
subdomains. This is the primary way to inspect the ADS output.

```cpp
#include <tax/ads.hpp>
#include <iostream>

int main()
{
    auto f = [](const auto& x) { return exp(-x * x); };

    tax::Box<double, 1> domain{{0.0}, {3.0}};
    auto runner = tax::makeAdsRunner<10, 1>(f, 1e-5, 60);
    auto tree = runner.run(domain);

    std::cout << "Subdomain summary:\n";
    for (int idx : tree.doneLeaves())
    {
        const auto& leaf = tree.node(idx).leaf();
        double lo = leaf.box.center[0] - leaf.box.halfWidth[0];
        double hi = leaf.box.center[0] + leaf.box.halfWidth[0];

        std::cout << "  [" << lo << ", " << hi << "]"
                  << "  value at center = " << leaf.tte[0] << "\n";
    }

    // You can also query overall statistics
    std::cout << "\nStatistics:\n";
    std::cout << "  Total nodes:  " << tree.numNodes() << "\n";
    std::cout << "  Leaf nodes:   " << tree.numLeaves() << "\n";
    std::cout << "  Done leaves:  " << tree.numDone() << "\n";
    std::cout << "  Active queue: " << tree.numActive() << "\n";
}
```

The `leafList()` method returns all current leaves (both done and any
still in the work queue), while `doneLeaves()` returns only the accepted
subdomains. After `run()` completes, `numActive()` is always zero and
`leafList()` and `doneLeaves()` contain the same indices.

---

## Custom Splitting Depth

The `maxDepth` parameter controls the maximum number of bisections from the
root to any leaf. This is a safety limit that prevents unbounded
recursion for functions that are difficult to approximate.

A **smaller** `maxDepth` produces fewer subdomains but may leave some
regions with higher approximation error. A **larger** `maxDepth` allows
finer splitting but increases the total number of subdomains and
computation time.

```cpp
#include <tax/ads.hpp>
#include <iostream>

int main()
{
    // A sharp Gaussian that requires deep splitting
    auto f = [](const auto& x) { return exp(-10.0 * x * x); };

    tax::Box<double, 1> domain{{0.0}, {5.0}};

    // Compare different max depths
    for (int depth : {5, 10, 20, 40})
    {
        auto runner = tax::makeAdsRunner<10, 1>(f, 1e-8, depth);
        auto tree = runner.run(domain);

        std::cout << "maxDepth=" << depth
                  << "  subdomains=" << tree.numDone()
                  << "  nodes=" << tree.numNodes() << "\n";
    }
}
```

For the Gaussian benchmark with tolerance \(10^{-5}\) and order \(N = 10\),
each bisection typically reduces the truncation error by a factor of
\(2^{-N} \approx 10^{-3}\), so even a moderate `maxDepth` of 10--20 is
sufficient for most applications. Setting `maxDepth` to 60 is a
conservative choice that accommodates very large or challenging domains.
