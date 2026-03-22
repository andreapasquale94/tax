# API Reference

Complete API documentation for the Automatic Domain Splitting (ADS) module.

---

## Box\<T, M\>

**Header:** `tax/ads/box.hpp`

An axis-aligned hyperrectangle (box) in \(M\) dimensions, represented as the
Cartesian product of closed intervals
\([c_k - h_k,\; c_k + h_k]\) for each dimension \(k\).

### Template Parameters

| Parameter | Description |
|-----------|-------------|
| `T` | Scalar type (e.g., `double`, `float`) |
| `M` | Number of dimensions (must be \(\ge 1\)) |

### Data Members

| Member | Type | Description |
|--------|------|-------------|
| `center` | `std::array<T, M>` | Center point of the box |
| `halfWidth` | `std::array<T, M>` | Half-width along each dimension |

### Member Functions

#### `contains`

```cpp
[[nodiscard]] bool contains(const std::array<T, M>& pt) const noexcept;
```

Returns `true` if `pt` lies inside or on the boundary of the box.

#### `split`

```cpp
[[nodiscard]] std::pair<Box, Box> split(int dim) const noexcept;
```

Bisects the box at its midpoint along dimension `dim`. Returns a pair
`{left, right}` where the left child covers the lower half and the right
child covers the upper half. Both children have half the original
half-width in dimension `dim`; all other dimensions are unchanged.

#### `splitValue`

```cpp
[[nodiscard]] T splitValue(int dim) const noexcept;
```

Returns `center[dim]`, i.e., the boundary value where a split along
dimension `dim` would occur.

### Example

```cpp
tax::Box<double, 2> box{{0.0, 1.0}, {3.0, 2.0}};
// Represents [-3, 3] x [-1, 3]

bool inside = box.contains({1.0, 2.0});     // true
auto [left, right] = box.split(0);          // split along x
// left:  center={-1.5, 1.0}, halfWidth={1.5, 2.0}  -> [-3, 0] x [-1, 3]
// right: center={ 1.5, 1.0}, halfWidth={1.5, 2.0}  -> [ 0, 3] x [-1, 3]
```

---

## AdsNode\<TTE\>

**Header:** `tax/ads/ads_node.hpp`

A node in the ADS arena tree. Each node is either a **Leaf** (holding a
polynomial and its subdomain) or an **Internal** node (created when a leaf
is split, holding split metadata and child indices).

### Template Parameters

| Parameter | Description |
|-----------|-------------|
| `TTE` | Truncated Taylor expansion type (e.g., `tax::TEn<N, M>`) |

### Nested Types

#### `AdsNode::Leaf`

| Member | Type | Description |
|--------|------|-------------|
| `tte` | `TTE` | Polynomial approximation on this subdomain |
| `box` | `Box<T, M>` | The subdomain this polynomial covers |
| `done` | `bool` | `true` once this leaf has been accepted (default: `false`) |
| `leavesPos` | `int` | Index of this node in `AdsTree::leafList_` (internal bookkeeping) |

#### `AdsNode::Internal`

| Member | Type | Description |
|--------|------|-------------|
| `splitDim` | `int` | Dimension along which this node was split |
| `splitValue` | `T` | Boundary value (center of the former leaf along `splitDim`) |
| `leftIdx` | `int` | Arena index of the left (lower) child |
| `rightIdx` | `int` | Arena index of the right (upper) child |

### Data Members

| Member | Type | Description |
|--------|------|-------------|
| `parentIdx` | `int` | Arena index of the parent node (`-1` for root nodes) |
| `data` | `std::variant<Leaf, Internal>` | The node payload |

### Member Functions

#### Type Queries

```cpp
[[nodiscard]] bool isLeaf() const noexcept;
[[nodiscard]] bool isInternal() const noexcept;
```

#### Accessors

```cpp
[[nodiscard]] Leaf&           leaf() noexcept;
[[nodiscard]] const Leaf&     leaf() const noexcept;
[[nodiscard]] Internal&       internal() noexcept;
[[nodiscard]] const Internal& internal() const noexcept;
```

Access the `Leaf` or `Internal` payload. Calling `leaf()` on an internal
node (or vice versa) is undefined behavior (calls `std::get` on the wrong
variant alternative).

---

## AdsTree\<TTE\>

**Header:** `tax/ads/ads_tree.hpp`

Arena-based binary tree for Automatic Domain Splitting. All nodes are stored
in a contiguous `std::vector` and referenced by integer index, so no pointer
invalidation occurs when the arena grows.

### Template Parameters

| Parameter | Description |
|-----------|-------------|
| `TTE` | Truncated Taylor expansion type (e.g., `tax::TEn<N, M>`) |

### Type Aliases

| Alias | Type |
|-------|------|
| `Node` | `AdsNode<TTE>` |
| `T` | Scalar type extracted from `TTE` |
| `M` | Number of variables extracted from `TTE` |

### Initialization

#### `addLeaf`

```cpp
int addLeaf(TTE tte, Box<T, M> box);
```

Creates a new root leaf node with the given polynomial and subdomain box.
The leaf is added to the arena, appended to the leaf list, and enqueued
in the work queue. May be called multiple times to create a multi-root tree.

**Returns:** Arena index of the new leaf.

### Work Queue

#### `empty`

```cpp
[[nodiscard]] bool empty() const noexcept;
```

Returns `true` when the work queue is empty (all subdomains have been
processed).

#### `front`

```cpp
[[nodiscard]] int front() const;
```

Returns the arena index of the next leaf to process, without removing it
from the queue.

#### `pop`

```cpp
int pop();
```

Removes and returns the arena index of the front leaf in the work queue.

### Leaf Operations

#### `markDone`

```cpp
void markDone(int idx);
```

Marks the leaf at arena index `idx` as done (accepted). The leaf remains
in the leaf list and is additionally recorded in the done-leaves list.

**Precondition:** `node(idx).isLeaf()`.

#### `split`

```cpp
std::pair<int, int> split(int idx, int dim, TTE left_tte, TTE right_tte);
```

Splits the leaf at arena index `idx` along dimension `dim`. The original
leaf is converted to an internal node. Two new leaf nodes are created
from the bisected box and the provided polynomials, then enqueued for
processing.

The original leaf is removed from the leaf list via \(O(1)\) swap-and-pop.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `idx` | Arena index of the leaf to split |
| `dim` | Dimension to bisect along (0-based) |
| `left_tte` | Polynomial for the left (lower) child |
| `right_tte` | Polynomial for the right (upper) child |

**Returns:** `{left_idx, right_idx}` -- arena indices of the two new children.

**Precondition:** `node(idx).isLeaf()`.

### Point Lookup

#### `findLeaf`

```cpp
[[nodiscard]] int findLeaf(const std::array<T, M>& pt) const;
```

Walks the tree to find the leaf whose box contains `pt`. For multi-root
trees, all roots are searched in order.

**Returns:** Arena index of the matching leaf, or `-1` if not found.

**Complexity:** \(O(\text{depth})\) per root.

#### `findFrom`

```cpp
[[nodiscard]] int findFrom(int start, const std::array<T, M>& pt) const;
```

Walks the subtree rooted at arena index `start` to find the leaf
containing `pt`.

**Returns:** Arena index of the matching leaf, or `-1` if not found.

### Accessors

#### `node`

```cpp
[[nodiscard]] Node&       node(int idx) noexcept;
[[nodiscard]] const Node& node(int idx) const noexcept;
```

Returns the node at arena index `idx`.

#### `leafList`

```cpp
[[nodiscard]] std::span<const int> leafList() const noexcept;
```

Returns all current leaf indices (both active and done; no internal nodes).

#### `doneLeaves`

```cpp
[[nodiscard]] const std::vector<int>& doneLeaves() const noexcept;
```

Returns the indices of leaves that have been marked done via `markDone()`.

#### `roots`

```cpp
[[nodiscard]] const std::vector<int>& roots() const noexcept;
```

Returns the root node indices (nodes with `parentIdx == -1`).

### Statistics

```cpp
[[nodiscard]] int numNodes()  const noexcept;  // total nodes in the arena
[[nodiscard]] int numLeaves() const noexcept;  // current leaf count
[[nodiscard]] int numDone()   const noexcept;  // leaves marked done
[[nodiscard]] int numActive() const noexcept;  // leaves remaining in the work queue
```

---

## AdsRunner\<N, M, F\>

**Header:** `tax/ads/ads_runner.hpp`

The ADS algorithm driver. Evaluates a user-supplied function on subdomains,
estimates truncation error, and splits subdomains that do not meet the
tolerance.

### Template Parameters

| Parameter | Description |
|-----------|-------------|
| `N` | DA (polynomial) order |
| `M` | Number of variables |
| `F` | Callable type |

### Type Aliases

| Alias | Type |
|-------|------|
| `TTE` | `TEn<N, M>` |
| `Tree` | `AdsTree<TTE>` |

### Constructor

```cpp
AdsRunner(F func, double tolerance, int maxDepth = 30);
```

| Parameter | Description |
|-----------|-------------|
| `func` | Function to approximate. Takes \(M\) DA variables and returns an expression. |
| `tolerance` | Maximum allowed truncation-error norm per subdomain. |
| `maxDepth` | Maximum number of bisections from root to any leaf (default: 30). |

!!! warning "Callable signature"
    The callable `F` **must** take its arguments by `const&`. Expression
    template nodes store leaf TTEs by reference. A by-value parameter would
    be destroyed when the function returns its lazy expression, causing
    undefined behavior.

    ```cpp
    // Correct: arguments by const reference
    auto f = [](const auto& x, const auto& y) { return sin(x) * cos(y); };

    // WRONG: arguments by value — will dangle!
    auto g = [](auto x, auto y) { return sin(x) * cos(y); };
    ```

### Member Functions

#### `run`

```cpp
Tree run(Box<double, M> initial_box);
```

Runs the ADS algorithm on the given initial domain. Returns the completed
tree with all subdomains either accepted (done) or split to `maxDepth`.

### Factory Function

#### `makeAdsRunner`

```cpp
template <int N, int M, typename F>
AdsRunner<N, M, F> makeAdsRunner(F func, double tol, int maxDepth = 30);
```

Convenience factory that deduces `F` from the argument. Equivalent to
constructing `AdsRunner<N, M, decltype(func)>` directly.

```cpp
auto runner = tax::makeAdsRunner<10, 1>(f, 1e-5);
auto tree = runner.run(domain);
```
