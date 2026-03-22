#pragma once

#include <tax/ads/ads_node.hpp>
#include <cassert>
#include <deque>
#include <span>
#include <vector>

namespace tax
{

/**
 * @brief Arena-based binary tree for Adaptive Domain Splitting (ADS).
 *
 * All nodes live in a contiguous `std::vector` (the "arena").  Children are
 * referenced by integer indices so no pointer invalidation occurs when the
 * arena grows via `push_back`.
 *
 * ### Typical propagation loop
 * @code
 *   AdsTree<TEn<4,3>> tree;
 *   tree.add_leaf(initial_tte, initial_box);
 *
 *   while (!tree.empty()) {
 *       int idx = tree.pop();
 *       auto& lf = tree.node(idx).leaf();
 *       if (needs_split(lf.tte)) {
 *           auto [lt, rt] = compute_children(lf.tte, lf.box);
 *           tree.split(idx, best_dim(lf.tte), std::move(lt), std::move(rt));
 *       } else {
 *           tree.mark_done(idx);
 *       }
 *   }
 * @endcode
 *
 * ### Iterating results
 * @code
 *   for (int i : tree.done_leaves())
 *       process(tree.node(i).leaf());
 * @endcode
 *
 * ### Point lookup
 * @code
 *   int idx = tree.find_leaf(pt);   // O(depth)
 *   if (idx >= 0) evaluate(tree.node(idx).leaf().tte, pt);
 * @endcode
 *
 * @tparam TTE  Truncated Taylor expansion type (e.g. tax::TEn<N, M>).
 */
template < typename TTE >
class AdsTree
{
   public:
    using Node = AdsNode< TTE >;
    using T    = typename TTE::Input::value_type;
    static constexpr int M = int( std::tuple_size_v< typename TTE::Input > );

    // =========================================================================
    // Initialization
    // =========================================================================

    /**
     * @brief Add a root leaf and enqueue it for processing.
     *
     * May be called multiple times to initialize with several independent
     * subdomains (multi-root tree).
     *
     * @return Arena index of the newly created leaf.
     */
    int add_leaf( TTE tte, Box< T, M > box )
    {
        const int idx = int( nodes_.size() );
        Node      n;
        n.parent_idx = -1;
        n.data       = typename Node::Leaf{ std::move( tte ),
                                      std::move( box ),
                                      /*done=*/false,
                                      /*leaves_pos=*/int( leaf_list_.size() ) };
        nodes_.push_back( std::move( n ) );
        leaf_list_.push_back( idx );
        work_queue_.push_back( idx );
        roots_.push_back( idx );
        return idx;
    }

    // =========================================================================
    // Work-queue interface
    // =========================================================================

    /// True when the work queue is empty (propagation complete).
    [[nodiscard]] bool empty() const noexcept { return work_queue_.empty(); }

    /// Index of the next leaf to process.
    [[nodiscard]] int front() const { return work_queue_.front(); }

    /**
     * @brief Pop the front leaf from the work queue.
     * @return Arena index of the popped node (still exists in the arena as a leaf).
     */
    int pop()
    {
        const int idx = work_queue_.front();
        work_queue_.pop_front();
        return idx;
    }

    // =========================================================================
    // Leaf operations
    // =========================================================================

    /**
     * @brief Mark a leaf as done.
     *
     * Call this after popping the leaf and determining it does not need
     * splitting.  The leaf remains in `leaf_list_` (it is still a leaf of the
     * tree); it is additionally recorded in `done_leaves_`.
     *
     * @param idx  Arena index of the leaf (must be a leaf node).
     */
    void mark_done( int idx )
    {
        assert( nodes_[idx].is_leaf() );
        nodes_[idx].leaf().done = true;
        done_leaves_.push_back( idx );
    }

    /**
     * @brief Split a leaf into two children and enqueue both.
     *
     * The node at @p idx is converted to an Internal node.  Two new Leaf nodes
     * are appended to the arena and pushed to the back of the work queue.
     * `leaf_list_` is updated in O(1) via swap-and-pop.
     *
     * @param idx        Arena index of the leaf to split (must be a leaf).
     * @param dim        Dimension along which to split (0-based).
     * @param left_tte   TTE for the left  (lower center[dim]) child.
     * @param right_tte  TTE for the right (upper center[dim]) child.
     * @return {left_idx, right_idx} — arena indices of the two new children.
     */
    std::pair< int, int > split( int idx, int dim, TTE left_tte, TTE right_tte )
    {
        assert( nodes_[idx].is_leaf() );

        // 1. Capture everything from nodes_[idx] before any push_back that
        //    might reallocate the arena.
        const auto [left_box, right_box] = nodes_[idx].leaf().box.split( dim );
        const T   sv        = nodes_[idx].leaf().box.center[dim];
        const int orig_pos  = nodes_[idx].leaf().leaves_pos;

        // 2. Remove idx from leaf_list_ via O(1) swap-and-pop.
        //    Safe: no arena mutation yet, so all node references are valid.
        const int last_leaf                        = leaf_list_.back();
        leaf_list_[orig_pos]                       = last_leaf;
        nodes_[last_leaf].leaf().leaves_pos        = orig_pos;
        leaf_list_.pop_back();

        // 3. Append left child.
        const int left_idx = int( nodes_.size() );
        {
            Node n;
            n.parent_idx = idx;
            n.data = typename Node::Leaf{ std::move( left_tte ),
                                          left_box,
                                          /*done=*/false,
                                          /*leaves_pos=*/int( leaf_list_.size() ) };
            nodes_.push_back( std::move( n ) );
            leaf_list_.push_back( left_idx );
            work_queue_.push_back( left_idx );
        }

        // 4. Append right child.
        const int right_idx = int( nodes_.size() );
        {
            Node n;
            n.parent_idx = idx;
            n.data = typename Node::Leaf{ std::move( right_tte ),
                                          right_box,
                                          /*done=*/false,
                                          /*leaves_pos=*/int( leaf_list_.size() ) };
            nodes_.push_back( std::move( n ) );
            leaf_list_.push_back( right_idx );
            work_queue_.push_back( right_idx );
        }

        // 5. Convert the split node to Internal (safe: arena is stable after
        //    push_backs complete; index-based access does not dangle).
        nodes_[idx].data =
            typename Node::Internal{ dim, sv, left_idx, right_idx };

        return { left_idx, right_idx };
    }

    // =========================================================================
    // Point lookup
    // =========================================================================

    /**
     * @brief Walk the tree to find the leaf whose box contains @p pt.
     *
     * For multi-root trees (multiple calls to `add_leaf` before any split),
     * all roots are searched in order.
     *
     * @return Arena index of the matching leaf, or -1 if not found.
     * Complexity: O(depth) per root.
     */
    [[nodiscard]] int find_leaf( const std::array< T, M >& pt ) const
    {
        for ( int r : roots_ )
        {
            const int idx = find_from( r, pt );
            if ( idx >= 0 ) return idx;
        }
        return -1;
    }

    /**
     * @brief Walk the subtree rooted at @p start to find the leaf containing @p pt.
     * @return Arena index of the matching leaf, or -1 if not found.
     */
    [[nodiscard]] int find_from( int start, const std::array< T, M >& pt ) const
    {
        int idx = start;
        while ( true )
        {
            if ( nodes_[idx].is_leaf() )
                return nodes_[idx].leaf().box.contains( pt ) ? idx : -1;

            const auto& in = nodes_[idx].internal();
            idx = ( pt[in.split_dim] <= in.split_value ) ? in.left_idx : in.right_idx;
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    [[nodiscard]] Node&       node( int idx )       noexcept { return nodes_[idx]; }
    [[nodiscard]] const Node& node( int idx ) const noexcept { return nodes_[idx]; }

    /// All current leaf indices (active + done; no internal nodes).
    [[nodiscard]] std::span< const int > leaf_list()    const noexcept { return leaf_list_; }
    /// Leaf indices marked done via mark_done().
    [[nodiscard]] const std::vector< int >& done_leaves() const noexcept { return done_leaves_; }
    /// Root indices (nodes with parent_idx == -1).
    [[nodiscard]] const std::vector< int >& roots()       const noexcept { return roots_; }

    [[nodiscard]] int num_nodes()  const noexcept { return int( nodes_.size() ); }
    [[nodiscard]] int num_leaves() const noexcept { return int( leaf_list_.size() ); }
    [[nodiscard]] int num_done()   const noexcept { return int( done_leaves_.size() ); }
    [[nodiscard]] int num_active() const noexcept { return int( work_queue_.size() ); }

   private:
    std::vector< Node > nodes_;        ///< arena: all nodes, root(s) at lowest indices
    std::deque< int >   work_queue_;   ///< indices of unprocessed (active) leaves
    std::vector< int >  leaf_list_;    ///< indices of all current leaf nodes (O(1) updates)
    std::vector< int >  done_leaves_;  ///< subset of leaf_list_ marked as done
    std::vector< int >  roots_;        ///< root node indices (parent_idx == -1)
};

}  // namespace tax
