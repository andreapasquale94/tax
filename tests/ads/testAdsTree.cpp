#include "testUtils.hpp"
#include <tax/ads.hpp>

#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Type aliases used throughout.
// ---------------------------------------------------------------------------
using TTE  = tax::TEn< 2, 2 >;
using Tree = tax::AdsTree< TTE >;
using Box2 = tax::Box< double, 2 >;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Unit box: center={0,0}, halfWidth={1,1}  →  [-1,1]^2
static Box2 unit_box() { return Box2{ { 0.0, 0.0 }, { 1.0, 1.0 } }; }

/// Constant TTE (all non-constant coefficients zero).
static TTE const_tte( double v ) { return TTE{ v }; }

// ===========================================================================
// Construction
// ===========================================================================

TEST( AdsTree, EmptyOnConstruction )
{
    Tree t;
    EXPECT_TRUE( t.empty() );
    EXPECT_EQ( t.numNodes(), 0 );
    EXPECT_EQ( t.numLeaves(), 0 );
    EXPECT_EQ( t.numDone(), 0 );
    EXPECT_EQ( t.numActive(), 0 );
}

TEST( AdsTree, AddLeafOneNode )
{
    Tree t;
    const int idx = t.addLeaf( const_tte( 1.0 ), unit_box() );
    EXPECT_EQ( idx, 0 );
    EXPECT_FALSE( t.empty() );
    EXPECT_EQ( t.numNodes(), 1 );
    EXPECT_EQ( t.numLeaves(), 1 );
    EXPECT_EQ( t.numActive(), 1 );
    EXPECT_TRUE( t.node( 0 ).isLeaf() );
    EXPECT_FALSE( t.node( 0 ).isInternal() );
}

// ===========================================================================
// Work-queue: pop / markDone
// ===========================================================================

TEST( AdsTree, PopReturnsFront )
{
    Tree t;
    t.addLeaf( const_tte( 1.0 ), unit_box() );
    EXPECT_EQ( t.front(), 0 );
    const int idx = t.pop();
    EXPECT_EQ( idx, 0 );
    EXPECT_TRUE( t.empty() );
    // The node still lives in the arena.
    EXPECT_EQ( t.numNodes(), 1 );
}

TEST( AdsTree, MarkDone )
{
    Tree t;
    t.addLeaf( const_tte( 2.0 ), unit_box() );
    const int idx = t.pop();
    t.markDone( idx );

    EXPECT_EQ( t.numDone(), 1 );
    EXPECT_EQ( t.doneLeaves().front(), 0 );
    EXPECT_TRUE( t.node( 0 ).leaf().done );
    // Leaf remains in leafList_.
    EXPECT_EQ( t.numLeaves(), 1 );
}

// ===========================================================================
// Splitting: structure
// ===========================================================================

TEST( AdsTree, SplitCreatesChildren )
{
    Tree t;
    t.addLeaf( const_tte( 1.0 ), unit_box() );
    const int idx         = t.pop();
    const auto [li, ri] = t.split( idx, 0, const_tte( 1.0 ), const_tte( 1.0 ) );

    EXPECT_EQ( li, 1 );
    EXPECT_EQ( ri, 2 );

    // Root converted to Internal.
    EXPECT_TRUE( t.node( 0 ).isInternal() );
    EXPECT_EQ( t.node( 0 ).internal().splitDim, 0 );

    // Children are Leaves.
    EXPECT_TRUE( t.node( 1 ).isLeaf() );
    EXPECT_TRUE( t.node( 2 ).isLeaf() );
    EXPECT_EQ( t.node( 1 ).parentIdx, 0 );
    EXPECT_EQ( t.node( 2 ).parentIdx, 0 );

    EXPECT_EQ( t.numNodes(), 3 );
    EXPECT_EQ( t.numLeaves(), 2 );   // leafList_ = {1, 2}
    EXPECT_EQ( t.numActive(), 2 );   // both children enqueued
}

TEST( AdsTree, SplitBoxGeometry )
{
    Tree t;
    // Box: center={0,0}, halfWidth={1,1}  →  x ∈ [-1,1], y ∈ [-1,1]
    t.addLeaf( const_tte( 0.0 ), unit_box() );
    const int idx         = t.pop();
    const auto [li, ri] = t.split( idx, 0, const_tte( 0.0 ), const_tte( 0.0 ) );

    const auto& lb = t.node( li ).leaf().box;
    const auto& rb = t.node( ri ).leaf().box;

    // Left:  center=(-0.5, 0), halfWidth=(0.5, 1)  →  x ∈ [-1, 0]
    EXPECT_DOUBLE_EQ( lb.center[0],    -0.5 );
    EXPECT_DOUBLE_EQ( lb.center[1],     0.0 );
    EXPECT_DOUBLE_EQ( lb.halfWidth[0],  0.5 );
    EXPECT_DOUBLE_EQ( lb.halfWidth[1],  1.0 );

    // Right: center=(+0.5, 0), halfWidth=(0.5, 1)  →  x ∈ [0, 1]
    EXPECT_DOUBLE_EQ( rb.center[0],    +0.5 );
    EXPECT_DOUBLE_EQ( rb.center[1],     0.0 );
    EXPECT_DOUBLE_EQ( rb.halfWidth[0],  0.5 );
    EXPECT_DOUBLE_EQ( rb.halfWidth[1],  1.0 );
}

TEST( AdsTree, SplitInternalSplitValue )
{
    Tree t;
    t.addLeaf( const_tte( 0.0 ), unit_box() );
    const int idx = t.pop();
    t.split( idx, 1, const_tte( 0.0 ), const_tte( 0.0 ) );

    // Split was along y (dim 1), boundary at center[1] = 0.
    EXPECT_EQ( t.node( idx ).internal().splitDim, 1 );
    EXPECT_DOUBLE_EQ( t.node( idx ).internal().splitValue, 0.0 );
}

// ===========================================================================
// leafList_ consistency
// ===========================================================================

TEST( AdsTree, LeafListAfterTwoSplits )
{
    Tree t;
    t.addLeaf( const_tte( 0.0 ), unit_box() );

    const int root      = t.pop();
    const auto [l, r] = t.split( root, 0, const_tte( 0.0 ), const_tte( 0.0 ) );

    // Split the left child along dim 1.
    t.pop();  // pop l
    const auto [ll, lr] = t.split( l, 1, const_tte( 0.0 ), const_tte( 0.0 ) );
    t.pop();  // pop r  (not splitting r)
    t.pop();  // pop ll
    t.pop();  // pop lr

    // Current leaves: {r, ll, lr}  (root and l are now internal)
    auto span = t.leafList();
    EXPECT_EQ( span.size(), 3u );

    std::vector< int > leaves( span.begin(), span.end() );
    std::sort( leaves.begin(), leaves.end() );

    std::vector< int > expected = { r, ll, lr };
    std::sort( expected.begin(), expected.end() );
    EXPECT_EQ( leaves, expected );
}

// ===========================================================================
// Point lookup: findLeaf
// ===========================================================================

TEST( AdsTree, FindLeafSingleSplit )
{
    Tree t;
    t.addLeaf( const_tte( 0.0 ), unit_box() );
    const int root      = t.pop();
    const auto [li, ri] = t.split( root, 0, const_tte( 10.0 ), const_tte( 20.0 ) );
    t.pop(); t.pop();  // drain work queue (findLeaf works regardless)

    // li covers x ∈ [-1, 0]
    EXPECT_EQ( t.findLeaf( { -0.5, 0.0 } ), li );
    EXPECT_DOUBLE_EQ( t.node( t.findLeaf( { -0.5, 0.0 } ) ).leaf().tte.value(), 10.0 );

    // ri covers x ∈ [0, 1]
    EXPECT_EQ( t.findLeaf( { +0.5, 0.0 } ), ri );
    EXPECT_DOUBLE_EQ( t.node( t.findLeaf( { +0.5, 0.0 } ) ).leaf().tte.value(), 20.0 );
}

TEST( AdsTree, FindLeafTwoLevelTree )
{
    Tree t;
    t.addLeaf( const_tte( 0.0 ), unit_box() );
    // Level 1: split root along x (dim 0).
    const int root      = t.pop();
    const auto [l, r] = t.split( root, 0, const_tte( 0.0 ), const_tte( 0.0 ) );
    // Level 2: split l along y (dim 1).
    t.pop();  // pop l
    const auto [ll, lr] = t.split( l, 1, const_tte( 11.0 ), const_tte( 12.0 ) );
    t.pop(); t.pop(); t.pop();  // drain

    // r  covers x ∈ [0,1],   y ∈ [-1,1]
    EXPECT_EQ( t.findLeaf( { 0.7, 0.3 } ), r );

    // ll covers x ∈ [-1,0],  y ∈ [-1,0]
    EXPECT_EQ( t.findLeaf( { -0.5, -0.5 } ), ll );
    EXPECT_DOUBLE_EQ( t.node( ll ).leaf().tte.value(), 11.0 );

    // lr covers x ∈ [-1,0],  y ∈ [0,1]
    EXPECT_EQ( t.findLeaf( { -0.5, +0.5 } ), lr );
    EXPECT_DOUBLE_EQ( t.node( lr ).leaf().tte.value(), 12.0 );
}

TEST( AdsTree, FindLeafReturnsMinusOneWhenMissing )
{
    Tree t;
    // Box only covers x ∈ [-1,0], y ∈ [-1,1]
    t.addLeaf( const_tte( 1.0 ), Box2{ { -0.5, 0.0 }, { 0.5, 1.0 } } );

    // Point outside the box.
    EXPECT_EQ( t.findLeaf( { 0.5, 0.0 } ), -1 );
}

// ===========================================================================
// Multi-root
// ===========================================================================

TEST( AdsTree, MultipleRoots )
{
    Tree t;
    // Two side-by-side root leaves covering [-1,0] and [0,1] in x, full y.
    const Box2 left_box{ { -0.5, 0.0 }, { 0.5, 1.0 } };
    const Box2 right_box{ { +0.5, 0.0 }, { 0.5, 1.0 } };

    const int r0 = t.addLeaf( const_tte( 1.0 ), left_box );
    const int r1 = t.addLeaf( const_tte( 2.0 ), right_box );
    t.pop(); t.pop();  // drain

    EXPECT_EQ( t.findLeaf( { -0.3, 0.5 } ), r0 );
    EXPECT_EQ( t.findLeaf( { +0.7, 0.5 } ), r1 );
}

// ===========================================================================
// Full propagation loop
// ===========================================================================

TEST( AdsTree, PropagationLoop )
{
    // Rule: split any leaf with tte.value() > 1.  Both children get half the
    // value.  Starting from 4.0:
    //   4.0 → {2.0, 2.0}  →  {1.0, 1.0, 1.0, 1.0}  →  done (all ≤ 1)
    Tree t;
    t.addLeaf( const_tte( 4.0 ), unit_box() );

    while ( !t.empty() )
    {
        const int idx = t.pop();
        const double v = t.node( idx ).leaf().tte.value();
        if ( v > 1.0 )
        {
            const double half = v * 0.5;
            t.split( idx, 0, const_tte( half ), const_tte( half ) );
        }
        else
        {
            t.markDone( idx );
        }
    }

    EXPECT_EQ( t.numDone(), 4 );
    for ( int i : t.doneLeaves() )
        EXPECT_LE( t.node( i ).leaf().tte.value(), 1.0 );

    // All nodes = 1 (root) + 2 + 4 = 7; internal = 3; leaves = 4.
    EXPECT_EQ( t.numNodes(), 7 );
    EXPECT_EQ( t.numLeaves(), 4 );
}

TEST( AdsTree, PropagationLoopAlternatingDim )
{
    // Same as above but split dim alternates: 0, 1, 0, 1, ...
    // Split 8.0 → 4.0 (×2) → 2.0 (×4) → 1.0 (×8): 3 levels deep.
    Tree t;
    t.addLeaf( const_tte( 8.0 ), unit_box() );

    int depth = 0;
    while ( !t.empty() )
    {
        const int    idx = t.pop();
        const double v   = t.node( idx ).leaf().tte.value();
        if ( v > 1.0 )
        {
            const int dim  = depth % 2;
            const double h = v * 0.5;
            t.split( idx, dim, const_tte( h ), const_tte( h ) );
            ++depth;
        }
        else
        {
            t.markDone( idx );
        }
    }

    EXPECT_EQ( t.numDone(), 8 );
    for ( int i : t.doneLeaves() )
        EXPECT_LE( t.node( i ).leaf().tte.value(), 1.0 );
}
