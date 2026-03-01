#include "testUtils.hpp"

#include <tax/utils/enumeration.hpp>

#include <set>
#include <vector>

using namespace tax::detail;

// =============================================================================
// forEachMonomial
// =============================================================================

TEST( Enumeration, MonomialM1Degree0 )
{
    std::vector< std::pair< std::array< int, 1 >, std::size_t > > visited;
    forEachMonomial< 1 >( 0, [&]( const auto& alpha, std::size_t ai ) {
        visited.push_back( { alpha, ai } );
    } );
    ASSERT_EQ( visited.size(), 1u );
    EXPECT_EQ( visited[0].first[0], 0 );
    EXPECT_EQ( visited[0].second, flatIndex< 1 >( { 0 } ) );
}

TEST( Enumeration, MonomialM1Degree3 )
{
    std::vector< int > degrees;
    forEachMonomial< 1 >( 3, [&]( const auto& alpha, std::size_t ) {
        degrees.push_back( alpha[0] );
    } );
    ASSERT_EQ( degrees.size(), 1u );
    EXPECT_EQ( degrees[0], 3 );
}

TEST( Enumeration, MonomialM2Degree2 )
{
    // For M=2, degree=2: monomials are (2,0), (1,1), (0,2) — 3 monomials
    std::vector< std::array< int, 2 > > visited;
    forEachMonomial< 2 >( 2, [&]( const auto& alpha, std::size_t ai ) {
        visited.push_back( alpha );
        EXPECT_EQ( ai, flatIndex< 2 >( alpha ) );
    } );
    ASSERT_EQ( visited.size(), 3u );
    // Check total degree
    for ( const auto& a : visited )
        EXPECT_EQ( a[0] + a[1], 2 );
}

TEST( Enumeration, MonomialM3Degree1 )
{
    // For M=3, degree=1: (1,0,0), (0,1,0), (0,0,1) — 3 monomials
    std::vector< std::array< int, 3 > > visited;
    forEachMonomial< 3 >( 1, [&]( const auto& alpha, std::size_t ai ) {
        visited.push_back( alpha );
        EXPECT_EQ( ai, flatIndex< 3 >( alpha ) );
    } );
    ASSERT_EQ( visited.size(), 3u );
}

TEST( Enumeration, MonomialCountM2 )
{
    // Total number of monomials of degree d in M=2 vars is d+1
    for ( int d = 0; d <= 5; ++d )
    {
        int count = 0;
        forEachMonomial< 2 >( d, [&]( const auto&, std::size_t ) { ++count; } );
        EXPECT_EQ( count, d + 1 );
    }
}

// =============================================================================
// forEachSubIndex (no degree constraint)
// =============================================================================

TEST( Enumeration, SubIndexM1 )
{
    // alpha = {3}: sub-indices are beta={0}..{3}, gamma = alpha - beta
    std::array< int, 1 > alpha = { 3 };
    std::vector< std::pair< std::size_t, std::size_t > > visited;
    forEachSubIndex< 1 >( alpha, [&]( auto bi, auto gi ) {
        visited.push_back( { bi, gi } );
    } );
    ASSERT_EQ( visited.size(), 4u );  // beta = 0, 1, 2, 3
    for ( const auto& [bi, gi] : visited )
    {
        // bi + gi indices should reconstruct alpha
        EXPECT_EQ( bi + gi, flatIndex< 1 >( alpha ) );
    }
}

TEST( Enumeration, SubIndexM2 )
{
    // alpha = {1, 1}: sub-indices are all (beta, gamma) with beta + gamma = (1,1)
    std::array< int, 2 > alpha = { 1, 1 };
    int count = 0;
    forEachSubIndex< 2 >( alpha, [&]( auto bi, auto gi ) {
        (void)bi;
        (void)gi;
        ++count;
    } );
    // beta can be: (0,0), (0,1), (1,0), (1,1) — 4 sub-indices
    EXPECT_EQ( count, 4 );
}

// =============================================================================
// forEachSubIndex (degree-bounded)
// =============================================================================

TEST( Enumeration, SubIndexBoundedM1 )
{
    // alpha = {4}, db_lo=1, db_hi=3
    std::array< int, 1 > alpha = { 4 };
    std::vector< int > db_values;
    forEachSubIndex< 1 >( alpha, 1, 3, [&]( auto, auto, int db ) {
        db_values.push_back( db );
    } );
    ASSERT_EQ( db_values.size(), 3u );
    EXPECT_EQ( db_values[0], 1 );
    EXPECT_EQ( db_values[1], 2 );
    EXPECT_EQ( db_values[2], 3 );
}

TEST( Enumeration, SubIndexBoundedM2 )
{
    // alpha = {2, 1}, only iterate beta with |beta| = 1
    std::array< int, 2 > alpha = { 2, 1 };
    int count = 0;
    forEachSubIndex< 2 >( alpha, 1, 1, [&]( auto bi, auto gi, int db ) {
        EXPECT_EQ( db, 1 );
        (void)bi;
        (void)gi;
        ++count;
    } );
    // |beta|=1 with beta <= (2,1): beta can be (1,0) or (0,1) — 2 sub-indices
    EXPECT_EQ( count, 2 );
}

TEST( Enumeration, SubIndexBoundedEmpty )
{
    // db_lo > db_hi should yield no iterations
    std::array< int, 2 > alpha = { 1, 1 };
    int count = 0;
    forEachSubIndex< 2 >( alpha, 3, 2, [&]( auto, auto, int ) { ++count; } );
    EXPECT_EQ( count, 0 );
}

TEST( Enumeration, SubIndexBoundedConsistency )
{
    // Verify: iterating all degrees [0, |alpha|] with bounded overload
    // should visit the same set as the unbounded overload
    std::array< int, 2 > alpha = { 2, 1 };
    int total_degree = alpha[0] + alpha[1];

    std::set< std::pair< std::size_t, std::size_t > > from_unbounded;
    forEachSubIndex< 2 >( alpha, [&]( auto bi, auto gi ) {
        from_unbounded.insert( { bi, gi } );
    } );

    std::set< std::pair< std::size_t, std::size_t > > from_bounded;
    forEachSubIndex< 2 >( alpha, 0, total_degree, [&]( auto bi, auto gi, int ) {
        from_bounded.insert( { bi, gi } );
    } );

    EXPECT_EQ( from_unbounded, from_bounded );
}
