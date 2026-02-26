
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <set>
#include <vector>

#include "tax/index.hpp"

using namespace tax::detail;

// ------------------------------
// Compile-time sanity checks
// ------------------------------
// These are great for consteval/constexpr logic: they fail at compile-time.
static_assert( totalDegree< 1 >( 0 ) == 1 );
static_assert( totalDegree< 1 >( 10 ) == 1 );
static_assert( totalDegree< 2 >( 0 ) == 1 );  // (0,0)
static_assert( totalDegree< 2 >( 1 ) == 2 );  // (0,1),(1,0)
static_assert( totalDegree< 3 >( 2 ) == 6 );  // compositions of 2 into 3 parts = C(4,2)=6

static_assert( countMonomials< 0, 1 >() == 1 );
static_assert( countMonomials< 2, 1 >() == 3 );   // degree<=2 in 1D => {0,1,2}
static_assert( countMonomials< 2, 2 >() == 6 );   // C(2+2,2)=6
static_assert( countMonomials< 3, 2 >() == 10 );  // C(3+2,2)=10

// ------------------------------
// Runtime tests
// ------------------------------

TEST( IndexSetCombinatorics, TotalDegreeMatchesKnownValues )
{
    EXPECT_EQ( totalDegree< 1 >( 0 ), 1u );
    EXPECT_EQ( totalDegree< 1 >( 7 ), 1u );

    EXPECT_EQ( totalDegree< 2 >( 0 ), 1u );
    EXPECT_EQ( totalDegree< 2 >( 1 ), 2u );
    EXPECT_EQ( totalDegree< 2 >( 2 ), 3u );  // (0,2),(1,1),(2,0)

    EXPECT_EQ( totalDegree< 3 >( 0 ), 1u );
    EXPECT_EQ( totalDegree< 3 >( 1 ), 3u );
    EXPECT_EQ( totalDegree< 3 >( 2 ), 6u );
    EXPECT_EQ( totalDegree< 3 >( 3 ), 10u );
}

TEST( IndexSetCombinatorics, CountMonomialsMatchesKnownValues )
{
    EXPECT_EQ( ( countMonomials< 0, 1 >() ), 1u );
    EXPECT_EQ( ( countMonomials< 1, 1 >() ), 2u );
    EXPECT_EQ( ( countMonomials< 2, 1 >() ), 3u );

    EXPECT_EQ( ( countMonomials< 0, 2 >() ), 1u );
    EXPECT_EQ( ( countMonomials< 1, 2 >() ), 3u );  // (0,0),(0,1),(1,0)
    EXPECT_EQ( ( countMonomials< 2, 2 >() ), 6u );
    EXPECT_EQ( ( countMonomials< 3, 2 >() ), 10u );

    EXPECT_EQ( ( countMonomials< 2, 3 >() ), 10u );  // C(2+3,3)=10
}

TEST( IndexSetCombinatorics, FindIndexWithinDegree )
{
    // For D=2, fixed degree=2 compositions order produced by makeCompositions:
    // (0,2), (1,1), (2,0)
    {
        std::array< std::size_t, 2 > a{ 0, 2 };
        std::array< std::size_t, 2 > b{ 1, 1 };
        std::array< std::size_t, 2 > c{ 2, 0 };

        EXPECT_EQ( findIndexWithinDegree< 2 >( a ), 0u );
        EXPECT_EQ( findIndexWithinDegree< 2 >( b ), 1u );
        EXPECT_EQ( findIndexWithinDegree< 2 >( c ), 2u );
    }

    // degree=3: (0,3),(1,2),(2,1),(3,0)
    {
        std::array< std::size_t, 2 > a{ 0, 3 };
        std::array< std::size_t, 2 > b{ 1, 2 };
        std::array< std::size_t, 2 > c{ 2, 1 };
        std::array< std::size_t, 2 > d{ 3, 0 };

        EXPECT_EQ( findIndexWithinDegree< 2 >( a ), 0u );
        EXPECT_EQ( findIndexWithinDegree< 2 >( b ), 1u );
        EXPECT_EQ( findIndexWithinDegree< 2 >( c ), 2u );
        EXPECT_EQ( findIndexWithinDegree< 2 >( d ), 3u );
    }
}

TEST( IndexSetCombinatorics, FindIndex )
{
    // For D=2, alphas for N=2 are:
    // deg0: (0,0)
    // deg1: (0,1),(1,0)
    // deg2: (0,2),(1,1),(2,0)
    // Therefore rank should map to indices 0..5 accordingly.

    EXPECT_EQ( ( findIndex< 2, 2 >( { 0, 0 } ) ), 0u );

    EXPECT_EQ( ( findIndex< 2, 2 >( { 0, 1 } ) ), 1u );
    EXPECT_EQ( ( findIndex< 2, 2 >( { 1, 0 } ) ), 2u );

    EXPECT_EQ( ( findIndex< 2, 2 >( { 0, 2 } ) ), 3u );
    EXPECT_EQ( ( findIndex< 2, 2 >( { 1, 1 } ) ), 4u );
    EXPECT_EQ( ( findIndex< 2, 2 >( { 2, 0 } ) ), 5u );
}

TEST( IndexSetPrecompute, MakeIndices )
{
    // Compare against the exact expected graded-lex order produced by makeIndexes<2,2>()
    constexpr auto alphas = makeIndices< 2, 2 >();

    using A = std::array< std::size_t, 2 >;
    constexpr std::array< A, 6 > expected = { A{ 0, 0 }, A{ 0, 1 }, A{ 1, 0 },
                                              A{ 0, 2 }, A{ 1, 1 }, A{ 2, 0 } };

    for ( std::size_t i = 0; i < expected.size(); ++i )
    {
        EXPECT_EQ( alphas[i], expected[i] ) << "Mismatch at i=" << i;
    }
}

TEST( IndexSetPrecompute, IndexOffsets )
{
    // For N=3, D=2:
    // totalDegree<2>(0)=1
    // totalDegree<2>(1)=2
    // totalDegree<2>(2)=3
    // totalDegree<2>(3)=4
    // starts: [0, 1, 3, 6, 10]
    constexpr auto start = makeIndexOffset< 3, 2 >();

    EXPECT_EQ( start[0], 0u );
    EXPECT_EQ( start[1], 1u );
    EXPECT_EQ( start[2], 3u );
    EXPECT_EQ( start[3], 6u );
    EXPECT_EQ( start[4], 10u );  // start[N+1]
}

TEST( IndexSet, BasicPropertiesAndConsistency )
{
    using IS = IndexSet< 3, 2 >;

    // size matches formula
    EXPECT_EQ( IS::size, ( countMonomials< 3, 2 >() ) );
    EXPECT_EQ( IS::size, 10u );

    // start should end at size
    EXPECT_EQ( IS::offset[IS::order + 1], IS::size );

    // degrees should be non-decreasing in the alphas listing
    for ( std::size_t i = 1; i < IS::size; ++i )
    {
        EXPECT_LE( IS::degreeOf( IS::data[i - 1] ), IS::degreeOf( IS::data[i] ) )
            << "Degree decreased at i=" << i;
    }

    // alphas should be unique
    // (convert arrays to strings or to a comparable container)
    std::set< std::array< std::size_t, 2 > > seen;
    for ( std::size_t i = 0; i < IS::size; ++i )
    {
        EXPECT_TRUE( seen.insert( IS::data[i] ).second ) << "Duplicate alpha at i=" << i;
    }

    // Key invariant: indexOf(alphas[i]) == i
    for ( std::size_t i = 0; i < IS::size; ++i )
    {
        EXPECT_EQ( IS::indexOf( IS::data[i] ), i )
            << "indexOf(alphas[i]) != i at i=" << i;
    }
}

TEST( IndexSet, Accessors )
{
    // Pick a small 3D set where we can sanity-check degrees and rank consistency
    using IS = IndexSet< 2, 3 >;  // size = C(2+3,3)=10

    EXPECT_EQ( IS::size, 10u );
    EXPECT_EQ( IS::offset[IS::order + 1], IS::size );

    // Invariant: indexOf(alphas[i]) == i
    for ( std::size_t i = 0; i < IS::size; ++i )
    {
        EXPECT_EQ( IS::indexOf( IS::data[i] ), i ) << "i=" << i;
        EXPECT_LE( IS::degreeOf( IS::data[i] ), IS::order );
    }
}
