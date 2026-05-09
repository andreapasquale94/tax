// SPDX-License-Identifier: BSD-3-Clause
//
// Multi-index utility tests.

#include <gtest/gtest.h>

#include <array>
#include <span>

#include "tax/util/multi_index.hpp"

using namespace tax::util;

TEST( MultiIndex, MonomialCount )
{
    EXPECT_EQ( monomialCount( 0, 1 ), 1u );
    EXPECT_EQ( monomialCount( 5, 1 ), 6u );
    EXPECT_EQ( monomialCount( 2, 2 ), 6u );
    EXPECT_EQ( monomialCount( 2, 3 ), 10u );
    EXPECT_EQ( monomialCount( 3, 3 ), 20u );
}

TEST( MultiIndex, DegreeSize )
{
    EXPECT_EQ( degreeSize( 0, 2 ), 1u );
    EXPECT_EQ( degreeSize( 1, 2 ), 2u );
    EXPECT_EQ( degreeSize( 2, 2 ), 3u );
    EXPECT_EQ( degreeSize( 2, 3 ), 6u );
}

TEST( MultiIndex, DegreeOffset )
{
    EXPECT_EQ( degreeOffset( 0, 2 ), 0u );
    EXPECT_EQ( degreeOffset( 1, 2 ), 1u );
    EXPECT_EQ( degreeOffset( 2, 2 ), 3u );
    EXPECT_EQ( degreeOffset( 3, 3 ), 10u );  // 1 + 3 + 6 = 10
}

TEST( MultiIndex, FlatIndexUnivariate )
{
    std::array< std::size_t, 1 > a0{ 0 };
    std::array< std::size_t, 1 > a1{ 1 };
    std::array< std::size_t, 1 > a2{ 2 };
    EXPECT_EQ( flatIndex( std::span< const std::size_t >{ a0 } ), 0u );
    EXPECT_EQ( flatIndex( std::span< const std::size_t >{ a1 } ), 1u );
    EXPECT_EQ( flatIndex( std::span< const std::size_t >{ a2 } ), 2u );
}

TEST( MultiIndex, FlatIndexBivariate )
{
    auto fi = []( std::initializer_list< std::size_t > il ) {
        std::array< std::size_t, 2 > a{};
        std::size_t i = 0;
        for ( auto v : il )
            a[ i++ ] = v;
        return flatIndex( std::span< const std::size_t >{ a } );
    };
    // Order (M=2): (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
    EXPECT_EQ( fi( { 0, 0 } ), 0u );
    EXPECT_EQ( fi( { 1, 0 } ), 1u );
    EXPECT_EQ( fi( { 0, 1 } ), 2u );
    EXPECT_EQ( fi( { 2, 0 } ), 3u );
    EXPECT_EQ( fi( { 1, 1 } ), 4u );
    EXPECT_EQ( fi( { 0, 2 } ), 5u );
}

TEST( MultiIndex, FlatIndexTrivariate )
{
    auto fi = []( std::initializer_list< std::size_t > il ) {
        std::array< std::size_t, 3 > a{};
        std::size_t i = 0;
        for ( auto v : il )
            a[ i++ ] = v;
        return flatIndex( std::span< const std::size_t >{ a } );
    };
    // M=3 order 2: (0,0,0), (1,0,0), (0,1,0), (0,0,1),
    //              (2,0,0), (1,1,0), (0,2,0), (1,0,1), (0,1,1), (0,0,2)
    EXPECT_EQ( fi( { 0, 0, 0 } ), 0u );
    EXPECT_EQ( fi( { 1, 0, 0 } ), 1u );
    EXPECT_EQ( fi( { 0, 1, 0 } ), 2u );
    EXPECT_EQ( fi( { 0, 0, 1 } ), 3u );
    EXPECT_EQ( fi( { 2, 0, 0 } ), 4u );
    EXPECT_EQ( fi( { 1, 1, 0 } ), 5u );
    EXPECT_EQ( fi( { 0, 2, 0 } ), 6u );
    EXPECT_EQ( fi( { 1, 0, 1 } ), 7u );
    EXPECT_EQ( fi( { 0, 1, 1 } ), 8u );
    EXPECT_EQ( fi( { 0, 0, 2 } ), 9u );
}

TEST( MultiIndex, UnflatIsInverseOfFlat )
{
    constexpr std::size_t M = 3;
    constexpr std::size_t N = 4;
    const std::size_t total = monomialCount( N, M );
    for ( std::size_t i = 0; i < total; ++i )
    {
        std::array< std::size_t, M > a{};
        unflatIndex( i, std::span< std::size_t >( a ) );
        EXPECT_EQ( flatIndex( std::span< const std::size_t >( a ) ), i ) << "i = " << i;
    }
}

TEST( MultiIndex, ForEachMultiIndexEnumeratesAllAndInOrder )
{
    constexpr std::size_t M = 3;
    constexpr std::size_t d = 3;
    std::size_t expected = degreeOffset( d, M );
    forEachMultiIndexOfDegree( d, M, [ & ]( std::span< const std::size_t > a ) {
        EXPECT_EQ( flatIndex( a ), expected );
        ++expected;
    } );
    EXPECT_EQ( expected, degreeOffset( d, M ) + degreeSize( d, M ) );
}

TEST( MultiIndex, FlatIndexWithinDegree )
{
    auto fi = []( std::initializer_list< std::size_t > il ) {
        std::array< std::size_t, 3 > a{};
        std::size_t i = 0;
        for ( auto v : il )
            a[ i++ ] = v;
        return flatIndexWithinDegree( std::span< const std::size_t >{ a } );
    };
    EXPECT_EQ( fi( { 0, 0, 0 } ), 0u );
    EXPECT_EQ( fi( { 0, 0, 1 } ), 2u );  // degree 1, position 2 within block
    EXPECT_EQ( fi( { 2, 0, 0 } ), 0u );
    EXPECT_EQ( fi( { 0, 0, 2 } ), 5u );  // last in degree-2 block
}

TEST( MultiIndex, Factorial )
{
    auto fact = []( std::initializer_list< std::size_t > il ) {
        std::array< std::size_t, 4 > a{};
        std::size_t i = 0;
        for ( auto v : il )
            a[ i++ ] = v;
        return factorial( std::span< const std::size_t >( a.data(), il.size() ) );
    };
    EXPECT_EQ( fact( { 0 } ), 1u );
    EXPECT_EQ( fact( { 3 } ), 6u );
    EXPECT_EQ( fact( { 2, 3 } ), 12u );
    EXPECT_EQ( fact( { 1, 1, 1 } ), 1u );
    EXPECT_EQ( fact( { 4, 2 } ), 48u );
}
