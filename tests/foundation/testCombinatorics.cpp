#include "testUtils.hpp"

// Access detail internals directly for unit-testing the combinatorics layer.
using namespace tax::detail;

template < int M >
static void ExpectFlatUnflatRoundTrip( int N )
{
    for ( int d = 0; d <= N; ++d )
    {
        forEachMonomial< M >( d, [&]( const tax::MultiIndex< M >& alpha, std::size_t k ) {
            EXPECT_EQ( unflatIndex< M >( k ), alpha ) << "M=" << M << " d=" << d << " k=" << k;
            EXPECT_EQ( flatIndex< M >( unflatIndex< M >( k ) ), k )
                << "M=" << M << " d=" << d << " k=" << k;
        } );
    }
}

// =============================================================================
// binom
// =============================================================================

TEST( Binom, BaseCase_Zero )
{
    EXPECT_EQ( binom( 0, 0 ), 1u );
    EXPECT_EQ( binom( 5, 0 ), 1u );
    EXPECT_EQ( binom( 5, 5 ), 1u );
}

TEST( Binom, NegativeArgs_ReturnsZero )
{
    EXPECT_EQ( binom( -1, 0 ), 0u );
    EXPECT_EQ( binom( 5, -1 ), 0u );
    EXPECT_EQ( binom( 3, 5 ), 0u );  // k > n
}

TEST( Binom, KnownValues )
{
    EXPECT_EQ( binom( 5, 1 ), 5u );
    EXPECT_EQ( binom( 5, 2 ), 10u );
    EXPECT_EQ( binom( 5, 3 ), 10u );
    EXPECT_EQ( binom( 6, 3 ), 20u );
    EXPECT_EQ( binom( 10, 4 ), 210u );
}

TEST( Binom, Symmetry )
{
    for ( int n = 0; n <= 8; ++n )
        for ( int k = 0; k <= n; ++k )
            EXPECT_EQ( binom( n, k ), binom( n, n - k ) ) << "n=" << n << " k=" << k;
}

// =============================================================================
// numMonomials
// =============================================================================

TEST( NumMonomials, Univariate )
{
    // M=1: monomials 1,x,...,x^N  → N+1 total
    for ( int N = 0; N <= 8; ++N )
        EXPECT_EQ( numMonomials( N, 1 ), std::size_t( N + 1 ) ) << "N=" << N;
}

TEST( NumMonomials, Bivariate )
{
    // M=2: C(N+2,2) = (N+1)(N+2)/2
    for ( int N = 0; N <= 6; ++N )
    {
        std::size_t expected = std::size_t( N + 1 ) * std::size_t( N + 2 ) / 2;
        EXPECT_EQ( numMonomials( N, 2 ), expected ) << "N=" << N;
    }
}

TEST( NumMonomials, KnownValues )
{
    EXPECT_EQ( numMonomials( 0, 3 ), 1u );   // only constant
    EXPECT_EQ( numMonomials( 1, 3 ), 4u );   // 1,x,y,z
    EXPECT_EQ( numMonomials( 2, 3 ), 10u );  // + xy,xz,yz,x^2,y^2,z^2
    EXPECT_EQ( numMonomials( 3, 3 ), 20u );
}

TEST( NumMonomials, Order0_IsAlways1 )
{
    for ( int M = 1; M <= 5; ++M ) EXPECT_EQ( numMonomials( 0, M ), 1u ) << "M=" << M;
}

// =============================================================================
// totalDegree
// =============================================================================

TEST( TotalDegree, Univariate )
{
    EXPECT_EQ( totalDegree< 1 >( { 0 } ), 0 );
    EXPECT_EQ( totalDegree< 1 >( { 3 } ), 3 );
}

TEST( TotalDegree, Bivariate )
{
    EXPECT_EQ( totalDegree< 2 >( { 1, 2 } ), 3 );
    EXPECT_EQ( totalDegree< 2 >( { 0, 0 } ), 0 );
}

TEST( TotalDegree, Trivariate )
{
    EXPECT_EQ( totalDegree< 3 >( { 1, 1, 1 } ), 3 );
    EXPECT_EQ( totalDegree< 3 >( { 2, 0, 1 } ), 3 );
}

// =============================================================================
// flatIndex — grlex ordering
// =============================================================================

TEST( FlatIndex, Univariate_MatchesDegree )
{
    // M=1: flatIndex({k}) == k for all k
    for ( int k = 0; k <= 6; ++k )
        EXPECT_EQ( flatIndex< 1 >( { k } ), std::size_t( k ) ) << "k=" << k;
}

TEST( FlatIndex, Bivariate_Order2 )
{
    // grlex ordering for M=2, up to degree 2:
    // {0,0}→0, {1,0}→1, {0,1}→2, {2,0}→3, {1,1}→4, {0,2}→5
    EXPECT_EQ( flatIndex< 2 >( { 0, 0 } ), 0u );
    EXPECT_EQ( flatIndex< 2 >( { 1, 0 } ), 1u );
    EXPECT_EQ( flatIndex< 2 >( { 0, 1 } ), 2u );
    EXPECT_EQ( flatIndex< 2 >( { 2, 0 } ), 3u );
    EXPECT_EQ( flatIndex< 2 >( { 1, 1 } ), 4u );
    EXPECT_EQ( flatIndex< 2 >( { 0, 2 } ), 5u );
}

TEST( FlatIndex, Trivariate_Degree1 )
{
    // degree-0: {0,0,0}→0
    // degree-1: {1,0,0}→1, {0,1,0}→2, {0,0,1}→3
    EXPECT_EQ( flatIndex< 3 >( { 0, 0, 0 } ), 0u );
    EXPECT_EQ( flatIndex< 3 >( { 1, 0, 0 } ), 1u );
    EXPECT_EQ( flatIndex< 3 >( { 0, 1, 0 } ), 2u );
    EXPECT_EQ( flatIndex< 3 >( { 0, 0, 1 } ), 3u );
}

TEST( FlatIndex, Indices_AreUnique )
{
    // All multi-indices of degree <= 3 in M=2 variables must map to distinct
    // indices in [0, numMonomials(3,2)).
    constexpr int N = 3, M = 2;
    std::vector< std::size_t > seen;
    for ( int a0 = 0; a0 <= N; ++a0 )
        for ( int a1 = 0; a0 + a1 <= N; ++a1 ) seen.push_back( flatIndex< M >( { a0, a1 } ) );
    std::sort( seen.begin(), seen.end() );
    EXPECT_EQ( seen.size(), numMonomials( N, M ) );
    for ( std::size_t i = 0; i < seen.size(); ++i ) EXPECT_EQ( seen[i], i );
}

TEST( FlatIndex, ConstantTerm_IsAlwaysZero )
{
    EXPECT_EQ( flatIndex< 1 >( { 0 } ), 0u );
    EXPECT_EQ( flatIndex< 2 >( { 0, 0 } ), 0u );
    EXPECT_EQ( flatIndex< 3 >( { 0, 0, 0 } ), 0u );
}

// =============================================================================
// unflatIndex — inverse grlex mapping
// =============================================================================

TEST( UnflatIndex, Univariate_MatchesDegree )
{
    for ( int k = 0; k <= 6; ++k )
        EXPECT_EQ( unflatIndex< 1 >( std::size_t( k ) ), ( tax::MultiIndex< 1 >{ k } ) )
            << "k=" << k;
}

TEST( UnflatIndex, Bivariate_Order2 )
{
    EXPECT_EQ( unflatIndex< 2 >( 0u ), ( tax::MultiIndex< 2 >{ 0, 0 } ) );
    EXPECT_EQ( unflatIndex< 2 >( 1u ), ( tax::MultiIndex< 2 >{ 1, 0 } ) );
    EXPECT_EQ( unflatIndex< 2 >( 2u ), ( tax::MultiIndex< 2 >{ 0, 1 } ) );
    EXPECT_EQ( unflatIndex< 2 >( 3u ), ( tax::MultiIndex< 2 >{ 2, 0 } ) );
    EXPECT_EQ( unflatIndex< 2 >( 4u ), ( tax::MultiIndex< 2 >{ 1, 1 } ) );
    EXPECT_EQ( unflatIndex< 2 >( 5u ), ( tax::MultiIndex< 2 >{ 0, 2 } ) );
}

TEST( UnflatIndex, RoundTrip_UpToOrder4 )
{
    ExpectFlatUnflatRoundTrip< 1 >( 4 );
    ExpectFlatUnflatRoundTrip< 2 >( 4 );
    ExpectFlatUnflatRoundTrip< 3 >( 4 );
    ExpectFlatUnflatRoundTrip< 4 >( 4 );
}
