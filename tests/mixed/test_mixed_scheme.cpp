#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/scheme.hpp>
#include <tax/tax.hpp>
#include <type_traits>

using tax::Group;
using tax::MixedScheme;

// BoxTE<Groups...> is a first-class dense TaylorExpansion over a MixedScheme.
TEST( BoxTE, InstantiatesAsTaylorExpansion )
{
    using ME = tax::BoxTE< Group< 1, 4 >, Group< 1, 3 > >;
    static_assert( ME::nCoefficients == 20 );  // 5 * 4
    static_assert( ME::vars_v == 2 );
    static_assert( ME::order_v == 7 );
    static_assert(
        std::is_same_v<
            ME, tax::TaylorExpansion< double, MixedScheme< Group< 1, 4 >, Group< 1, 3 > > > > );

    // Full math + value surface works on the mixed type.
    typename ME::Input p{ 0.5, -0.3 };
    auto x = ME::variable< 0 >( p );
    auto y = ME::variable< 1 >( p );
    auto f = sin( x ) + x * y + exp( y );
    EXPECT_NEAR( f.value(), std::sin( 0.5 ) + 0.5 * ( -0.3 ) + std::exp( -0.3 ), 1e-12 );
    SUCCEED();
}

// Box count = product of per-group simplex sizes; differs from the joint simplex.
TEST( MixedScheme, KeptCountIsBoxProduct )
{
    // x@2 (1 var) box t@2 (1 var): 3 * 3 = 9 (joint simplex numMonomials(2,2)=6).
    static_assert( MixedScheme< Group< 1, 2 >, Group< 1, 2 > >::nCoeff == 9 );
    // x@4 (1 var) box p@20 (1 var): 5 * 21 = 105.
    static_assert( MixedScheme< Group< 1, 4 >, Group< 1, 20 > >::nCoeff == 105 );
    // x@4 over 3 vars (numMonomials(4,3)=35) box t@20: 35 * 21 = 735.
    static_assert( MixedScheme< Group< 3, 4 >, Group< 1, 20 > >::nCoeff == 735 );
    SUCCEED();
}

// flatOf/multiOf are inverse over the whole box, dense in [0,nCoeff), graded.
TEST( MixedScheme, FlatRoundTripDenseAndGraded )
{
    using S = MixedScheme< Group< 1, 2 >, Group< 2, 3 > >;  // x(1 var)@2, y(2 vars)@3
    std::set< std::size_t > seen;
    int prev_degree = -1;
    bool graded = true;
    // Walk all flats; each maps to an in-box multi-index, round-trips, and is graded.
    for ( std::size_t k = 0; k < S::nCoeff; ++k )
    {
        auto a = S::multiOf( k );
        EXPECT_EQ( S::flatOf( a ), k );  // round trip
        // in-box: per-group degree caps
        EXPECT_LE( a[0], 2 );         // x degree
        EXPECT_LE( a[1] + a[2], 3 );  // y total degree
        int d = tax::totalDegree( a );
        if ( d < prev_degree ) graded = false;  // non-decreasing total degree
        prev_degree = d;
        seen.insert( k );
    }
    EXPECT_TRUE( graded );
    EXPECT_EQ( seen.size(), S::nCoeff );  // dense, no gaps
}

// An out-of-box multi-index is rejected.
TEST( MixedScheme, OutOfBoxRejected )
{
    using S = MixedScheme< Group< 1, 2 >, Group< 1, 2 > >;
    tax::MultiIndex< 2 > over{ 3, 0 };  // x^3, x order is 2
    EXPECT_EQ( S::flatOf( over ), S::kNotInBox );
}

TEST( MixedScheme, CauchyProductMatchesBruteForce )
{
    using S = MixedScheme< Group< 1, 2 >, Group< 2, 3 > >;
    std::array< double, S::nCoeff > a{}, b{}, out{};
    for ( std::size_t k = 0; k < S::nCoeff; ++k )
    {
        a[k] = 0.1 + 0.03 * double( k );
        b[k] = -0.2 + 0.05 * double( k );
    }
    tax::cauchyProduct< double, S >( out, a, b );

    // Brute force: for every pair of in-box monomials whose product is in-box, accumulate.
    std::array< double, S::nCoeff > ref{};
    for ( std::size_t i = 0; i < S::nCoeff; ++i )
        for ( std::size_t j = 0; j < S::nCoeff; ++j )
        {
            auto ai = S::multiOf( i );
            auto aj = S::multiOf( j );
            tax::MultiIndex< S::vars > sum{};
            for ( int v = 0; v < S::vars; ++v )
                sum[std::size_t( v )] = ai[std::size_t( v )] + aj[std::size_t( v )];
            std::size_t o = S::flatOf( sum );
            if ( o != S::kNotInBox ) ref[o] += a[i] * b[j];
        }
    for ( std::size_t k = 0; k < S::nCoeff; ++k ) EXPECT_DOUBLE_EQ( out[k], ref[k] );
}
