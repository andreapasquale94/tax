#include <gtest/gtest.h>

#include <tax/experimental/fused.hpp>

#include "../testUtils.hpp"

using tax::fuse;

namespace
{
// Largest absolute coefficient difference between two dense expansions.
template < typename E >
double maxCoeffDiff( const E& x, const E& y )
{
    double m = 0.0;
    for ( std::size_t k = 0; k < E::nCoefficients; ++k )
        m = std::max( m, std::abs( double( x[k] ) - double( y[k] ) ) );
    return m;
}
}  // namespace

// A fused elementwise expression must equal the eager evaluation coefficient
// for coefficient, for both univariate and multivariate schemes.
TEST( Fused, MatchesEagerUnivariate )
{
    auto a = tax::TE< 8 >::variable( 0.3 );
    auto b = tax::TE< 8 >::variable( 0.7 );
    auto c = exp( a );
    auto d = sin( b );

    tax::TE< 8 > eager = 2.0 * a + 3.0 * b - c + 0.5 * d + 1.5;
    tax::TE< 8 > fusedr = 2.0 * fuse( a ) + 3.0 * fuse( b ) - fuse( c ) + 0.5 * fuse( d ) + 1.5;

    EXPECT_LT( maxCoeffDiff( eager, fusedr ), 1e-14 );
}

TEST( Fused, MatchesEagerMultivariate )
{
    using E = tax::TEn< 6, 4 >;
    typename E::Input p{ 0.2, 0.3, 0.4, 0.5 };
    E a = E::variable< 0 >( p );
    E b = E::variable< 1 >( p );
    E c = exp( a );
    E d = E::variable< 2 >( p );

    E eager = 2.0 * a + 3.0 * b - c + 0.5 * d + 1.5;
    E fusedr = 2.0 * fuse( a ) + 3.0 * fuse( b ) - fuse( c ) + 0.5 * fuse( d ) + 1.5;

    EXPECT_LT( maxCoeffDiff( eager, fusedr ), 1e-14 );
}

// Mixing a fused node with a bare expansion (auto-wrapped) and a division.
TEST( Fused, MixedOperandsAndDivision )
{
    using E = tax::TEn< 5, 3 >;
    typename E::Input p{ 0.1, 0.2, 0.3 };
    E a = E::variable< 0 >( p );
    E b = E::variable< 1 >( p );
    E c = E::variable< 2 >( p );

    E eager = a + b - c;           // all eager
    E fusedr = fuse( a ) + b - c;  // head fused, b/c auto-wrapped
    EXPECT_LT( maxCoeffDiff( eager, fusedr ), 1e-14 );

    E eager2 = a / 4.0 - b;
    E fused2 = fuse( a ) / 4.0 - fuse( b );
    EXPECT_LT( maxCoeffDiff( eager2, fused2 ), 1e-14 );
}

// Batch (SIMD-style) coefficients must fuse identically too.
TEST( Fused, MatchesEagerBatch )
{
    using E = tax::TE< 6, 3, 4 >;  // Batch<double,4>
    typename E::Input p{};
    for ( int k = 0; k < 3; ++k ) p[std::size_t( k )] = tax::Batch< double, 4 >( 0.2 + 0.1 * k );
    E a = E::variable< 0 >( p );
    E b = E::variable< 1 >( p );
    E c = exp( a );

    E eager = 2.0 * a + 3.0 * b - c;
    E fusedr = 2.0 * fuse( a ) + 3.0 * fuse( b ) - fuse( c );

    double m = 0.0;
    for ( std::size_t k = 0; k < E::nCoefficients; ++k )
        for ( int l = 0; l < 4; ++l ) m = std::max( m, std::abs( eager[k][l] - fusedr[k][l] ) );
    EXPECT_LT( m, 1e-14 );
}

// The eager operators must still win overload resolution for bare expansions —
// `a + b` with no fuse() stays eager (this just has to compile and be correct).
TEST( Fused, EagerOperatorsUnaffected )
{
    auto a = tax::TE< 3 >::variable( 1.0 );
    auto b = tax::TE< 3 >::variable( 2.0 );
    auto c = a + b;
    static_assert( std::is_same_v< decltype( c ), tax::TE< 3 > > );
    EXPECT_DOUBLE_EQ( c.value(), 3.0 );
}

// Fused expressions are usable in constant evaluation.
TEST( Fused, Constexpr )
{
    constexpr double v = [] {
        auto x = tax::TE< 4 >::variable( 0.3 );
        tax::TE< 4 > r = 2.0 * fuse( x ) - fuse( x );  // == x
        return r.value();
    }();
    EXPECT_DOUBLE_EQ( v, 0.3 );
}
