#include <gtest/gtest.h>

#include <tax/tax.hpp>

#include "../testUtils.hpp"

// Dense: compile-time, order-reducing truncate<N2>().
TEST( TruncateDense, OrderReducingTypeAndCoeffs )
{
    // f = exp(x) at x0 = 0 : coeffs 1, 1, 1/2, 1/6, 1/24, 1/120
    auto x = tax::TE< 5 >::variable( 0.0 );
    auto f = tax::exp( x );

    auto g = f.truncate< 2 >();
    static_assert( decltype( g )::order_v == 2, "truncate<2> must lower the order to 2" );

    // Retained coefficients (degree <= 2) equal the originals.
    for ( std::size_t k = 0; k < g.nCoefficients; ++k )
        EXPECT_NEAR( g[k], f[k], 1e-12 ) << "flat index " << k;
}

TEST( TruncateDense, OrderReducingMultivariatePrefix )
{
    // f = (1 + x)*(1 + y) over order 3, 2 vars : has degree-0,1,2 terms.
    typename tax::TE< 3, 2 >::Input p{ 0.0, 0.0 };
    auto x = tax::TE< 3, 2 >::variable< 0 >( p );
    auto y = tax::TE< 3, 2 >::variable< 1 >( p );
    auto f = ( 1.0 + x ) * ( 1.0 + y );

    auto g = f.truncate< 1 >();
    static_assert( decltype( g )::order_v == 1 );
    // Prefix block [0, numMonomials(1,2)) is copied exactly.
    for ( std::size_t k = 0; k < g.nCoefficients; ++k )
        EXPECT_NEAR( g[k], f[k], 1e-12 ) << "flat index " << k;
}

TEST( TruncateDense, OrderReducingIsConstexpr )
{
    constexpr auto x = tax::TE< 4 >::variable( 0.0 );
    constexpr auto g = x.truncate< 2 >();
    static_assert( decltype( g )::order_v == 2 );
    static_assert( g[1] == 1.0 );  // x's linear term survives
}

// Dense: runtime, same-order truncate(d).
TEST( TruncateDense, RuntimeZeroesHighDegree )
{
    auto x = tax::TE< 5 >::variable( 0.0 );
    auto f = tax::exp( x );

    auto h = f.truncate( 2 );
    static_assert( decltype( h )::order_v == 5, "runtime form keeps the same order" );
    for ( std::size_t k = 0; k < f.nCoefficients; ++k )
    {
        const int deg = tax::totalDegree( tax::unflatIndex< 1 >( k ) );
        const double want = ( deg > 2 ) ? 0.0 : f[k];
        EXPECT_NEAR( h[k], want, 1e-12 ) << "flat index " << k << " (deg " << deg << ")";
    }
}

TEST( TruncateDense, RuntimeDGeqNIsFullCopy )
{
    auto x = tax::TE< 5 >::variable( 0.0 );
    auto f = tax::exp( x );
    auto h = f.truncate( 5 );
    tax::test::ExpectCoeffsNear( h, f, 1e-12 );
    auto h2 = f.truncate( 99 );
    tax::test::ExpectCoeffsNear( h2, f, 1e-12 );
}

TEST( TruncateDense, RuntimeNegativeDIsZero )
{
    auto x = tax::TE< 5 >::variable( 0.0 );
    auto f = tax::exp( x );
    auto h = f.truncate( -1 );
    for ( std::size_t k = 0; k < f.nCoefficients; ++k ) EXPECT_EQ( h[k], 0.0 );
}

// Named: both forms (axes preserved).
TEST( TruncateNamed, OrderReducingPreservesAxes )
{
    auto x = tax::variable< "x", 4 >( 0.0 );
    auto f = exp( x );  // ADL finds tax::named::exp (named math is not qualified-exported)

    auto g = f.truncate< 2 >();
    static_assert( decltype( g )::order_v == 2 );
    // Same axis set as f (assignable back into a width-checked named type).
    for ( std::size_t k = 0; k < g.inner().nCoefficients; ++k )
        EXPECT_NEAR( g.inner()[k], f.inner()[k], 1e-12 ) << "flat index " << k;
}

TEST( TruncateNamed, RuntimeSameType )
{
    auto x = tax::variable< "x", 4 >( 0.0 );
    auto f = exp( x );  // ADL finds tax::named::exp (named math is not qualified-exported)

    auto h = f.truncate( 2 );
    static_assert( decltype( h )::order_v == 4 );
    for ( std::size_t k = 0; k < f.inner().nCoefficients; ++k )
    {
        const int deg = tax::totalDegree( tax::unflatIndex< 1 >( k ) );
        const double want = ( deg > 2 ) ? 0.0 : f.inner()[k];
        EXPECT_NEAR( h.inner()[k], want, 1e-12 ) << "flat index " << k;
    }
}
