#include <gtest/gtest.h>


#include "tax/kernel.hpp"
#include "tax/md_kernel.hpp"
#include "tax/_testutils.hpp"


using namespace tax::detail;
using FloatTypes = ::testing::Types< float, double, long double >;

// -------------------------------------------------------------------------------------------
template < typename T >
class Kernel1D : public ::testing::Test
{
};
TYPED_TEST_SUITE( Kernel1D, FloatTypes );

TYPED_TEST( Kernel1D, CauchyProduct )
{
    constexpr std::size_t N = 4;
    using T = TypeParam;

    // a(x) = b(x) = 2 + 1*x + 0*x^2 + 3*x^3
    std::array< T, N + 1 > a{ 2, 1, 0, 3, 0 };
    std::array< T, N + 1 > b{ 2, 1, 0, 3, 0 };
    std::array< T, N + 1 > out{ 0, 0, 0, 0, 0 };

    const std::array< T, N + 1 > expected{ 4, 4, 1, 12, 6 };

    seriesProduct1D< T, N >( out, a, b );

    expectArrayNear( out, expected );
}

TYPED_TEST( Kernel1D, SeriesReciprocal )
{
    constexpr std::size_t N = 10;
    using T = TypeParam;

    // a(x) = 2 + 1*x + 0*x^2 + 3*x^3
    std::array< T, N + 1 > a{ 2, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0 };
    std::array< T, N + 1 > out;

    // Reference:
    const std::array< T, N + 1 > expected{
        0.5,       -0.25,       0.125,       -0.8125,       0.78125,      -0.578125,
        1.5078125, -1.92578125, 1.830078125, -3.1767578125, 4.47705078125 };

    seriesReciprocal1D< T, N >( out, a );

    expectArrayNear( out, expected );
}

TYPED_TEST( Kernel1D, SeriesDivision )
{
    constexpr std::size_t N = 10;
    using T = TypeParam;

    // num(x) = 1 + 2x + 3x^2 + 4x^3
    // den(x) = 2 + 1x + 0x^2 + 1x^3
    std::array< T, N + 1 > num{ 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0 };
    std::array< T, N + 1 > den{ 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    std::array< T, N + 1 > out;

    const std::array< T, N + 1 > expected{
        0.5,        0.75,       1.125,        1.1875,       -0.96875,       -0.078125,
        -0.5546875, 0.76171875, -0.341796875, 0.4482421875, -0.60498046875,
    };

    seriesDivision1D< T, N >( out, num, den );

    expectArrayNear( out, expected );
}

// -------------------------------------------------------------------------------------------
template < typename T >
class Kernel2D : public ::testing::Test
{
};
TYPED_TEST_SUITE( Kernel2D, FloatTypes );

TYPED_TEST( Kernel2D, CauchyProduct )
{
    constexpr std::size_t N = 4;
    constexpr std::size_t D = 2;
    constexpr std::size_t M = countMonomials< N, D >();
    using IdxSet = IndexSet< N, D >;
    using T = TypeParam;

    std::array< T, M > a{};
    std::array< T, M > b{};
    std::array< T, M > out{};
    std::array< T, M > expected{};
    expected.fill( T{ 0 } );

    // p(x1) = 1.0 + 1.0x1
    a[IdxSet::indexOf( { 0, 0 } )] = 1.0;
    a[IdxSet::indexOf( { 1, 0 } )] = 1.0;
    // p(x2) = 2.0 + 1.0x2
    b[IdxSet::indexOf( { 0, 0 } )] = 2.0;
    b[IdxSet::indexOf( { 0, 1 } )] = 1.0;

    // p(x1) * p(x2)
    expected[IdxSet::indexOf( { 0, 0 } )] = 2.0;
    expected[IdxSet::indexOf( { 1, 0 } )] = 2.0;
    expected[IdxSet::indexOf( { 0, 1 } )] = 1.0;
    expected[IdxSet::indexOf( { 1, 1 } )] = 1.0;

    seriesProductND< T, N, D >( out, a, b );

    expectArrayNear( out, expected );
}

TYPED_TEST( Kernel2D, SeriesReciprocal )
{
    constexpr std::size_t N = 3;
    constexpr std::size_t D = 2;
    constexpr std::size_t M = countMonomials< N, D >();
    using IdxSet = IndexSet< N, D >;
    using T = TypeParam;

    std::array< T, M > a{};
    std::array< T, M > out{};
    std::array< T, M > expected{};
    expected.fill( T{ 0 } );

    // p(x1) = 1.0 + 1.0x1
    // p(x2) = 2.0 + 1.0x2
    // g(x1, x2) = p(x1) * p(x2)
    a[IdxSet::indexOf( { 0, 0 } )] = 2.0;
    a[IdxSet::indexOf( { 1, 0 } )] = 2.0;
    a[IdxSet::indexOf( { 0, 1 } )] = 1.0;
    a[IdxSet::indexOf( { 1, 1 } )] = 1.0;

    // 1 / g(x1, x2)
    expected[IdxSet::indexOf( { 0, 0 } )] = 0.5;
    expected[IdxSet::indexOf( { 1, 0 } )] = -0.5;
    expected[IdxSet::indexOf( { 0, 1 } )] = -0.25;
    expected[IdxSet::indexOf( { 2, 0 } )] = 0.5;
    expected[IdxSet::indexOf( { 1, 1 } )] = 0.25;
    expected[IdxSet::indexOf( { 0, 2 } )] = 0.125;
    expected[IdxSet::indexOf( { 3, 0 } )] = -0.5;
    expected[IdxSet::indexOf( { 2, 1 } )] = -0.25;
    expected[IdxSet::indexOf( { 1, 2 } )] = -0.125;
    expected[IdxSet::indexOf( { 0, 3 } )] = -0.0625;

    seriesReciprocalND< T, N, D >( out, a );

    expectArrayNear( out, expected );
}

TYPED_TEST( Kernel2D, SeriesDivision )
{
    constexpr std::size_t N = 3;
    constexpr std::size_t D = 2;
    constexpr std::size_t M = countMonomials< N, D >();
    using IdxSet = IndexSet< N, D >;
    using T = TypeParam;

    std::array< T, M > a{};
    std::array< T, M > b{};
    std::array< T, M > out{};
    std::array< T, M > expected{};
    expected.fill( T{ 0 } );

    // p(x1) = 1.0 + 1.0x1
    a[IdxSet::indexOf( { 0, 0 } )] = 1.0;
    a[IdxSet::indexOf( { 1, 0 } )] = 1.0;
    // p(x2) = 2.0 + 1.0x2
    b[IdxSet::indexOf( { 0, 0 } )] = 2.0;
    b[IdxSet::indexOf( { 0, 1 } )] = 1.0;

    // p(x1)/p(x2)
    expected[IdxSet::indexOf( { 0, 0 } )] = 0.5;
    expected[IdxSet::indexOf( { 1, 0 } )] = 0.5;
    expected[IdxSet::indexOf( { 0, 1 } )] = -0.25;
    expected[IdxSet::indexOf( { 1, 1 } )] = -0.25;
    expected[IdxSet::indexOf( { 0, 2 } )] = 0.125;
    expected[IdxSet::indexOf( { 1, 2 } )] = 0.125;
    expected[IdxSet::indexOf( { 0, 3 } )] = -0.0625;

    seriesDivisionND< T, N, D >( out, a, b );

    expectArrayNear( out, expected );
}

// -------------------------------------------------------------------------------------------
template < typename T >
class ArrayOps : public ::testing::Test
{
};
TYPED_TEST_SUITE( ArrayOps, FloatTypes );

TYPED_TEST( ArrayOps, Add )
{
    using T = TypeParam;
    std::array< T, 4 > a{ T( 1 ), T( 2 ), T( 3 ), T( 4 ) };
    std::array< T, 4 > b{ T( 0.5 ), T( -1 ), T( 2.5 ), T( 0 ) };
    add( a, b );
    EXPECT_NEAR( static_cast< double >( a[0] ), 1.5, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[1] ), 1.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[2] ), 5.5, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[3] ), 4.0, 1e-12 );
}

TYPED_TEST( ArrayOps, Sub )
{
    using T = TypeParam;
    std::array< T, 3 > a{ T( 5 ), T( 0 ), T( -2 ) };
    std::array< T, 3 > b{ T( 1 ), T( 2 ), T( 3 ) };
    sub( a, b );
    EXPECT_NEAR( static_cast< double >( a[0] ), 4.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[1] ), -2.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[2] ), -5.0, 1e-12 );
}

TYPED_TEST( ArrayOps, Negate )
{
    using T = TypeParam;
    std::array< T, 5 > a{ T( 0 ), T( 1 ), T( -1.5 ), T( 2.25 ), T( -3.5 ) };
    negate( a );
    EXPECT_NEAR( static_cast< double >( a[0] ), -0.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[1] ), -1.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[2] ), 1.5, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[3] ), -2.25, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[4] ), 3.5, 1e-12 );
}

TYPED_TEST( ArrayOps, Scale )
{
    using T = TypeParam;
    std::array< T, 4 > a{ T( 1 ), T( -2 ), T( 3.5 ), T( 0 ) };
    scale( a, T( 2.0 ) );
    EXPECT_NEAR( static_cast< double >( a[0] ), 2.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[1] ), -4.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[2] ), 7.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[3] ), 0.0, 1e-12 );
}

TYPED_TEST( ArrayOps, ScaleInv )
{
    using T = TypeParam;
    std::array< T, 4 > a{ T( 1 ), T( -2 ), T( 3.5 ), T( 0 ) };
    scaleInv( a, T( 2.0 ) );
    EXPECT_NEAR( static_cast< double >( a[0] ), 0.5, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[1] ), -1.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[2] ), 3.5 / 2, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[3] ), 0.0, 1e-12 );
}

TYPED_TEST( ArrayOps, ScaleByZero )
{
    using T = TypeParam;
    std::array< T, 3 > a{ T( 1 ), T( -2.5 ), T( 4 ) };
    scale( a, T( 0 ) );
    EXPECT_NEAR( static_cast< double >( a[0] ), 0.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[1] ), 0.0, 1e-12 );
    EXPECT_NEAR( static_cast< double >( a[2] ), 0.0, 1e-12 );
}
