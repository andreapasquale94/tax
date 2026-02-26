#include <gtest/gtest.h>

#include "tax/utils.hpp"

using namespace tax::detail;

TEST( Binom, Basics )
{
    EXPECT_EQ( binom( 0, 0 ), 1u );
    EXPECT_EQ( binom( 5, 0 ), 1u );
    EXPECT_EQ( binom( 5, 5 ), 1u );
    EXPECT_EQ( binom( 5, 1 ), 5u );
    EXPECT_EQ( binom( 5, 2 ), 10u );
    EXPECT_EQ( binom( 10, 3 ), 120u );
}

TEST( Binom, SymmetryAndBounds )
{
    // Symmetry: C(n, k) == C(n, n-k)
    EXPECT_EQ( binom( 8, 3 ), binom( 8, 5 ) );
    EXPECT_EQ( binom( 12, 2 ), binom( 12, 10 ) );

    // k > n -> 0
    EXPECT_EQ( binom( 4, 5 ), 0u );
    EXPECT_EQ( binom( 0, 1 ), 0u );
}

TEST( Binom, ReasonableRangeNoOverflow )
{
    // This implementation uses size_t and will overflow for large n; here we keep it small.
    EXPECT_EQ( binom( 20, 10 ), 184756u );
}

// ----

TEST( Factorial, SmallValues )
{
    EXPECT_EQ( factorial( 0 ), 1u );
    EXPECT_EQ( factorial( 1 ), 1u );
    EXPECT_EQ( factorial( 2 ), 2u );
    EXPECT_EQ( factorial( 3 ), 6u );
    EXPECT_EQ( factorial( 4 ), 24u );
    EXPECT_EQ( factorial( 5 ), 120u );
    EXPECT_EQ( factorial( 10 ), 3628800u );
}

TEST( MultiIndex, Degree )
{
    {
        std::array< std::size_t, 3 > a{ 1, 2, 3 };
        EXPECT_EQ( degree( a ), 6u );
    }
    {
        std::array< std::size_t, 4 > a{ 0, 0, 0, 0 };
        EXPECT_EQ( degree( a ), 0u );
    }
    {
        std::array< std::size_t, 2 > a{ 5, 7 };
        EXPECT_EQ( degree( a ), 12u );
    }
}

TEST( MultiIndex, Leq )
{
    {
        std::array< std::size_t, 3 > a{ 1, 2, 3 };
        std::array< std::size_t, 3 > b{ 1, 2, 3 };
        EXPECT_TRUE( leq( a, b ) );
        EXPECT_TRUE( leq( b, a ) );
    }
    {
        std::array< std::size_t, 3 > a{ 0, 1, 2 };
        std::array< std::size_t, 3 > b{ 1, 1, 2 };
        EXPECT_TRUE( leq( a, b ) );
        EXPECT_FALSE( leq( b, a ) );
    }
    {
        std::array< std::size_t, 3 > a{ 0, 0, 1 };
        std::array< std::size_t, 3 > b{ 0, 0, 0 };
        EXPECT_FALSE( leq( a, b ) );
    }
}

TEST( MultiIndex, Multifactorial )
{
    {
        std::array< std::size_t, 3 > a{ 0, 0, 0 };
        EXPECT_EQ( multifactorial( a ), 1u );  // 0! * 0! * 0! = 1
    }
    {
        std::array< std::size_t, 3 > a{ 1, 2, 3 };
        // 1! * 2! * 3! = 1 * 2 * 6 = 12
        EXPECT_EQ( multifactorial( a ), 12u );
    }
    {
        std::array< std::size_t, 4 > a{ 1, 4, 0, 2 };
        // 1! * 4! * 0! * 2! = 1 * 24 * 1 * 2 = 48
        EXPECT_EQ( multifactorial( a ), 48u );
    }
}
