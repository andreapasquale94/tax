#include <gtest/gtest.h>

#include <cmath>
#include <tax/tax.hpp>

// Binary-math parity for named + mixed expansions: pow(x,int/real),
// pow(x,x), pow(scalar,x), and the three atan2 forms must all resolve and
// match the inner (anonymous) expansion's result.

TEST( NamedMixedBinary, NamedPowAtan2Parity )
{
    auto x = tax::variable< "x", 1 >( 1.3 );  // NE<1, Axis<"x",1>> (TaylorBasis)
    auto y = tax::variable< "y", 1 >( 0.7 );

    // pow(NE, NE) and pow(scalar, NE)
    auto p_nn = pow( x, y );
    auto p_sn = pow( 2.0, x );
    EXPECT_NEAR( p_nn.value(), std::pow( 1.3, 0.7 ), 1e-10 );
    EXPECT_NEAR( p_sn.value(), std::pow( 2.0, 1.3 ), 1e-10 );

    // atan2(NE, const) and atan2(const, NE)
    auto a_nc = atan2( x, 2.0 );
    auto a_cn = atan2( 2.0, x );
    EXPECT_NEAR( a_nc.value(), std::atan2( 1.3, 2.0 ), 1e-10 );
    EXPECT_NEAR( a_cn.value(), std::atan2( 2.0, 1.3 ), 1e-10 );
}

TEST( NamedMixedBinary, MixedPowAtan2 )
{
    auto x = tax::mixed::variable< "x", 4 >( 1.3 );  // MTE
    auto y = tax::mixed::variable< "y", 4 >( 0.7 );

    auto p_int = pow( x, 2 );
    auto p_mm  = pow( x, y );
    auto p_sm  = pow( 2.0, x );
    auto a_mm  = atan2( x, y );
    auto a_mc  = atan2( x, 2.0 );
    auto a_cm  = atan2( 2.0, x );

    EXPECT_NEAR( p_int.value(), 1.3 * 1.3, 1e-10 );
    EXPECT_NEAR( p_mm.value(), std::pow( 1.3, 0.7 ), 1e-10 );
    EXPECT_NEAR( p_sm.value(), std::pow( 2.0, 1.3 ), 1e-10 );
    EXPECT_NEAR( a_mm.value(), std::atan2( 1.3, 0.7 ), 1e-10 );
    EXPECT_NEAR( a_mc.value(), std::atan2( 1.3, 2.0 ), 1e-10 );
    EXPECT_NEAR( a_cm.value(), std::atan2( 2.0, 1.3 ), 1e-10 );
}
