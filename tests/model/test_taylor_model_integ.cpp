#include <gtest/gtest.h>

#include <cmath>

#include "modelTestUtils.hpp"

using I = tax::Interval< double >;

// ---------------------------------------------------------------------------
// Antiderivation (4.12)
// ---------------------------------------------------------------------------

TEST( TMInteg, CosineIntegratesToSine )
{
    // F(dx) = integral of cos from 0 to dx; compare with sin(dx).
    const auto x = tax::TM< 6 >::variable( 0.0, I{ -0.8, 0.8 } );
    const auto f = cos( x );
    const auto F = f.integ< 0 >();

    // Polynomial part: dx - dx^3/6 + dx^5/120 (the integral of P5(cos)).
    EXPECT_NEAR( F.polynomial()[0], 0.0, 1e-15 );
    EXPECT_NEAR( F.polynomial()[1], 1.0, 1e-13 );
    EXPECT_NEAR( F.polynomial()[2], 0.0, 1e-13 );
    EXPECT_NEAR( F.polynomial()[3], -1.0 / 6.0, 1e-13 );
    EXPECT_NEAR( F.polynomial()[5], 1.0 / 120.0, 1e-13 );

    // Containment of the true antiderivative sin(dx) (x0 = 0).
    tax::test::ExpectEncloses( F, []( const auto& p ) { return std::sin( p[0] ); }, 41 );

    // Remainder scale: (I^6 + I) * hull(0, D) with |D| = 0.8.
    EXPECT_LT( F.remainder().width(), 1e-3 );
}

TEST( TMInteg, DefiniteIntegralEnclosure )
{
    // (4.13): definite integral bounds from the antiderivative Taylor model.
    // integral_{-1/2}^{1/2} cos = 2 sin(0.5) = 0.958851...
    const auto x = tax::TM< 8 >::variable( 0.0, I{ -0.5, 0.5 } );
    const auto F = cos( x ).integ< 0 >();

    const auto upper = F.eval( { 0.5 } );
    const auto lower = F.eval( { -0.5 } );
    const auto integral = upper - lower;

    const double exact = 2.0 * std::sin( 0.5 );
    EXPECT_TRUE( integral.contains( exact ) );
    EXPECT_LT( integral.width(), 1e-4 );
}

TEST( TMInteg, RemainderAbsorbsFreedTopOrder )
{
    // f = dx^2 at order 2: integrating drops the entire polynomial into the
    // freed top-order block, so F's polynomial is 0 and the remainder must
    // hold [0, max dx^2] * hull(0, D) = [-1/3 ..ish scale] conservatively.
    const auto x = tax::TM< 2 >::variable( 0.0, I{ -1.0, 1.0 } );
    const auto f = x * x;
    const auto F = f.integ< 0 >();
    for ( std::size_t k = 0; k < F.nCoefficients; ++k ) EXPECT_DOUBLE_EQ( F.polynomial()[k], 0.0 );
    // True antiderivative dx^3/3 has range [-1/3, 1/3] on [-1, 1].
    tax::test::ExpectEncloses( F, []( const auto& p ) { return p[0] * p[0] * p[0] / 3.0; } );
}

TEST( TMInteg, MultivariatePartialIntegral )
{
    // f(x, y) = x y integrated over x from 0: x^2 y / 2, exactly representable.
    using TMT = tax::TM< 3, 2 >;
    const TMT::Point x0{ 0.0, 0.0 };
    const TMT::Domain dom{ I{ -1.0, 1.0 }, I{ -1.0, 1.0 } };
    const auto x = TMT::variable< 0 >( x0, dom );
    const auto y = TMT::variable< 1 >( x0, dom );

    const auto F = ( x * y ).integ< 0 >();
    tax::test::ExpectEncloses( F, []( const auto& p ) { return p[0] * p[0] * p[1] / 2.0; }, 9 );

    // Runtime-index form agrees; invalid index throws.
    const auto G = ( x * y ).integ( 0 );
    EXPECT_TRUE( G.remainder().contains( F.remainder() ) );
    EXPECT_THROW( (void)( x * y ).integ( 5 ), std::out_of_range );
}
