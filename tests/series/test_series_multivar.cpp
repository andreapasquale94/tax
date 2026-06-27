#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/tax.hpp>

using tax::ChebyshevSeries;
using tax::TaylorSeries;

// ---------------------------------------------------------------------------
// Multivariate Taylor
// ---------------------------------------------------------------------------

TEST( SeriesMultivar, TaylorEvalAndProduct )
{
    constexpr int N = 4;
    auto x = TaylorSeries< N, 2 >::variable< 0 >();
    auto y = TaylorSeries< N, 2 >::variable< 1 >();
    auto f = 1.0 + x + 2.0 * y;  // degree 1
    auto g = 1.0 - x + y;        // degree 1
    auto h = f * g;              // degree 2 <= N, exact

    for ( auto pt : { std::array< double, 2 >{ 0.3, -0.4 }, std::array< double, 2 >{ -0.7, 0.5 } } )
    {
        EXPECT_NEAR( f.eval( pt ), 1.0 + pt[0] + 2.0 * pt[1], 1e-13 );
        EXPECT_NEAR( h.eval( pt ), f.eval( pt ) * g.eval( pt ), 1e-13 );
    }
}

TEST( SeriesMultivar, TaylorPartialDerivatives )
{
    constexpr int N = 4;
    auto x = TaylorSeries< N, 2 >::variable< 0 >();
    auto y = TaylorSeries< N, 2 >::variable< 1 >();
    auto f = x * x * y + 3.0 * y;  // ∂/∂x = 2xy, ∂/∂y = x^2 + 3
    auto fx = f.deriv< 0 >();
    auto fy = f.deriv< 1 >();
    std::array< double, 2 > pt{ 0.5, -0.2 };
    EXPECT_NEAR( fx.eval( pt ), 2.0 * pt[0] * pt[1], 1e-13 );
    EXPECT_NEAR( fy.eval( pt ), pt[0] * pt[0] + 3.0, 1e-13 );
}

TEST( SeriesMultivar, TaylorTranscendentalMultivariate )
{
    // exp of a 2-var Taylor series uses the existing scheme-generic kernel.
    constexpr int N = 6;
    auto x = TaylorSeries< N, 2 >::variable< 0 >();
    auto y = TaylorSeries< N, 2 >::variable< 1 >();
    auto f = exp( x + y );
    std::array< double, 2 > pt{ 0.2, 0.1 };
    EXPECT_NEAR( f.eval( pt ), std::exp( pt[0] + pt[1] ), 1e-6 );
}

// ---------------------------------------------------------------------------
// Multivariate Chebyshev (tensor product)
// ---------------------------------------------------------------------------

TEST( SeriesMultivar, ChebyshevVariableEval )
{
    auto x = ChebyshevSeries< 4, 2 >::variable< 0 >();
    auto y = ChebyshevSeries< 4, 2 >::variable< 1 >();
    std::array< double, 2 > pt{ 0.3, 0.7 };
    EXPECT_NEAR( x.eval( pt ), 0.3, 1e-13 );
    EXPECT_NEAR( y.eval( pt ), 0.7, 1e-13 );
}

TEST( SeriesMultivar, ChebyshevTensorProductExact )
{
    // f, g of total degree <= 2 -> product degree <= 4 stays in box -> exact.
    constexpr int N = 4;
    auto x = ChebyshevSeries< N, 2 >::variable< 0 >();
    auto y = ChebyshevSeries< N, 2 >::variable< 1 >();
    auto f = 1.0 + 2.0 * x + y;
    auto g = 0.5 - x + 3.0 * y;
    auto h = f * g;
    for ( auto pt : { std::array< double, 2 >{ 0.2, -0.5 }, std::array< double, 2 >{ -0.8, 0.6 },
                      std::array< double, 2 >{ 0.9, 0.9 } } )
        EXPECT_NEAR( h.eval( pt ), f.eval( pt ) * g.eval( pt ), 1e-12 );
}

TEST( SeriesMultivar, ChebyshevPartialDerivativeNumeric )
{
    constexpr int N = 5;
    auto x = ChebyshevSeries< N, 2 >::variable< 0 >();
    auto y = ChebyshevSeries< N, 2 >::variable< 1 >();
    auto f = 1.0 + 2.0 * x + y + 0.5 * ( x * y );
    auto fx = f.deriv< 0 >();
    auto fy = f.deriv< 1 >();
    const double h = 1e-6;
    std::array< double, 2 > pt{ 0.3, -0.4 };
    auto fxn = ( f.eval( { pt[0] + h, pt[1] } ) - f.eval( { pt[0] - h, pt[1] } ) ) / ( 2 * h );
    auto fyn = ( f.eval( { pt[0], pt[1] + h } ) - f.eval( { pt[0], pt[1] - h } ) ) / ( 2 * h );
    EXPECT_NEAR( fx.eval( pt ), fxn, 1e-6 );
    EXPECT_NEAR( fy.eval( pt ), fyn, 1e-6 );
}
