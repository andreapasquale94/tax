#include <dace/dace.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <tax/tax.hpp>

template < int N >
::testing::AssertionResult expectCoeffsMatch( const tax::TE< N >& tested, const DACE::DA& ref,
                                              double tol = 1e-12 )
{
    const auto& tax_coeffs = tested.coeffs();  // expected size N+1

    for ( unsigned int k = 0; k <= N; ++k )
    {
        const double c_dace = ref.getCoefficient( { k } );  // 1D multi-index
        const double c_tax = static_cast< double >( tax_coeffs[k] );

        if ( !( std::isfinite( c_dace ) && std::isfinite( c_tax ) ) )
        {
            return ::testing::AssertionFailure() << "Non-finite coefficient at k=" << k
                                                 << " (DACE=" << c_dace << ", tax=" << c_tax << ")";
        }

        const double diff = std::abs( c_dace - c_tax );
        if ( diff > tol )
        {
            return ::testing::AssertionFailure()
                   << "Coefficient mismatch at k=" << k << " (DACE=" << std::setprecision( 17 )
                   << c_dace << ", tax=" << std::setprecision( 17 ) << c_tax << ", |diff|=" << diff
                   << ", tol=" << tol << ")";
        }
    }

    return ::testing::AssertionSuccess();
}

// Operators

TEST( DaceUnivariate, Div )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = 1 / ( 1 + xr );

    auto x = tax::TE< N >::variable( 1.0 );
    tax::TE< N > y = 1.0 / x;

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, MulDiv )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = 1 / ( ( 1 + xr ) * ( 1 + xr ) );

    auto x = tax::TE< N >::variable( 1.0 );
    tax::TE< N > y = 1.0 / ( x * x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

// Math

TEST( DaceUnivariate, Cos )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.cos();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::cos( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Sin )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.sin();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::sin( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Tan )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.tan();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::tan( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ASin )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.asin();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::asin( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ACos )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.acos();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::acos( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ATan )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.atan();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::atan( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Cosh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.cosh();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::cosh( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Sinh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.sinh();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::sinh( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Tanh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.tanh();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::tanh( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ASinh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( xr + ( xr * xr + 1 ).sqrt() ).log();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::asinh( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ACosh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( ( xr + 2 ) + ( ( xr + 2 ) * ( xr + 2 ) - 1 ).sqrt() ).log();

    auto x = tax::TE< N >::variable( 2.0 );
    tax::TE< N > y = tax::acosh( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ATanh )
{
    constexpr int N = 20;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.atanh();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::atanh( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr, 1e-9 ) );
}

TEST( DaceUnivariate, Exp )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.exp();

    auto x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::exp( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Log )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 1 + xr ).log();

    auto x = tax::TE< N >::variable( 1.0 );
    tax::TE< N > y = tax::log( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Log10 )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 1 + xr ).log10();

    auto x = tax::TE< N >::variable( 1.0 );
    tax::TE< N > y = tax::log10( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Sqrt )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 2 + xr ).sqrt();

    auto x = tax::TE< N >::variable( 2.0 );
    tax::TE< N > y = tax::sqrt( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Cbrt )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 8 + xr ).pow( 1.0 / 3.0 );

    auto x = tax::TE< N >::variable( 8.0 );
    tax::TE< N > y = tax::cbrt( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Erf )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 2 + xr ).erf();

    auto x = tax::TE< N >::variable( 2.0 );
    tax::TE< N > y = tax::erf( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, IPow )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 2 + xr ).pow( 5 );

    auto x = tax::TE< N >::variable( 2.0 );
    tax::TE< N > y = tax::pow( x, 5 );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Pow )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 2 + xr ).pow( 0.5 );

    auto x = tax::TE< N >::variable( 2.0 );
    tax::TE< N > y = tax::pow( x, 0.5 );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}
