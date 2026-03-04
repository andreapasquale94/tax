#include <dace/dace.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <tax/tax.hpp>

template < int N, int M >
::testing::AssertionResult expectCoeffsMatch( const tax::TEn< N, M >& tested, const DACE::DA& ref,
                                              double tol = 1e-12 )
{
    for ( unsigned int k = 0; k < tax::detail::numMonomials( N, M ); ++k )
    {
        const auto index = tax::detail::unflatIndex< M >( k );
        const double c_tax = tested.coeff( index );

        const std::vector< unsigned int > vindex( index.begin(), index.end() );
        const double c_dace = ref.getCoefficient( vindex );

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

TEST( DaceMultivariate, Div )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 2.0;
    auto zr = dzr + 3.0;
    auto vr = 1 / ( xr * yr * zr );

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 2.0, 3.0 } );
    tax::TEn< N, 3 > v = 1 / ( x * y * z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

// Operations

TEST( DaceMultivariate, Sin )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 1.0;
    auto zr = dzr + 1.0;

    auto vr = ( xr * yr / zr ).sin();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 1.0, 1.0 } );
    tax::TEn< N, 3 > v = tax::sin( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Cos )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 1.0;
    auto zr = dzr + 1.0;

    auto vr = ( xr * yr / zr ).cos();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 1.0, 1.0 } );
    tax::TEn< N, 3 > v = tax::cos( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Tan )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 0.0;
    auto yr = dyr + 0.0;
    auto zr = dzr + 0.5;

    auto vr = ( xr * yr / zr ).tan();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 0.0, 0.0, 0.5 } );
    tax::TEn< N, 3 > v = tax::tan( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Sinh )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 1.0;
    auto zr = dzr + 1.0;

    auto vr = ( xr * yr / zr ).sinh();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 1.0, 1.0 } );
    tax::TEn< N, 3 > v = tax::sinh( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Cosh )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 1.0;
    auto zr = dzr + 1.0;

    auto vr = ( xr * yr / zr ).cosh();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 1.0, 1.0 } );
    tax::TEn< N, 3 > v = tax::cosh( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Tanh )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 0.0;
    auto yr = dyr + 0.0;
    auto zr = dzr + 0.5;

    auto vr = ( xr * yr / zr ).tanh();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 0.0, 0.0, 0.5 } );
    tax::TEn< N, 3 > v = tax::tanh( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, ASinh )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 0.5;
    auto zr = dzr + 1.0;

    auto wr = xr * yr / zr;
    auto vr = ( wr + ( wr * wr + 1 ).sqrt() ).log();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 0.5, 1.0 } );
    tax::TEn< N, 3 > v = tax::asinh( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, ACosh )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 2.0;
    auto yr = dyr + 1.0;
    auto zr = dzr + 1.0;

    auto wr = xr * yr / zr;
    auto vr = ( wr + ( wr * wr - 1 ).sqrt() ).log();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 2.0, 1.0, 1.0 } );
    tax::TEn< N, 3 > v = tax::acosh( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, ATanh )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 0.2;
    auto yr = dyr + 0.5;
    auto zr = dzr + 1.0;

    auto wr = xr * yr / zr;
    auto vr = 0.5 * ( ( 1 + wr ) / ( 1 - wr ) ).log();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 0.2, 0.5, 1.0 } );
    tax::TEn< N, 3 > v = tax::atanh( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, ASin )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 0.9;
    auto yr = dyr + 0.5;
    auto zr = dzr + 1.0;

    auto vr = ( xr * yr / zr ).asin();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 0.9, 0.5, 1.0 } );
    tax::TEn< N, 3 > v = tax::asin( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, ACos )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 0.9;
    auto yr = dyr + 0.5;
    auto zr = dzr + 1.0;

    auto vr = ( xr * yr / zr ).acos();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 0.9, 0.5, 1.0 } );
    tax::TEn< N, 3 > v = tax::acos( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, ATan )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 0.0;
    auto yr = dyr + 0.0;
    auto zr = dzr + 0.5;

    auto vr = ( xr * yr / zr ).atan();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 0.0, 0.0, 0.5 } );
    tax::TEn< N, 3 > v = tax::atan( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Log )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 2.0;
    auto zr = dzr + 0.5;

    auto vr = ( xr * yr / zr ).log();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 2.0, 0.5 } );
    tax::TEn< N, 3 > v = tax::log( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Log10 )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 2.0;
    auto zr = dzr + 0.5;

    auto vr = ( xr * yr / zr ).log10();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 2.0, 0.5 } );
    tax::TEn< N, 3 > v = tax::log10( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Exp )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 0.2;
    auto zr = dzr + 0.5;

    auto vr = ( xr * yr / zr ).exp();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 0.2, 0.5 } );
    tax::TEn< N, 3 > v = tax::exp( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Sqrt )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 2.0;
    auto zr = dzr + 0.5;

    auto vr = ( xr * yr / zr ).sqrt();

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 2.0, 0.5 } );
    tax::TEn< N, 3 > v = tax::sqrt( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Cbrt )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 4.0;
    auto yr = dyr + 2.0;
    auto zr = dzr + 1.0;

    auto vr = ( xr * yr / zr ).pow( 1.0 / 3.0 );

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 4.0, 2.0, 1.0 } );
    tax::TEn< N, 3 > v = tax::cbrt( x * y / z );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, IPow )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 2.0;
    auto zr = dzr + 0.5;

    auto vr = ( xr * yr / zr ).pow( 4 );

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 2.0, 0.5 } );
    tax::TEn< N, 3 > v = tax::pow( x * y / z, 4 );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}

TEST( DaceMultivariate, Pow )
{
    constexpr int N = 5;

    DACE::DA::init( N, 3 );
    DACE::DA dxr( 1 );
    DACE::DA dyr( 2 );
    DACE::DA dzr( 3 );

    auto xr = dxr + 1.0;
    auto yr = dyr + 2.0;
    auto zr = dzr + 0.5;

    auto vr = ( xr * yr / zr ).pow( 0.5 );

    auto [x, y, z] = tax::TEn< N, 3 >::variables( { 1.0, 2.0, 0.5 } );
    tax::TEn< N, 3 > v = tax::pow( x * y / z, 0.5 );

    EXPECT_TRUE( expectCoeffsMatch( v, vr ) );
}
