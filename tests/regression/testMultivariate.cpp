// tests/regression/testMultivariate.cpp
//
// DACE-vs-tax parity for multivariate ops on tax::TE<N, M>, using the
// Eigen-form variables factory. Every input variable is wrapped in
// tax_regression::prepareInput before the op under test, on both sides.

#include <dace/dace.h>
#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "regressionUtils.hpp"

using tax_regression::expectCoeffsMatch;
using tax_regression::prepareInput;

namespace
{
// Scale prepareInput from [1, 1.5] down to [0, 0.4] for ops whose domain
// excludes the default prep range. Applied identically on both sides.
inline DACE::DA scaleToUnit( const DACE::DA& x )
{
    return ( prepareInput( x ) - 1.0 ) * 0.8;
}
template < int N, int M >
inline tax::TE< N, M > scaleToUnit( const tax::TE< N, M >& x )
{
    return ( prepareInput( x ) - 1.0 ) * 0.8;
}
}  // namespace

// -----------------------------------------------------------------------
// Operators
// -----------------------------------------------------------------------

TEST( DaceMultivariate, Div )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA fxr = prepareInput( dxr );
    DACE::DA fyr = prepareInput( dyr );
    DACE::DA fzr = prepareInput( dzr );
    DACE::DA vr  = 1.0 / ( fxr * fyr * fzr );

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v      = tax::variables< tax::TE< N, M > >( x0 );
    auto                  fx     = prepareInput( v( 0 ) );
    auto                  fy     = prepareInput( v( 1 ) );
    auto                  fz     = prepareInput( v( 2 ) );
    tax::TE< N, M >       result = tax::reciprocal( fx * fy * fz );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

// -----------------------------------------------------------------------
// Math: trig
// -----------------------------------------------------------------------

TEST( DaceMultivariate, Sin )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).sin();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::sin( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, Cos )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).cos();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::cos( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, Tan )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).tan();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::tan( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

// asin/acos need argument in [-1, 1]; scale combined product to that range.
TEST( DaceMultivariate, ASin )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( scaleToUnit( dxr ) * scaleToUnit( dyr ) / ( 1.0 + scaleToUnit( dzr ) ) ).asin();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result = tax::asin(
        scaleToUnit( v( 0 ) ) * scaleToUnit( v( 1 ) ) / ( 1.0 + scaleToUnit( v( 2 ) ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, ACos )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( scaleToUnit( dxr ) * scaleToUnit( dyr ) / ( 1.0 + scaleToUnit( dzr ) ) ).acos();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result = tax::acos(
        scaleToUnit( v( 0 ) ) * scaleToUnit( v( 1 ) ) / ( 1.0 + scaleToUnit( v( 2 ) ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, ATan )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).atan();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::atan( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

// -----------------------------------------------------------------------
// Math: hyperbolic
// -----------------------------------------------------------------------

TEST( DaceMultivariate, Sinh )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).sinh();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::sinh( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, Cosh )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).cosh();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::cosh( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, Tanh )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).tanh();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::tanh( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, ASinh )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).asinh();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::asinh( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

// acosh: constant value of (prep^2/prep) is 1 — domain boundary. Shift by +1.
TEST( DaceMultivariate, ACosh )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr =
        ( 1.0 + prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).acosh();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result = tax::acosh(
        1.0 + prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, ATanh )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr =
        ( scaleToUnit( dxr ) * scaleToUnit( dyr ) / ( 1.0 + scaleToUnit( dzr ) ) ).atanh();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result = tax::atanh(
        scaleToUnit( v( 0 ) ) * scaleToUnit( v( 1 ) ) / ( 1.0 + scaleToUnit( v( 2 ) ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

// -----------------------------------------------------------------------
// Math: exp / log / roots / power
// -----------------------------------------------------------------------

TEST( DaceMultivariate, Exp )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).exp();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::exp( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, Log )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).log();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::log( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, Sqrt )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).sqrt();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::sqrt( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, Cbrt )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).cbrt();

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::cbrt( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, IPow )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).pow( 4 );

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result =
        tax::pow( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ), 4 );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}

TEST( DaceMultivariate, Pow )
{
    constexpr int N = 5;
    constexpr int M = 3;

    DACE::DA::init( N, M );
    DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
    DACE::DA vr = ( prepareInput( dxr ) * prepareInput( dyr ) / prepareInput( dzr ) ).pow( 0.5 );

    const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
    auto                  v = tax::variables< tax::TE< N, M > >( x0 );
    tax::TE< N, M >       result = tax::pow(
        prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ), 0.5 );

    EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
}
