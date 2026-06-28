#include <gtest/gtest.h>

#include <Eigen/Core>
#include <tax/tax.hpp>

// Basis-generic tax::la: point-form gradient/Hessian/Jacobian and Eigen
// NumTraits work for every polynomial family, not just Taylor. The differential
// operators of a polynomial are basis-independent, so the same analytic answers
// hold whatever basis represents the polynomial.

template < typename Series2 >
static void checkGradientHessian()
{
    auto x = Series2::template variable< 0 >();
    auto y = Series2::template variable< 1 >();
    auto f = x * y + 2.0 * x;  // ∇f = (y+2, x);  H = [[0,1],[1,0]]

    for ( Eigen::Vector2d pt : { Eigen::Vector2d{ 0.3, -0.4 }, Eigen::Vector2d{ -0.6, 0.5 } } )
    {
        auto g = tax::la::gradient( f, pt );
        EXPECT_NEAR( g( 0 ), pt( 1 ) + 2.0, 1e-10 );
        EXPECT_NEAR( g( 1 ), pt( 0 ), 1e-10 );

        auto H = tax::la::hessian( f, pt );
        EXPECT_NEAR( H( 0, 0 ), 0.0, 1e-10 );
        EXPECT_NEAR( H( 0, 1 ), 1.0, 1e-10 );
        EXPECT_NEAR( H( 1, 0 ), 1.0, 1e-10 );
        EXPECT_NEAR( H( 1, 1 ), 0.0, 1e-10 );
    }
}

template < typename Series2 >
static void checkJacobian()
{
    auto x = Series2::template variable< 0 >();
    auto y = Series2::template variable< 1 >();
    Eigen::Matrix< Series2, 2, 1 > F;
    F( 0 ) = x * y;  // ∂ = (y, x)
    F( 1 ) = x + y;  // ∂ = (1, 1)

    Eigen::Vector2d pt{ 0.4, -0.7 };
    auto J = tax::la::jacobian( F, pt );
    EXPECT_NEAR( J( 0, 0 ), pt( 1 ), 1e-10 );
    EXPECT_NEAR( J( 0, 1 ), pt( 0 ), 1e-10 );
    EXPECT_NEAR( J( 1, 0 ), 1.0, 1e-10 );
    EXPECT_NEAR( J( 1, 1 ), 1.0, 1e-10 );
}

TEST( BasisGenericLa, GradientHessianChebyshev )
{
    checkGradientHessian< tax::ChebyshevSeries< 5, 2 > >();
}
TEST( BasisGenericLa, GradientHessianLegendre )
{
    checkGradientHessian< tax::LegendreSeries< 5, 2 > >();
}
TEST( BasisGenericLa, GradientHessianHermite )
{
    checkGradientHessian< tax::HermiteSeries< 5, 2 > >();
}

TEST( BasisGenericLa, JacobianChebyshev ) { checkJacobian< tax::ChebyshevSeries< 5, 2 > >(); }
TEST( BasisGenericLa, JacobianLegendre ) { checkJacobian< tax::LegendreSeries< 5, 2 > >(); }
TEST( BasisGenericLa, JacobianHermite ) { checkJacobian< tax::HermiteSeries< 5, 2 > >(); }

// Matches the legacy Taylor gradient (point = displacement 0 from x0).
TEST( BasisGenericLa, PointFormAgreesWithTaylorAtCenter )
{
    std::array< double, 2 > x0{ 0.0, 0.0 };
    using TE = tax::TEn< 4, 2 >;
    auto x = TE::variable< 0 >( x0 );
    auto y = TE::variable< 1 >( x0 );
    auto f = x * y + 2.0 * x;
    auto gLegacy = tax::la::gradient( f );  // expansion-point form
    auto gPoint = tax::la::gradient( f, Eigen::Vector2d{ 0.0, 0.0 } );
    EXPECT_NEAR( gLegacy( 0 ), gPoint( 0 ), 1e-12 );
    EXPECT_NEAR( gLegacy( 1 ), gPoint( 1 ), 1e-12 );
}

// NumTraits: an Eigen matrix of non-Taylor expansions behaves as a scalar field.
template < typename Series2 >
static void checkNumTraitsMatmul()
{
    auto x = Series2::template variable< 0 >();
    auto y = Series2::template variable< 1 >();

    Eigen::Matrix< Series2, 2, 2 > A;
    A( 0, 0 ) = 1.0 + x;
    A( 0, 1 ) = y;
    A( 1, 0 ) = x;
    A( 1, 1 ) = 1.0 + y;
    Eigen::Matrix< Series2, 2, 1 > v;
    v( 0 ) = x;
    v( 1 ) = y;

    Eigen::Matrix< Series2, 2, 1 > w = A * v;  // exercises operator*/+ via NumTraits

    Eigen::Vector2d pt{ 0.3, -0.5 };
    const double xv = pt( 0 ), yv = pt( 1 );
    EXPECT_NEAR( w( 0 ).eval( { xv, yv } ), ( 1.0 + xv ) * xv + yv * yv, 1e-10 );
    EXPECT_NEAR( w( 1 ).eval( { xv, yv } ), xv * xv + ( 1.0 + yv ) * yv, 1e-10 );
}

TEST( BasisGenericLa, NumTraitsMatmulChebyshev )
{
    checkNumTraitsMatmul< tax::ChebyshevSeries< 4, 2 > >();
}
TEST( BasisGenericLa, NumTraitsMatmulLegendre )
{
    checkNumTraitsMatmul< tax::LegendreSeries< 4, 2 > >();
}
TEST( BasisGenericLa, NumTraitsMatmulHermite )
{
    checkNumTraitsMatmul< tax::HermiteSeries< 4, 2 > >();
}

// value() extracts the constant (P_0) coefficient for any basis: scalar form
// (tax::value) and the Eigen matrix form (tax::la::value).
TEST( BasisGenericLa, ValueAccessor )
{
    tax::LegendreSeries< 3 > f{ std::array< double, 4 >{ 1.5, 2.0, -1.0, 0.0 } };
    EXPECT_DOUBLE_EQ( tax::value( f ), 1.5 );

    Eigen::Matrix< tax::HermiteSeries< 3, 2 >, 2, 1 > F;
    F( 0 ) = 3.0 + tax::HermiteSeries< 3, 2 >::variable< 0 >();
    F( 1 ) = -1.0 + tax::HermiteSeries< 3, 2 >::variable< 1 >();
    auto v = tax::la::value( F );  // constant terms, element-wise
    EXPECT_DOUBLE_EQ( v( 0 ), 3.0 );
    EXPECT_DOUBLE_EQ( v( 1 ), -1.0 );
}
