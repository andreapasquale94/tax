#include <gtest/gtest.h>

#include <tax/la.hpp>

#include "../testUtils.hpp"

TEST( EigenEval, ScalarTE )
{
    auto x = tax::TE< 3 >::variable( 1.0 );
    auto f = x * x;  // (1 + dx)^2 = 1 + 2*dx + dx^2
    tax::la::VecNT< 1, double > dx;
    dx << 0.1;
    double v = tax::la::eval( f, dx );
    EXPECT_NEAR( v, 1.21, 1e-12 );
}

TEST( EigenEval, ScalarMemberEval )
{
    auto x = tax::TE< 3 >::variable( 2.0 );
    auto f = x * x;  // (2 + dx)^2
    typename tax::TE< 3 >::Input p{ 0.5 };
    // f = 4 + 4*dx + dx^2 => at dx=0.5 => 4 + 2 + 0.25 = 6.25
    EXPECT_NEAR( f.eval( p ), 6.25, 1e-12 );
}

TEST( EigenEval, VectorOfTE )
{
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< tax::TE< 3, 2 > >( x0 );
    tax::la::VecNT< 2, tax::TE< 3, 2 > > F;
    F( 0 ) = v( 0 ) * v( 1 );
    F( 1 ) = v( 0 ) + v( 1 );
    Eigen::Vector2d dx{ 0.1, -0.1 };
    auto out = tax::la::eval( F, dx );
    EXPECT_NEAR( out( 0 ), 1.1 * 1.9, 1e-12 );
    EXPECT_NEAR( out( 1 ), 1.1 + 1.9, 1e-12 );
}

TEST( EigenValue, ElementWise )
{
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< tax::TE< 3, 2 > >( x0 );
    tax::la::VecNT< 2, tax::TE< 3, 2 > > F;
    F( 0 ) = v( 0 ) * v( 1 );
    F( 1 ) = v( 0 ) + v( 1 );
    auto val = tax::la::value( F );
    EXPECT_NEAR( val( 0 ), 2.0, 1e-12 );
    EXPECT_NEAR( val( 1 ), 3.0, 1e-12 );
}

// NumTraits<TaylorExpansion> must expose the Real-typed value functions
// (epsilon/dummy_precision/highest/lowest), returning Real rather than double.
TEST( EigenNumTraits, RealReturningValueFunctions )
{
    using TE = tax::TE< 3, 2 >;
    using NT = Eigen::NumTraits< TE >;
    static_assert( std::is_same_v< NT::Real, TE > );
    static_assert( std::is_same_v< decltype( NT::epsilon() ), TE > );
    static_assert( std::is_same_v< decltype( NT::dummy_precision() ), TE > );
    static_assert( std::is_same_v< decltype( NT::highest() ), TE > );
    EXPECT_DOUBLE_EQ( NT::epsilon().value(), Eigen::NumTraits< double >::epsilon() );
}
