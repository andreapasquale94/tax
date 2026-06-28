// tests/eigen/test_tax_spelling.cpp
//
// Resolution test: the documented public spelling is tax::FN — it must resolve
// for plain (dense) expansions AND named expansions uniformly.
// Previously tax::gradient(te) did not resolve; only tax::gradient<"x">(ne) did.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <tax/tax.hpp>

TEST( TaxSpelling, GradientResolvesForDenseAndNamed )
{
    // Dense: f(x,y) = x*y at (0,0) on TE<3,2>; grad = [y, x] = [0,0] here, but
    // the point is that tax::gradient(te) RESOLVES (the LA-5 gap).
    typename tax::TE< 3, 2 >::Input p{ 0.0, 0.0 };
    auto fx = tax::TE< 3, 2 >::variable< 0 >( p );
    auto fy = tax::TE< 3, 2 >::variable< 1 >( p );
    auto g  = tax::gradient( fx * fy );  // <-- previously unresolved at tax::
    EXPECT_EQ( g.size(), 2 );

    // Named: ∂(x*p)/∂x via the axis-addressed form, also under tax::.
    auto nx = tax::variable< "x", 1 >( 0.0 );
    auto np = tax::variable< "p", 1 >( 0.0 );
    auto gn = tax::gradient< "x" >( nx * np, Eigen::Vector2d{ 0.3, -0.4 } );
    EXPECT_EQ( gn.size(), 1 );

    // tax::value / tax::invert also resolve at tax::.
    EXPECT_DOUBLE_EQ( tax::value( fx ), 0.0 );
    Eigen::Matrix< tax::TE< 3, 2 >, 2, 1 > F;
    F( 0 ) = fx;
    F( 1 ) = fy;
    auto Finv = tax::invert( F );  // <-- tax::invert resolves
    EXPECT_EQ( Finv.size(), 2 );
}
