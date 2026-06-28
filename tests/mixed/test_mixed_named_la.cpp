// tests/mixed/test_mixed_named_la.cpp
//
// Eigen integration for tax::mixed::MixedTaylorExpansion:
//   NumTraits (usable as Eigen scalar) and per-axis gradient/hessian/jacobian.
//
// Mirrors tests/eigen/test_named_la.cpp for NamedTaylorExpansion.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <tax/tax.hpp>

// ---------------------------------------------------------------------------
// Type aliases (via tax:: re-exports)
// ---------------------------------------------------------------------------

// Two axes: "p" dim 1 order 3, "x" dim 2 order 4 (canonical: p < x).
using PAx = tax::OrderedAxis< "p", 1, 3 >;
using XAx = tax::OrderedAxis< "x", 2, 4 >;
using MEpx = tax::MixedTaylorExpansion< double, PAx, XAx >;

// Canonical union sorted by name → p (var 0), x (var 1, var 2).
// Total vars_v = 3.
static_assert( MEpx::vars_v == 3 );

// ---------------------------------------------------------------------------
// TEST: NumTraits lets MixedTaylorExpansion act as an Eigen scalar
// ---------------------------------------------------------------------------

TEST( MixedNamedLa, NumTraitsUsableAsEigenScalar )
{
    auto x = tax::mixed::variables< "x", 4, 2 >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = tax::mixed::variable< "p", 3 >( 0.0 );

    // Build an Eigen vector whose scalar is a MEpx expansion.
    // Both x[i]*p and x[i] individually have the full {p,x} union type,
    // so they can be stored directly in MEpx slots.
    auto f0 = x[0] * p;                  // type: BoxTE<double, PAx, XAx>
    auto f1 = x[1] * p + p * 0.0 + 1.0;  // embed x[1] into {p,x} via p*0

    Eigen::Matrix< MEpx, 2, 1 > v;
    v( 0 ) = f0;
    v( 1 ) = f1;

    // Element-wise Eigen arithmetic — relies on NumTraits<MEpx>.
    Eigen::Matrix< MEpx, 2, 1 > w = v + v;
    EXPECT_DOUBLE_EQ( w( 1 ).value(), 2.0 );
}

// ---------------------------------------------------------------------------
// TEST: NumTraits::epsilon() / dummy_precision() return Self, not T
// ---------------------------------------------------------------------------

TEST( MixedNamedLa, NumTraitsRealReturningValueFunctions )
{
    using NT = Eigen::NumTraits< MEpx >;
    static_assert( std::is_same_v< decltype( NT::epsilon() ), MEpx > );
    static_assert( std::is_same_v< decltype( NT::dummy_precision() ), MEpx > );
    EXPECT_DOUBLE_EQ( NT::epsilon().value(), Eigen::NumTraits< double >::epsilon() );
}

// ---------------------------------------------------------------------------
// TEST: gradient<"x">(f) and gradient<"p">(f) match analytic values
//
// f = x0^2 * p0 + 2*x1  at x=(3,5), p=(2).
// df/dx0 = 2*x0*p0 = 12,  df/dx1 = 2,  df/dp0 = x0^2 = 9.
// ---------------------------------------------------------------------------

TEST( MixedNamedLa, GradientByAxis )
{
    auto x = tax::mixed::variables< "x", 4, 2 >( std::array< double, 2 >{ 3.0, 5.0 } );
    auto p = tax::mixed::variable< "p", 3 >( 2.0 );

    // f = x[0]*x[0]*p + 2*x[1]; all terms compose to {p,x} union type
    auto f = x[0] * x[0] * p + 2.0 * x[1];

    // gradient w.r.t. "x" → [df/dx0, df/dx1] = [12, 2]
    auto gx = tax::mixed::gradient< "x" >( f );
    ASSERT_EQ( gx.rows(), 2 );
    EXPECT_DOUBLE_EQ( gx( 0 ), 12.0 );
    EXPECT_DOUBLE_EQ( gx( 1 ), 2.0 );

    // gradient w.r.t. "p" → [df/dp0] = [9]
    auto gp = tax::mixed::gradient< "p" >( f );
    ASSERT_EQ( gp.rows(), 1 );
    EXPECT_DOUBLE_EQ( gp( 0 ), 9.0 );
}

// ---------------------------------------------------------------------------
// TEST: hessian<"x">(f) matches analytic Hessian over the "x" variables
//
// f = x0^2 * p0 ; H_x = [[2*p0, 0], [0, 0]] = [[4, 0], [0, 0]].
// ---------------------------------------------------------------------------

TEST( MixedNamedLa, HessianByAxis )
{
    auto x = tax::mixed::variables< "x", 4, 2 >( std::array< double, 2 >{ 3.0, 5.0 } );
    auto p = tax::mixed::variable< "p", 3 >( 2.0 );
    auto f = x[0] * x[0] * p;

    auto H = tax::mixed::hessian< "x" >( f );
    ASSERT_EQ( H.rows(), 2 );
    ASSERT_EQ( H.cols(), 2 );
    EXPECT_DOUBLE_EQ( H( 0, 0 ), 4.0 );
    EXPECT_DOUBLE_EQ( H( 0, 1 ), 0.0 );
    EXPECT_DOUBLE_EQ( H( 1, 0 ), 0.0 );
    EXPECT_DOUBLE_EQ( H( 1, 1 ), 0.0 );
}

// ---------------------------------------------------------------------------
// TEST: jacobian<"x">(F) matches analytic Jacobian
//
// F = [x0*p0, x1^2]  at x=(0,5), p=(2).
// J_x = [[p0, 0], [0, 2*x1]] = [[2, 0], [0, 10]].
// ---------------------------------------------------------------------------

TEST( MixedNamedLa, JacobianByAxis )
{
    auto x = tax::mixed::variables< "x", 4, 2 >( std::array< double, 2 >{ 0.0, 5.0 } );
    auto p = tax::mixed::variable< "p", 3 >( 2.0 );

    // F(0) = x[0]*p0, F(1) = x[1]^2 + 0*p (embed into {p,x} space via p*0)
    auto f0 = x[0] * p;
    auto f1 = x[1] * x[1] + p * 0.0;  // force union type {p,x}

    Eigen::Matrix< MEpx, 2, 1 > F;
    F( 0 ) = f0;
    F( 1 ) = f1;

    auto J = tax::mixed::jacobian< "x" >( F );
    ASSERT_EQ( J.rows(), 2 );
    ASSERT_EQ( J.cols(), 2 );
    EXPECT_DOUBLE_EQ( J( 0, 0 ), 2.0 );  // p0 = 2
    EXPECT_DOUBLE_EQ( J( 0, 1 ), 0.0 );
    EXPECT_DOUBLE_EQ( J( 1, 0 ), 0.0 );
    EXPECT_DOUBLE_EQ( J( 1, 1 ), 10.0 );  // 2*x1 = 10
}

// ---------------------------------------------------------------------------
// TEST: gradient<"x"> agrees with isotropic-superset oracle
//
// f = sin(x*t) + exp(x) where x is dim-1 axis "x" @ order 3 and t is
// dim-1 axis "t" @ order 4.  Canonical order: t (var 0), x (var 1).
// gradient<"x"> should match df/dx from TE<7,2>.
// ---------------------------------------------------------------------------

TEST( MixedNamedLa, GradientMatchesIsotropicOracle )
{
    auto x = tax::mixed::variable< "x", 3 >( 0.7 );
    auto t = tax::mixed::variable< "t", 4 >( 1.3 );
    // Use unqualified ADL: sin/exp for MixedTaylorExpansion are in tax::mixed.
    auto f = sin( x * t ) + exp( x );

    // gradient<"x"> should give df/dx at the expansion point.
    auto gx = tax::mixed::gradient< "x" >( f );
    ASSERT_EQ( gx.rows(), 1 );

    // Isotropic oracle: TE<7,2> with t=var0, x=var1.
    typename tax::TE< 7, 2 >::Input p{ 1.3, 0.7 };
    auto it = tax::TE< 7, 2 >::variable< 0 >( p );
    auto ix = tax::TE< 7, 2 >::variable< 1 >( p );
    auto iso = tax::sin( ix * it ) + tax::exp( ix );

    // df/dx = t*cos(x*t) + exp(x) at (0.7, 1.3)
    tax::MultiIndex< 2 > alpha_x{};
    alpha_x[1] = 1;  // d/dx (var 1 in isotropic ordering)
    EXPECT_NEAR( gx( 0 ), iso.derivative( alpha_x ), 1e-12 );
}

// ---------------------------------------------------------------------------
// TEST: VecNT<D, MEpx> works in Eigen linear algebra (dot product via NumTraits)
// ---------------------------------------------------------------------------

TEST( MixedNamedLa, VecNTDotProductViaNumTraits )
{
    auto x = tax::mixed::variables< "x", 4, 2 >( std::array< double, 2 >{ 1.0, 2.0 } );
    auto p = tax::mixed::variable< "p", 3 >( 0.0 );

    // Embed each x variable into the {p,x} space by adding p*0.
    auto f0 = x[0] + p * 0.0;
    auto f1 = x[1] + p * 0.0;

    tax::la::VecNT< 2, MEpx > v;
    v( 0 ) = f0;
    v( 1 ) = f1;

    // dot(v, v) = x[0]^2 + x[1]^2
    MEpx d = v.dot( v );

    // Value = 1^2 + 2^2 = 5
    EXPECT_NEAR( d.value(), 5.0, 1e-12 );

    // gradient w.r.t. "x" = [2*x[0], 2*x[1]] = [2, 4]
    auto g = tax::mixed::gradient< "x" >( d );
    ASSERT_EQ( g.rows(), 2 );
    EXPECT_NEAR( g( 0 ), 2.0, 1e-12 );
    EXPECT_NEAR( g( 1 ), 4.0, 1e-12 );
}

// ---------------------------------------------------------------------------
// TEST: the public `tax::` spelling resolves for MixedTaylorExpansion.
// `tax::gradient<"name">`/`jacobian<"name">` must find the mixed overloads via
// the re-export in la/mixed_named.hpp (the la/named.hpp using-block predates
// them and would not pick them up).
// ---------------------------------------------------------------------------

TEST( MixedNamedLa, PublicTaxSpellingResolves )
{
    auto x = tax::mixed::variables< "x", 4, 2 >( std::array< double, 2 >{ 3.0, 5.0 } );
    auto p = tax::mixed::variable< "p", 3 >( 2.0 );
    auto f = x[0] * x[0] * p + 2.0 * x[1];  // df/dx0=12, df/dx1=2

    // Qualified public spelling (NOT tax::named::) — exercises the re-export.
    auto gx = tax::gradient< "x" >( f );
    ASSERT_EQ( gx.rows(), 2 );
    EXPECT_DOUBLE_EQ( gx( 0 ), 12.0 );
    EXPECT_DOUBLE_EQ( gx( 1 ), 2.0 );

    tax::la::VecNT< 1, decltype( f ) > F;
    F( 0 ) = f;
    auto J = tax::jacobian< "x" >( F );  // 1x2 Jacobian = [12, 2]
    ASSERT_EQ( J.rows(), 1 );
    ASSERT_EQ( J.cols(), 2 );
    EXPECT_DOUBLE_EQ( J( 0, 0 ), 12.0 );
    EXPECT_DOUBLE_EQ( J( 0, 1 ), 2.0 );
}
