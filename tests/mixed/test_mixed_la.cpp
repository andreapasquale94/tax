// tax::la on MixedTE — verify gradient / hessian / jacobian / value / variables
// work on an anisotropic MixedTE and produce results matching the isotropic
// TE<7,2> superset or analytic reference values.
//
// ME  = MixedTE<Group<1,4>, Group<1,3>> : vars=2, Σorder=7, nCoeff=20
// ISO = TE<7,2>                          : vars=2, order=7,  nCoeff=36
//
// All first- and second-order monomials are inside the ME box (group-0 order=4,
// group-1 order=3 — each single variable has total group-degree ≤ 1), so
// gradient and hessian values must agree between ME and ISO exactly (1e-12).

#include <gtest/gtest.h>

#include <cmath>
#include <tax/tax.hpp>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using ME = tax::MixedTE< tax::Group< 1, 4 >, tax::Group< 1, 3 > >;
using ISO = tax::TE< 7, 2 >;

static_assert( ME::nCoefficients == 20 );
static_assert( ISO::nCoefficients == 36 );

// Both schemes have 2 variables.
static_assert( ME::scheme::vars == 2 );
static_assert( ISO::scheme::vars == 2 );

// ---------------------------------------------------------------------------
// Expansion point
// ---------------------------------------------------------------------------
static constexpr double kX0 = 0.7;
static constexpr double kY0 = 1.3;

// Build ME variables at (kX0, kY0).
static ME makeMeX()
{
    typename ME::Input p{ kX0, kY0 };
    return ME::variable< 0 >( p );
}
static ME makeMeY()
{
    typename ME::Input p{ kX0, kY0 };
    return ME::variable< 1 >( p );
}

// Build ISO variables at the same expansion point.
static ISO makeIsoX()
{
    typename ISO::Input p{ kX0, kY0 };
    return ISO::variable< 0 >( p );
}
static ISO makeIsoY()
{
    typename ISO::Input p{ kX0, kY0 };
    return ISO::variable< 1 >( p );
}

// ---------------------------------------------------------------------------
// Helper: compare a 2-vector against analytic values.
// ---------------------------------------------------------------------------
static void expectVec2( const Eigen::Matrix< double, 2, 1 >& v, double v0, double v1,
                        const char* label )
{
    EXPECT_NEAR( v( 0 ), v0, 1e-12 ) << label << " component 0";
    EXPECT_NEAR( v( 1 ), v1, 1e-12 ) << label << " component 1";
}

// ---------------------------------------------------------------------------
// Test 1: member gradient() agrees with ISO and with analytic values
//
// f = sin(x*y) + exp(x)
// df/dx = y*cos(x*y) + exp(x)   at (0.7, 1.3)
// df/dy = x*cos(x*y)            at (0.7, 1.3)
// ---------------------------------------------------------------------------

TEST( MixedLA, MemberGradientMatchesIso )
{
    auto x_me = makeMeX();
    auto y_me = makeMeY();
    auto f_me = tax::sin( x_me * y_me ) + tax::exp( x_me );

    auto x_iso = makeIsoX();
    auto y_iso = makeIsoY();
    auto f_iso = tax::sin( x_iso * y_iso ) + tax::exp( x_iso );

    const auto g_me = tax::la::gradient( f_me );
    const auto g_iso = tax::la::gradient( f_iso );

    EXPECT_NEAR( g_me( 0 ), g_iso( 0 ), 1e-12 ) << "gradient x: ME vs ISO";
    EXPECT_NEAR( g_me( 1 ), g_iso( 1 ), 1e-12 ) << "gradient y: ME vs ISO";

    // Analytic reference
    const double xy = kX0 * kY0;
    const double df_dx = kY0 * std::cos( xy ) + std::exp( kX0 );
    const double df_dy = kX0 * std::cos( xy );
    expectVec2( g_me, df_dx, df_dy, "gradient analytic" );
}

// ---------------------------------------------------------------------------
// Test 2: free-function tax::la::gradient matches analytic values
//
// f = sin(x*y) + exp(x)
// df/dx = y*cos(x*y) + exp(x)   at (kX0, kY0)
// df/dy = x*cos(x*y)            at (kX0, kY0)
// ---------------------------------------------------------------------------

TEST( MixedLA, FreeFunctionGradientMatchesAnalytic )
{
    auto x = makeMeX();
    auto y = makeMeY();
    auto f = tax::sin( x * y ) + tax::exp( x );

    const auto g = tax::la::gradient( f );

    const double xy = kX0 * kY0;
    const double df_dx = kY0 * std::cos( xy ) + std::exp( kX0 );
    const double df_dy = kX0 * std::cos( xy );
    EXPECT_NEAR( g( 0 ), df_dx, 1e-10 ) << "free-function gradient x analytic";
    EXPECT_NEAR( g( 1 ), df_dy, 1e-10 ) << "free-function gradient y analytic";
}

// ---------------------------------------------------------------------------
// Test 3: member hessian() agrees with ISO
//
// H(0,0) = d²f/dx² = -y²*sin(x*y) + exp(x)
// H(0,1) = H(1,0) = d²f/dxdy = cos(x*y) - x*y*sin(x*y)
// H(1,1) = d²f/dy² = -x²*sin(x*y)
// ---------------------------------------------------------------------------

TEST( MixedLA, HessianMatchesIso )
{
    auto x_me = makeMeX();
    auto y_me = makeMeY();
    auto f_me = tax::sin( x_me * y_me ) + tax::exp( x_me );

    auto x_iso = makeIsoX();
    auto y_iso = makeIsoY();
    auto f_iso = tax::sin( x_iso * y_iso ) + tax::exp( x_iso );

    const auto H_me = tax::la::hessian( f_me );
    const auto H_iso = tax::la::hessian( f_iso );

    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 2; ++j )
            EXPECT_NEAR( H_me( i, j ), H_iso( i, j ), 1e-12 )
                << "hessian(" << i << "," << j << "): ME vs ISO";

    // Analytic reference
    const double xy = kX0 * kY0;
    const double sxy = std::sin( xy );
    const double cxy = std::cos( xy );
    const double ex = std::exp( kX0 );
    const double h00 = -kY0 * kY0 * sxy + ex;
    const double h01 = cxy - kX0 * kY0 * sxy;
    const double h11 = -kX0 * kX0 * sxy;
    EXPECT_NEAR( H_me( 0, 0 ), h00, 1e-12 ) << "H(0,0) analytic";
    EXPECT_NEAR( H_me( 0, 1 ), h01, 1e-12 ) << "H(0,1) analytic";
    EXPECT_NEAR( H_me( 1, 0 ), h01, 1e-12 ) << "H(1,0) analytic";
    EXPECT_NEAR( H_me( 1, 1 ), h11, 1e-12 ) << "H(1,1) analytic";
}

// ---------------------------------------------------------------------------
// Test 4: free-function tax::la::hessian matches analytic values
//
// H(0,0) = d²f/dx² = -y²*sin(x*y) + exp(x)
// H(0,1) = H(1,0) = d²f/dxdy = cos(x*y) - x*y*sin(x*y)
// H(1,1) = d²f/dy² = -x²*sin(x*y)
// ---------------------------------------------------------------------------

TEST( MixedLA, FreeFunctionHessianMatchesAnalytic )
{
    auto x = makeMeX();
    auto y = makeMeY();
    auto f = tax::sin( x * y ) + tax::exp( x );

    const auto H = tax::la::hessian( f );

    const double xy = kX0 * kY0;
    const double sxy = std::sin( xy );
    const double cxy = std::cos( xy );
    const double ex = std::exp( kX0 );
    const double h00 = -kY0 * kY0 * sxy + ex;
    const double h01 = cxy - kX0 * kY0 * sxy;
    const double h11 = -kX0 * kX0 * sxy;
    EXPECT_NEAR( H( 0, 0 ), h00, 1e-10 ) << "free-function H(0,0) analytic";
    EXPECT_NEAR( H( 0, 1 ), h01, 1e-10 ) << "free-function H(0,1) analytic";
    EXPECT_NEAR( H( 1, 0 ), h01, 1e-10 ) << "free-function H(1,0) analytic";
    EXPECT_NEAR( H( 1, 1 ), h11, 1e-10 ) << "free-function H(1,1) analytic";
}

// ---------------------------------------------------------------------------
// Test 5: tax::la::value on a VecNT<2, ME>
// ---------------------------------------------------------------------------

TEST( MixedLA, ValueOfVector )
{
    auto x = makeMeX();
    auto y = makeMeY();

    tax::la::VecNT< 2, ME > F;
    F( 0 ) = tax::sin( x * y ) + tax::exp( x );
    F( 1 ) = x - y;

    const auto vals = tax::la::value( F );

    // value() = constant term = f(x0, y0)
    const double xy = kX0 * kY0;
    EXPECT_NEAR( vals( 0 ), std::sin( xy ) + std::exp( kX0 ), 1e-12 ) << "value F(0)";
    EXPECT_NEAR( vals( 1 ), kX0 - kY0, 1e-12 ) << "value F(1)";
}

// ---------------------------------------------------------------------------
// Test 6: tax::la::jacobian on a VecNT<2, ME>
//
// F(0) = sin(x*y) + exp(x)
// F(1) = x - y
//
// J = [ df0/dx  df0/dy ] = [ y*cos(xy)+exp(x)    x*cos(xy) ]
//     [ df1/dx  df1/dy ]   [ 1                  -1         ]
// ---------------------------------------------------------------------------

TEST( MixedLA, JacobianOfVector )
{
    auto x = makeMeX();
    auto y = makeMeY();

    tax::la::VecNT< 2, ME > F;
    F( 0 ) = tax::sin( x * y ) + tax::exp( x );
    F( 1 ) = x - y;

    const auto J = tax::la::jacobian( F );

    const double xy = kX0 * kY0;
    EXPECT_NEAR( J( 0, 0 ), kY0 * std::cos( xy ) + std::exp( kX0 ), 1e-12 ) << "J(0,0)";
    EXPECT_NEAR( J( 0, 1 ), kX0 * std::cos( xy ), 1e-12 ) << "J(0,1)";
    EXPECT_NEAR( J( 1, 0 ), 1.0, 1e-12 ) << "J(1,0)";
    EXPECT_NEAR( J( 1, 1 ), -1.0, 1e-12 ) << "J(1,1)";
}

// ---------------------------------------------------------------------------
// Test 7: tax::la::jacobian matches ISO jacobian
// ---------------------------------------------------------------------------

TEST( MixedLA, JacobianMatchesIso )
{
    auto x_me = makeMeX();
    auto y_me = makeMeY();
    tax::la::VecNT< 2, ME > F_me;
    F_me( 0 ) = tax::sin( x_me * y_me ) + tax::exp( x_me );
    F_me( 1 ) = x_me - y_me;

    auto x_iso = makeIsoX();
    auto y_iso = makeIsoY();
    tax::la::VecNT< 2, ISO > F_iso;
    F_iso( 0 ) = tax::sin( x_iso * y_iso ) + tax::exp( x_iso );
    F_iso( 1 ) = x_iso - y_iso;

    const auto J_me = tax::la::jacobian( F_me );
    const auto J_iso = tax::la::jacobian( F_iso );

    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 2; ++j )
            EXPECT_NEAR( J_me( i, j ), J_iso( i, j ), 1e-12 )
                << "jacobian(" << i << "," << j << "): ME vs ISO";
}

// ---------------------------------------------------------------------------
// Test 8: tax::la::variables builds a VecNT<2,ME> of coordinate variables
// ---------------------------------------------------------------------------

TEST( MixedLA, VariablesBuilder )
{
    Eigen::Vector2d x0{ kX0, kY0 };
    const auto v = tax::la::variables< ME >( x0 );

    static_assert( std::is_same_v< std::decay_t< decltype( v ) >, tax::la::VecNT< 2, ME > > );

    // v(0) is the x-variable: value = kX0, gradient = (1, 0)
    EXPECT_NEAR( v( 0 ).value(), kX0, 1e-15 ) << "variables v(0) value";
    const auto gv0 = tax::la::gradient( v( 0 ) );
    EXPECT_NEAR( gv0( 0 ), 1.0, 1e-15 ) << "variables v(0) grad[0]";
    EXPECT_NEAR( gv0( 1 ), 0.0, 1e-15 ) << "variables v(0) grad[1]";

    // v(1) is the y-variable: value = kY0, gradient = (0, 1)
    EXPECT_NEAR( v( 1 ).value(), kY0, 1e-15 ) << "variables v(1) value";
    const auto gv1 = tax::la::gradient( v( 1 ) );
    EXPECT_NEAR( gv1( 0 ), 0.0, 1e-15 ) << "variables v(1) grad[0]";
    EXPECT_NEAR( gv1( 1 ), 1.0, 1e-15 ) << "variables v(1) grad[1]";
}

// ---------------------------------------------------------------------------
// Test 9: NumTraits — ME in an Eigen expression (dot product)
//
// Exercises Eigen::NumTraits<ME> by computing an Eigen dot product of
// VecNT<2,ME> with itself, which internally uses MulCost/AddCost and
// the ME arithmetic operators through Eigen's machinery.
// ---------------------------------------------------------------------------

TEST( MixedLA, NumTraitsEigenDot )
{
    auto x = makeMeX();
    auto y = makeMeY();

    tax::la::VecNT< 2, ME > v;
    v( 0 ) = x;
    v( 1 ) = y;

    // dot(v, v) = x² + y²  (as a ME expansion)
    const ME dot = v.dot( v );

    // Value at expansion point = x0² + y0²
    EXPECT_NEAR( dot.value(), kX0 * kX0 + kY0 * kY0, 1e-12 ) << "NumTraits dot value";

    // Gradient of x²+y² = (2x, 2y) at expansion point
    const auto g = tax::la::gradient( dot );
    EXPECT_NEAR( g( 0 ), 2.0 * kX0, 1e-12 ) << "NumTraits dot grad[0]";
    EXPECT_NEAR( g( 1 ), 2.0 * kY0, 1e-12 ) << "NumTraits dot grad[1]";
}
