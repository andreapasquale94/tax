// Oracle tests: BoxTE math surface matches the isotropic superset.
//
// For every in-box coefficient k of a BoxTE result, the value must equal the
// same coefficient of a tax::TE<Σorder, vars> computing the SAME expression,
// accessed via tax::flatIndex<vars>(ME::scheme::multiOf(k)).
//
// Shape: ME = BoxTE<Group<1,4>, Group<1,3>> (vars=2, Σ=7, box=20 coefficients).
//        ISO = TE<7, 2> (order-7 isotropic superset, 36 coefficients).

#include <gtest/gtest.h>

#include <cmath>
#include <tax/tax.hpp>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using ME = tax::BoxTE< tax::Group< 1, 4 >, tax::Group< 1, 3 > >;
using ISO = tax::TE< 7, 2 >;

static_assert( ME::nCoefficients == 20 );
static_assert( ISO::nCoefficients == 36 );

// ---------------------------------------------------------------------------
// Oracle helper: compare every box coefficient of `me` against `iso`.
// ---------------------------------------------------------------------------
static void checkOracle( const ME& me, const ISO& iso, const char* label )
{
    for ( std::size_t k = 0; k < ME::nCoefficients; ++k )
    {
        const auto alpha = ME::scheme::multiOf( k );
        const double iso_v = iso[tax::flatIndex< 2 >( alpha )];
        EXPECT_NEAR( me[k], iso_v, 1e-12 ) << label << ": mismatch at flat=" << k << " alpha=("
                                           << alpha[0] << "," << alpha[1] << ")";
    }
}

// ---------------------------------------------------------------------------
// Seed helpers
// ---------------------------------------------------------------------------
// ME variables: x = x0 + dx0, y = y0 + dx1 (box-filtered).
// ISO variables: same expansion point, full isotropic order-7.

static constexpr double kX0 = 0.7;
static constexpr double kY0 = 1.3;

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
// Arithmetic
// ---------------------------------------------------------------------------

TEST( BoxTE, Multiply ) { checkOracle( makeMeX() * makeMeY(), makeIsoX() * makeIsoY(), "x*y" ); }

TEST( BoxTE, Add ) { checkOracle( makeMeX() + makeMeY(), makeIsoX() + makeIsoY(), "x+y" ); }

TEST( BoxTE, Subtract ) { checkOracle( makeMeX() - makeMeY(), makeIsoX() - makeIsoY(), "x-y" ); }

TEST( BoxTE, Divide )
{
    // x / (y + 2.0): denominator constant = kY0 + 2.0 > 0
    auto me_result = makeMeX() / ( makeMeY() + 2.0 );
    auto iso_result = makeIsoX() / ( makeIsoY() + 2.0 );
    checkOracle( me_result, iso_result, "x/(y+2)" );
}

// ---------------------------------------------------------------------------
// Math functions — transcendental
// ---------------------------------------------------------------------------

TEST( BoxTE, Exp )
{
    // exp(x): x0 = 0.7, well in domain
    checkOracle( tax::exp( makeMeX() ), tax::exp( makeIsoX() ), "exp(x)" );
}

TEST( BoxTE, Log )
{
    // log(x): x0 = 0.7 > 0
    checkOracle( tax::log( makeMeX() ), tax::log( makeIsoX() ), "log(x)" );
}

TEST( BoxTE, Sqrt )
{
    // sqrt(x): x0 = 0.7 > 0
    checkOracle( tax::sqrt( makeMeX() ), tax::sqrt( makeIsoX() ), "sqrt(x)" );
}

TEST( BoxTE, Cbrt )
{
    // cbrt(x): x0 = 0.7 (all reals ok, but keep > 0 for numerical comfort)
    checkOracle( tax::cbrt( makeMeX() ), tax::cbrt( makeIsoX() ), "cbrt(x)" );
}

// ---------------------------------------------------------------------------
// Trigonometric
// ---------------------------------------------------------------------------

TEST( BoxTE, Sin ) { checkOracle( tax::sin( makeMeX() ), tax::sin( makeIsoX() ), "sin(x)" ); }

TEST( BoxTE, Cos ) { checkOracle( tax::cos( makeMeX() ), tax::cos( makeIsoX() ), "cos(x)" ); }

TEST( BoxTE, Tan )
{
    // x0 = 0.7, away from π/2
    checkOracle( tax::tan( makeMeX() ), tax::tan( makeIsoX() ), "tan(x)" );
}

// ---------------------------------------------------------------------------
// Inverse trigonometric — domain restrictions; seed into (-1,1) for asin/acos
// ---------------------------------------------------------------------------

TEST( BoxTE, Asin )
{
    // Use 0.3 + dx0, so constant term = 0.3+kX0 is NOT good — we need base in (-1,1).
    // Build constant = 0.3 for both and add a small-dx variable around 0.
    // Since variable<0>(p) gives p[0] + dx0, set p[0] = 0.0 so constant = 0.
    // Then asin(x) around 0: |0| < 1.
    typename ME::Input pm{ 0.0, kY0 };
    typename ISO::Input pi{ 0.0, kY0 };
    ME mx = ME::variable< 0 >( pm );
    ISO ix = ISO::variable< 0 >( pi );
    checkOracle( tax::asin( mx ), tax::asin( ix ), "asin(x) around 0" );
}

TEST( BoxTE, Acos )
{
    // acos needs argument in (-1,1); seed around 0.
    typename ME::Input pm{ 0.0, kY0 };
    typename ISO::Input pi{ 0.0, kY0 };
    ME mx = ME::variable< 0 >( pm );
    ISO ix = ISO::variable< 0 >( pi );
    checkOracle( tax::acos( mx ), tax::acos( ix ), "acos(x) around 0" );
}

TEST( BoxTE, Atan ) { checkOracle( tax::atan( makeMeX() ), tax::atan( makeIsoX() ), "atan(x)" ); }

// ---------------------------------------------------------------------------
// Hyperbolic
// ---------------------------------------------------------------------------

TEST( BoxTE, Sinh ) { checkOracle( tax::sinh( makeMeX() ), tax::sinh( makeIsoX() ), "sinh(x)" ); }

TEST( BoxTE, Cosh ) { checkOracle( tax::cosh( makeMeX() ), tax::cosh( makeIsoX() ), "cosh(x)" ); }

TEST( BoxTE, Tanh ) { checkOracle( tax::tanh( makeMeX() ), tax::tanh( makeIsoX() ), "tanh(x)" ); }

TEST( BoxTE, Asinh )
{
    checkOracle( tax::asinh( makeMeX() ), tax::asinh( makeIsoX() ), "asinh(x)" );
}

TEST( BoxTE, Acosh )
{
    // acosh requires argument >= 1; seed at 1.5
    typename ME::Input pm{ 1.5, kY0 };
    typename ISO::Input pi{ 1.5, kY0 };
    ME mx = ME::variable< 0 >( pm );
    ISO ix = ISO::variable< 0 >( pi );
    checkOracle( tax::acosh( mx ), tax::acosh( ix ), "acosh(x) around 1.5" );
}

TEST( BoxTE, Atanh )
{
    // atanh requires argument in (-1,1); seed at 0 for safety
    typename ME::Input pm{ 0.0, kY0 };
    typename ISO::Input pi{ 0.0, kY0 };
    ME mx = ME::variable< 0 >( pm );
    ISO ix = ISO::variable< 0 >( pi );
    checkOracle( tax::atanh( mx ), tax::atanh( ix ), "atanh(x) around 0" );
}

// ---------------------------------------------------------------------------
// Erf
// ---------------------------------------------------------------------------

TEST( BoxTE, Erf ) { checkOracle( tax::erf( makeMeX() ), tax::erf( makeIsoX() ), "erf(x)" ); }

// ---------------------------------------------------------------------------
// Pow
// ---------------------------------------------------------------------------

TEST( BoxTE, PowIntExponent )
{
    // pow(x + 1.5, 3): integer exponent; base constant = kX0 + 1.5 > 0
    auto me_base = makeMeX() + 1.5;
    auto iso_base = makeIsoX() + 1.5;
    checkOracle( tax::pow( me_base, 3 ), tax::pow( iso_base, 3 ), "pow(x+1.5, 3)" );
}

TEST( BoxTE, PowRealExponent )
{
    // pow(x + 1.5, 2.5): real exponent; base > 0
    auto me_base = makeMeX() + 1.5;
    auto iso_base = makeIsoX() + 1.5;
    checkOracle( tax::pow( me_base, 2.5 ), tax::pow( iso_base, 2.5 ), "pow(x+1.5, 2.5)" );
}

// ---------------------------------------------------------------------------
// Atan2
// ---------------------------------------------------------------------------

TEST( BoxTE, Atan2 )
{
    // atan2(x + 1.0, y + 2.0): both constant terms > 0, well-defined
    auto me_a = makeMeX() + 1.0;
    auto me_b = makeMeY() + 2.0;
    auto iso_a = makeIsoX() + 1.0;
    auto iso_b = makeIsoY() + 2.0;
    checkOracle( tax::atan2( me_a, me_b ), tax::atan2( iso_a, iso_b ), "atan2(x+1, y+2)" );
}

// ---------------------------------------------------------------------------
// Free-function factories: tax::mixed::variable / tax::mixed::variables must
// match the static-member factories used by makeMeX()/makeMeY().
// ---------------------------------------------------------------------------

TEST( BoxTE, FreeFunctionVariable )
{
    typename ME::Input p{ kX0, kY0 };
    auto x = tax::mixed::variable< 0, tax::Group< 1, 4 >, tax::Group< 1, 3 > >( p );
    auto y = tax::mixed::variable< 1, tax::Group< 1, 4 >, tax::Group< 1, 3 > >( p );
    static_assert( std::is_same_v< decltype( x ), ME > );
    const ME mx = makeMeX();
    const ME my = makeMeY();
    for ( std::size_t k = 0; k < ME::nCoefficients; ++k )
    {
        EXPECT_EQ( x[k], mx[k] ) << "x mismatch at " << k;
        EXPECT_EQ( y[k], my[k] ) << "y mismatch at " << k;
    }
}

TEST( BoxTE, FreeFunctionVariables )
{
    typename ME::Input p{ kX0, kY0 };
    auto v = tax::mixed::variables< tax::Group< 1, 4 >, tax::Group< 1, 3 > >( p );
    static_assert( std::is_same_v< decltype( v ), std::array< ME, 2 > > );
    const ME mx = makeMeX();
    const ME my = makeMeY();
    for ( std::size_t k = 0; k < ME::nCoefficients; ++k )
    {
        EXPECT_EQ( v[0][k], mx[k] ) << "v[0] mismatch at " << k;
        EXPECT_EQ( v[1][k], my[k] ) << "v[1] mismatch at " << k;
    }
}

TEST( BoxTE, FreeFunctionVariablesConstexpr )
{
    constexpr typename ME::Input p{ kX0, kY0 };
    constexpr auto v = tax::mixed::variables< tax::Group< 1, 4 >, tax::Group< 1, 3 > >( p );
    static_assert( std::is_same_v< std::remove_const_t< decltype( v ) >, std::array< ME, 2 > > );
    static_assert( v[0].value() == kX0 );
    static_assert( v[1].value() == kY0 );
    SUCCEED();
}

// Multi-dimensional group (Group<2,2>): variable<I> flattens a dim>1 block.
// Pin absolute coefficients, independent of the static-member factory.
TEST( BoxTE, FreeFunctionVariableMultiDim )
{
    using M22 = tax::BoxTE< tax::Group< 2, 2 > >;
    std::array< double, 2 > p{ 0.5, 1.0 };
    auto v0 = tax::mixed::variable< 0, tax::Group< 2, 2 > >( p );
    auto v1 = tax::mixed::variable< 1, tax::Group< 2, 2 > >( p );
    static_assert( std::is_same_v< decltype( v0 ), M22 > );
    EXPECT_EQ( v0.value(), 0.5 );
    EXPECT_EQ( v0.coeff( tax::MultiIndex< 2 >{ 1, 0 } ), 1.0 );
    EXPECT_EQ( v0.coeff( tax::MultiIndex< 2 >{ 0, 1 } ), 0.0 );
    EXPECT_EQ( v1.value(), 1.0 );
    EXPECT_EQ( v1.coeff( tax::MultiIndex< 2 >{ 1, 0 } ), 0.0 );
    EXPECT_EQ( v1.coeff( tax::MultiIndex< 2 >{ 0, 1 } ), 1.0 );
}

// Three groups: exercises the N-ary group iteration; pin coord 0 and coord 2.
TEST( BoxTE, FreeFunctionVariable3Groups )
{
    using ME3 = tax::BoxTE< tax::Group< 1, 2 >, tax::Group< 1, 3 >, tax::Group< 1, 1 > >;
    typename ME3::Input p{ 0.1, 0.2, 0.3 };
    auto x =
        tax::mixed::variable< 0, tax::Group< 1, 2 >, tax::Group< 1, 3 >, tax::Group< 1, 1 > >( p );
    auto z =
        tax::mixed::variable< 2, tax::Group< 1, 2 >, tax::Group< 1, 3 >, tax::Group< 1, 1 > >( p );
    static_assert( std::is_same_v< decltype( x ), ME3 > );
    EXPECT_EQ( x.value(), 0.1 );
    EXPECT_EQ( x.coeff( tax::MultiIndex< 3 >{ 1, 0, 0 } ), 1.0 );
    EXPECT_EQ( x.coeff( tax::MultiIndex< 3 >{ 0, 0, 1 } ), 0.0 );
    EXPECT_EQ( z.value(), 0.3 );
    EXPECT_EQ( z.coeff( tax::MultiIndex< 3 >{ 0, 0, 1 } ), 1.0 );
    EXPECT_EQ( z.coeff( tax::MultiIndex< 3 >{ 1, 0, 0 } ), 0.0 );
}

// Single-group mixed scheme: the plural factory returns a 1-element array.
TEST( BoxTE, FreeFunctionVariablesSingleGroup )
{
    using ME1 = tax::BoxTE< tax::Group< 1, 4 > >;
    typename ME1::Input p{ 0.7 };
    auto v = tax::mixed::variables< tax::Group< 1, 4 > >( p );
    static_assert( std::is_same_v< decltype( v ), std::array< ME1, 1 > > );
    EXPECT_EQ( v[0].value(), 0.7 );
    EXPECT_EQ( v[0].coeff( tax::MultiIndex< 1 >{ 1 } ), 1.0 );
}

// Non-default scalar deduces through the mixed factory.
TEST( BoxTE, FreeFunctionVariableFloat )
{
    std::array< float, 2 > p{ 0.7F, 1.3F };
    auto x = tax::mixed::variable< 0, tax::Group< 1, 4 >, tax::Group< 1, 3 > >( p );
    static_assert(
        std::is_same_v< decltype( x ),
                        tax::TaylorExpansion<
                            float, tax::MixedScheme< tax::Group< 1, 4 >, tax::Group< 1, 3 > > > > );
    EXPECT_FLOAT_EQ( x.value(), 0.7F );
}
