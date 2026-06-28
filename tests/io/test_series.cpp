#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <tax/tax.hpp>

namespace
{

template < typename F >
std::string str( const F& f )
{
    std::ostringstream os;
    os << f;
    return os.str();
}

template < typename F >
std::string str( const F& f, tax::SeriesOptions opts )
{
    std::ostringstream os;
    os << tax::series( f, opts );
    return os.str();
}

}  // namespace

// ---------------------------------------------------------------------------
// Polynomial style: subscripts, superscripts, implicit multiplication
// ---------------------------------------------------------------------------

TEST( SeriesPolynomial, UnivariateExp )
{
    // exp(x) at 0, order 3 : 1 + x + 0.5 x^2 + 1/6 x^3
    auto x = tax::TE< 3 >::variable( 0.0 );
    auto f = tax::exp( x );
    EXPECT_EQ( str( f ), "1 + x₀ + 0.5x₀² + 0.166667x₀³" );
}

TEST( SeriesPolynomial, MultivariateImplicitMult )
{
    // f = 2 + 3 x0 + 4 x0 x1 - 0.5 x1^2
    auto x = tax::TE< 2, 2 >::variable< 0 >( { 0.0, 0.0 } );
    auto y = tax::TE< 2, 2 >::variable< 1 >( { 0.0, 0.0 } );
    auto f = 2.0 + 3.0 * x + 4.0 * x * y - 0.5 * y * y;
    EXPECT_EQ( str( f ), "2 + 3x₀ + 4x₀x₁ - 0.5x₁²" );
}

TEST( SeriesPolynomial, CoefficientOneElided )
{
    // x itself : "x0" (no "1*"), and -x.
    auto x = tax::TE< 1 >::variable( 0.0 );
    EXPECT_EQ( str( x ), "x₀" );
    EXPECT_EQ( str( -x ), "-x₀" );
}

TEST( SeriesPolynomial, ZeroPolynomial )
{
    tax::TE< 3 > z{};
    EXPECT_EQ( str( z ), "0" );
}

TEST( SeriesPolynomial, MultiDigitSubAndSuperscript )
{
    // 11 variables -> x10 has a two-digit subscript; check x10^23-style rendering
    // via a high-order univariate power instead (order 23).
    auto x = tax::TE< 23 >::variable( 0.0 );
    auto f = tax::pow( x, 23 );  // x^23
    EXPECT_NE( str( f ).find( "x₀²³" ), std::string::npos );
}

// ---------------------------------------------------------------------------
// Named: axis names, bare name for 1-D, subscripts for multi-D
// ---------------------------------------------------------------------------

TEST( SeriesNamed, OneDimAxisUsesBareName )
{
    auto x = tax::variable< "x", 3 >( 0.0 );
    auto f = exp( x );
    EXPECT_EQ( str( f ), "1 + x + 0.5x² + 0.166667x³" );
}

// ---------------------------------------------------------------------------
// Options: threshold dropping
// ---------------------------------------------------------------------------

TEST( SeriesOptions, ThresholdDropsSmallTerms )
{
    auto x = tax::TE< 3 >::variable( 0.0 );
    auto f = tax::exp( x );  // 1/6 ~ 0.1667 term should drop at threshold 0.2
    auto s = str( f, { .threshold = 0.2 } );
    EXPECT_EQ( s, "1 + x₀ + 0.5x₀²" );
}

// ---------------------------------------------------------------------------
// to_string convenience
// ---------------------------------------------------------------------------

TEST( SeriesToString, ScalarDefaultAndOptions )
{
    auto x = tax::TE< 3 >::variable( 0.0 );
    auto f = tax::exp( x );
    EXPECT_EQ( tax::to_string( f ), "1 + x₀ + 0.5x₀² + 0.166667x₀³" );
    EXPECT_EQ( tax::to_string( f, { .threshold = 0.2 } ), "1 + x₀ + 0.5x₀²" );
}

TEST( SeriesToString, NamedAndVector )
{
    auto nx = tax::variable< "x", 3 >( 0.0 );
    EXPECT_EQ( tax::to_string( exp( nx ) ), "1 + x + 0.5x² + 0.166667x³" );

    tax::la::VecNT< 2, tax::TE< 2 > > v;
    v( 0 ) = tax::TE< 2 >::variable( 0.0 );
    v( 1 ) = tax::exp( tax::TE< 2 >::variable( 0.0 ) );
    const auto s = tax::to_string( v );
    EXPECT_NE( s.find( "[0]" ), std::string::npos );
    EXPECT_NE( s.find( "[1]" ), std::string::npos );
}

// ---------------------------------------------------------------------------
// Tabular style (ASCII)
// ---------------------------------------------------------------------------

TEST( SeriesTabular, UnivariateHasHeaderAndRows )
{
    auto x = tax::TE< 2 >::variable( 0.0 );
    auto f = tax::exp( x );
    auto s = str( f, { .style = tax::SeriesStyle::Tabular } );
    EXPECT_NE( s.find( "COEFFICIENT" ), std::string::npos );
    EXPECT_NE( s.find( "ORDER" ), std::string::npos );
    EXPECT_NE( s.find( "EXPONENTS" ), std::string::npos );
    // Three nonzero rows for order-2 exp: degrees 0,1,2.
    EXPECT_NE( s.find( "1.000000000000000e+00" ), std::string::npos );
    EXPECT_NE( s.find( "5.000000000000000e-01" ), std::string::npos );
}

// ---------------------------------------------------------------------------
// Sparse parity with dense
// ---------------------------------------------------------------------------

TEST( SeriesSparse, MatchesDensePolynomial )
{
    auto xd = tax::TE< 3 >::variable( 0.0 );
    auto fd = tax::exp( xd );
    auto fs = tax::sparse( fd );
    EXPECT_EQ( str( fs ), str( fd ) );
}

// ---------------------------------------------------------------------------
// Eigen vector of expansions
// ---------------------------------------------------------------------------

TEST( SeriesVector, LabeledRows )
{
    using TE = tax::TE< 2 >;
    tax::la::VecNT< 2, TE > v;
    v( 0 ) = TE::variable( 0.0 );  // x0
    v( 1 ) = tax::exp( TE::variable( 0.0 ) );
    auto s = str( v, tax::SeriesOptions{} );
    EXPECT_NE( s.find( "[0]" ), std::string::npos );
    EXPECT_NE( s.find( "[1]" ), std::string::npos );
}

// ---------------------------------------------------------------------------
// to_string is constrained: only printable expansions (any basis) + Eigen
// matrices of them resolve; arbitrary types are a clean constraint error.
// ---------------------------------------------------------------------------

namespace
{
template < typename F, typename = void >
struct HasToString : std::false_type
{
};
template < typename F >
struct HasToString< F, std::void_t< decltype( tax::to_string( std::declval< const F& >() ) ) > >
    : std::true_type
{
};
}  // namespace

static_assert( HasToString< tax::TE< 3 > >::value, "Taylor expansion must print" );
static_assert( HasToString< tax::ChebyshevSeries< 2 > >::value, "orthogonal expansion must print" );
static_assert( HasToString< tax::NE< 3, tax::Axis< "x", 1 > > >::value,
               "named expansion must print" );
static_assert( !HasToString< int >::value, "scalars are not printable expansions" );
static_assert( !HasToString< std::string >::value, "std::string is not a printable expansion" );
static_assert( !HasToString< tax::MTE< tax::OrderedAxis< "x", 1, 3 > > >::value,
               "mixed-expansion printing is out of scope" );
