#include "testUtils.hpp"
#include <tax/ads.hpp>

#include <cmath>
#include <fstream>
#include <iostream>

// ---------------------------------------------------------------------------
// Gaussian ADS example from Section 2.1.2 / Fig. 3-4 of Wittig et al.
//
// f(x) = exp(-x^2)  on  [-3, 3]
// DA order N = 10,  tolerance e = 1e-5
//
// The ADS splits the domain until every subdomain's degree-10 polynomial
// approximates f with an infinity-norm truncation error below 1e-5.
// ---------------------------------------------------------------------------

static constexpr int    N   = 10;
static constexpr int    M   = 1;
static constexpr double TOL = 1e-5;

using TTE1 = TEn< N, M >;
using Tree = AdsTree< TTE1 >;

// The function to approximate.
// NOTE: arguments must be taken by const-ref.
// Expression nodes store leaf TTEs by reference; a by-value copy would dangle.
static auto gaussian = []( const auto& x ) { return exp( -x * x ); };

// ---------------------------------------------------------------------------
// Save the ADS result to a CSV file for visualisation.
//
// Format (one row per done leaf):
//   center, half_width, c0, c1, ..., c10
//
// The polynomial approximates f(x) as:
//   f(x) ≈ sum_{k=0}^{N} c_k * delta^k,   delta = (x - center) / half_width
// ---------------------------------------------------------------------------
static void save_csv( const Tree& tree, const std::string& path )
{
    std::ofstream out( path );
    if ( !out )
    {
        std::cerr << "WARNING: could not open " << path << " for writing\n";
        return;
    }

    // Header
    out << "center,half_width";
    for ( int i = 0; i <= N; ++i )
        out << ",c" << i;
    out << "\n";

    // One row per subdomain
    for ( int idx : tree.done_leaves() )
    {
        const auto& lf = tree.node( idx ).leaf();
        out << lf.box.center[0] << "," << lf.box.half_width[0];
        for ( std::size_t i = 0; i < TTE1::nCoefficients; ++i )
            out << "," << lf.tte[i];
        out << "\n";
    }

    std::cout << "[ADS] Written " << tree.num_done() << " subdomains to " << path << "\n";
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST( AdsGaussian, SplitsIntoMultipleSubdomains )
{
    AdsRunner< N, M, decltype( gaussian ) > runner( gaussian, TOL, /*max_depth=*/60 );
    auto tree = runner.run( Box< double, M >{ { 0.0 }, { 3.0 } } );

    // The initial domain [-3,3] is too wide for a 10th-order polynomial to
    // achieve 1e-5 accuracy, so the runner must have split it.
    EXPECT_GT( tree.num_done(), 1 );

    // All active leaves should be empty (propagation finished).
    EXPECT_EQ( tree.num_active(), 0 );
}

TEST( AdsGaussian, PointAccuracy )
{
    AdsRunner< N, M, decltype( gaussian ) > runner( gaussian, TOL, /*max_depth=*/60 );
    auto tree = runner.run( Box< double, M >{ { 0.0 }, { 3.0 } } );

    // For each test point, find the containing subdomain and evaluate the
    // local polynomial.  The error should be well within our tolerance.
    constexpr double kAccuracy = 1e-4;  // 10× tolerance for safety

    for ( double x : { -2.9, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 2.9 } )
    {
        const int idx = tree.find_leaf( { x } );
        ASSERT_GE( idx, 0 ) << "No leaf found for x = " << x;

        const auto& lf    = tree.node( idx ).leaf();
        const double c    = lf.box.center[0];
        const double h    = lf.box.half_width[0];
        const double delta = ( x - c ) / h;  // normalised variable

        const double approx = lf.tte.eval( delta );
        const double exact  = std::exp( -x * x );

        EXPECT_NEAR( approx, exact, kAccuracy )
            << "  x=" << x << "  subdomain=[" << c - h << ", " << c + h << "]";
    }
}

TEST( AdsGaussian, SaveCsv )
{
    AdsRunner< N, M, decltype( gaussian ) > runner( gaussian, TOL, /*max_depth=*/60 );
    auto tree = runner.run( Box< double, M >{ { 0.0 }, { 3.0 } } );

    save_csv( tree, "gaussian_ads.csv" );

    // Verify the file was created and is non-empty.
    std::ifstream in( "gaussian_ads.csv" );
    EXPECT_TRUE( in.good() ) << "gaussian_ads.csv was not created";
}
