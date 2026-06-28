#include <gtest/gtest.h>

#include <array>
#include <tax/expansion/scheme.hpp>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/detail/algebra.hpp>
#include <tax/expansion/detail/transcendental.hpp>
#include <tax/tax.hpp>

using tax::Group;
using tax::MixedScheme;

// ---------------------------------------------------------------------------
// Oracle: for any in-box monomial α, the MixedScheme result coefficient at
// S::flatOf(α) must equal the same coefficient of an isotropic TE<Σorder,vars>
// computing the same expression. The box is a subset of the order-Σorder
// simplex and monomial degrees add, so no out-of-box factor can contribute to
// an in-box output.
// ---------------------------------------------------------------------------

// Shape used throughout: Group<1,4> x Group<1,3> → vars=2, Σorder=7.
// Box size = numMonomials(4,1) * numMonomials(3,1) = 5 * 4 = 20.
using S = MixedScheme< Group< 1, 4 >, Group< 1, 3 > >;
static_assert( S::vars == 2 );
static_assert( S::order == 7 );
static_assert( S::nCoeff == 20 );

constexpr int V = S::vars;
constexpr int SUM = 7;  // Σ order

using ISO = tax::TE< SUM, V >;  // order-7 simplex superset

// ---------------------------------------------------------------------------
// Helper: check every in-box slot matches the isotropic reference.
// ---------------------------------------------------------------------------
static void checkBoxVsIso( const std::array< double, S::nCoeff >& mixed, const ISO& iso,
                           const char* label )
{
    for ( std::size_t k = 0; k < S::nCoeff; ++k )
    {
        auto alpha = S::multiOf( k );
        double iso_val = iso[tax::flatIndex< V >( alpha )];
        EXPECT_NEAR( mixed[k], iso_val, 1e-12 ) << label << ": mismatch at flat=" << k << " alpha=("
                                                << alpha[0] << "," << alpha[1] << ")";
    }
}

// ---------------------------------------------------------------------------
// TEST 1: seriesExp — exp(c0 + x0) on both layouts.
// This is the exact example from the brief.
// ---------------------------------------------------------------------------
TEST( MixedKernels, ExpMatchesIsotropicSuperset )
{
    const double c0 = 0.3;

    // Mixed input: constant c0 at flat 0, linear in x0 (coefficient 1).
    std::array< double, S::nCoeff > a{};
    a[0] = c0;
    tax::MultiIndex< V > e0{ 1, 0 };
    a[S::flatOf( e0 )] = 1.0;

    std::array< double, S::nCoeff > mexp{};
    tax::detail::kernels::seriesExp< double, S >( mexp, a );

    // Isotropic: same polynomial c0 + x0 in the order-7 simplex.
    typename ISO::Input p{ c0, 0.0 };
    ISO ax = ISO::template variable< 0 >( p );
    auto iexp = exp( ax );

    checkBoxVsIso( mexp, iexp, "Exp" );
}

// ---------------------------------------------------------------------------
// TEST 2: seriesSqrt — sqrt(1.5 + x1) on both layouts.
// Seed constant away from 0; linear term on coord 1.
// ---------------------------------------------------------------------------
TEST( MixedKernels, SqrtMatchesIsotropicSuperset )
{
    const double c0 = 1.5;

    // Mixed input: constant c0, linear in x1.
    std::array< double, S::nCoeff > a{};
    a[0] = c0;
    tax::MultiIndex< V > e1{ 0, 1 };
    a[S::flatOf( e1 )] = 1.0;

    std::array< double, S::nCoeff > msqrt{};
    tax::detail::kernels::seriesSqrt< double, S >( msqrt, a );

    // Isotropic: same polynomial c0 + x1 in the order-7 simplex.
    // variable<1>(p) reads p[1] as the constant term, so set p[1] = c0.
    typename ISO::Input p{ 0.0, c0 };
    ISO ax = ISO::template variable< 1 >( p );
    auto isqrt = sqrt( ax );

    checkBoxVsIso( msqrt, isqrt, "Sqrt" );
}

// ---------------------------------------------------------------------------
// TEST 3: cauchyProduct — (a * b) on MixedScheme vs isotropic product.
// Use two simple polynomials: a = c0 + x0, b = d0 + x1.
// ---------------------------------------------------------------------------
TEST( MixedKernels, CauchyProductMatchesIsotropicSuperset )
{
    const double c0 = 0.7;
    const double d0 = 1.2;

    // Mixed inputs.
    std::array< double, S::nCoeff > ma{}, mb{};
    ma[0] = c0;
    tax::MultiIndex< V > e0{ 1, 0 };
    ma[S::flatOf( e0 )] = 1.0;

    mb[0] = d0;
    tax::MultiIndex< V > e1{ 0, 1 };
    mb[S::flatOf( e1 )] = 1.0;

    std::array< double, S::nCoeff > mprod{};
    tax::cauchyProduct< double, S >( mprod, ma, mb );

    // Isotropic counterparts: ax = c0 + x0, bx = d0 + x1 in order-7 simplex.
    // variable<I>(p) uses p[I] as the constant term.
    {
        typename ISO::Input pa{ c0, 0.0 };  // p[0]=c0 → variable<0> gives c0+x0
        typename ISO::Input pb{ 0.0, d0 };  // p[1]=d0 → variable<1> gives d0+x1
        ISO ax = ISO::template variable< 0 >( pa );
        ISO bx = ISO::template variable< 1 >( pb );
        auto iprod = ax * bx;
        checkBoxVsIso( mprod, iprod, "CauchyProduct" );
    }
}

// ---------------------------------------------------------------------------
// TEST 4: cauchySelfProduct — f*f on MixedScheme vs isotropic self-product.
// Use f = 0.4 + x0 + 0.5*x1.
// ---------------------------------------------------------------------------
TEST( MixedKernels, CauchySelfProductMatchesIsotropicSuperset )
{
    const double c0 = 0.4;

    // Mixed input: constant + linear combination.
    std::array< double, S::nCoeff > mf{};
    mf[0] = c0;
    tax::MultiIndex< V > e0{ 1, 0 };
    tax::MultiIndex< V > e1{ 0, 1 };
    mf[S::flatOf( e0 )] = 1.0;
    mf[S::flatOf( e1 )] = 0.5;

    std::array< double, S::nCoeff > mself{};
    tax::cauchySelfProduct< double, S >( mself, mf );

    // Isotropic counterpart: f = c0 + x0 + 0.5*x1 in order-7 simplex.
    {
        typename ISO::Input p{ c0, 0.0 };
        ISO ax = ISO::template variable< 0 >( p );
        typename ISO::Input q{ 0.0, 0.0 };
        ISO bx = ISO::template variable< 1 >( q );
        // Build f = c0 + x0 + 0.5*x1 from scratch.
        // ax is (c0 + x0), we need just x0: ax - c0. But let's build properly.
        typename ISO::Input px{ 0.0, 0.0 };
        ISO x0 = ISO::template variable< 0 >( px );
        ISO x1 = ISO::template variable< 1 >( px );
        // Construct the constant expansion c0.
        ISO cexp = ISO( c0 );
        ISO f = cexp + x0 + 0.5 * x1;
        auto iself = f * f;
        checkBoxVsIso( mself, iself, "CauchySelfProduct" );
    }
}
