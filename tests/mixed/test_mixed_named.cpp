#include <gtest/gtest.h>

#include <tax/tax.hpp>

TEST( MixedNamed, ConstructAndType )
{
    auto x = tax::mixed::variable< "x", 4 >( 1.0 );  // axis "x" dim 1 order 4
    using X = decltype( x );
    static_assert( X::vars_v == 1 );
    static_assert( X::Inner::nCoefficients == 5 );  // numMonomials(4,1)
    EXPECT_DOUBLE_EQ( x.value(), 1.0 );
}

TEST( MixedNamed, VariablesArrayAndAxisDim )
{
    std::array< double, 3 > p{ 0.1, 0.2, 0.3 };
    auto v = tax::mixed::variables< "p", 6, 3 >( p );  // 3-D axis "p" order 6
    // v is std::array; v[0] is a reference, so decay before member access.
    static_assert( std::decay_t< decltype( v[0] ) >::vars_v == 3 );
    static_assert( std::tuple_size_v< decltype( v ) > == 3 );
    EXPECT_DOUBLE_EQ( v[0].value(), 0.1 );
    EXPECT_DOUBLE_EQ( v[2].value(), 0.3 );
}

// scalar / expansion (reciprocal): the scalar-lhs division operator must exist
// for mixed-order named expansions, matching the named (single-order) surface.
TEST( MixedNamed, ScalarDivideExpansion )
{
    auto x = tax::mixed::variable< "x", 4 >( 2.0 );
    auto f = 6.0 / x;  // scalar / expansion
    EXPECT_NEAR( f.value(), 3.0, 1e-12 );
    auto ref = 6.0 / x.inner();
    for ( std::size_t k = 0; k < decltype( x )::Inner::nCoefficients; ++k )
        EXPECT_NEAR( f.inner()[k], ref[k], 1e-12 );
}

TEST( MixedNamed, ComposeUnionAxesNoBlowup )
{
    auto x = tax::mixed::variable< "x", 4 >( 0.3 );
    auto t = tax::mixed::variable< "t", 20 >( 0.1 );
    auto f = x * t + x;  // union {t@20, x@4} (sorted by name)
    using F = decltype( f );
    static_assert( F::vars_v == 2 );
    // box size = numMonomials(4,1) * numMonomials(20,1) = 5 * 21 = 105 (NOT (24+2 choose 2))
    static_assert( F::Inner::nCoefficients == 105 );

    // Pin the x*t coefficient. Union axes sorted by name: t (var 0), x (var 1).
    // The x*t monomial is exponent (t=1, x=1) in the union variable layout.
    tax::MultiIndex< 2 > xt{};
    xt[0] = 1;  // t
    xt[1] = 1;  // x
    EXPECT_NEAR( f.inner()[F::Inner::scheme::flatOf( xt )], 1.0, 1e-12 );
}

// canonical type equality: x*t and t*x are the same type
TEST( MixedNamed, CanonicalTypeOrderIndependent )
{
    auto x = tax::mixed::variable< "x", 4 >( 0.3 );
    auto t = tax::mixed::variable< "t", 20 >( 0.1 );
    static_assert( std::is_same_v< decltype( x * t ), decltype( t * x ) > );
    SUCCEED();
}

// max-order promotion on a shared axis
TEST( MixedNamed, SharedAxisPromotesToMaxOrder )
{
    auto x2 = tax::mixed::variable< "x", 2 >( 0.3 );
    auto x5 = tax::mixed::variable< "x", 5 >( 0.3 );
    auto p = x2 * x5;                                           // shared axis x -> order 5
    static_assert( decltype( p )::Inner::nCoefficients == 6 );  // numMonomials(5,1)
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Task 3: axis-name differential / projection ops
// ---------------------------------------------------------------------------

// deriv<"x"> of a composed {t@4, x@3} expansion matches the analytical derivative
// for monomials that don't require source terms above the x-order cap.
// We verify two things:
//  (1) the type of df is the same as f (axis set and orders preserved)
//  (2) for monomials (t^a, x^b) with x-degree b < x-order (= 3), df matches
//      the finite-difference derivative in x from the isotropic oracle.
TEST( MixedNamed, DerivByAxisNameMatchesIsotropicOracle )
{
    // small orders to keep the isotropic oracle affordable: x@3, t@4 (Σ=7)
    auto x = tax::mixed::variable< "x", 3 >( 0.7 );
    auto t = tax::mixed::variable< "t", 4 >( 1.3 );
    auto f = sin( x * t ) + exp( x );
    using F = decltype( f );
    // Union axes sorted by name: t (var 0), x (var 1)
    auto df = f.template deriv< "x" >();
    using DF = decltype( df );
    static_assert( std::is_same_v< DF, F >, "deriv preserves the axis set and orders" );

    // Isotropic oracle: t=var0 at 1.3, x=var1 at 0.7
    typename tax::TE< 7, 2 >::Input p{ 1.3, 0.7 };
    auto it = tax::TE< 7, 2 >::variable< 0 >( p );
    auto ix = tax::TE< 7, 2 >::variable< 1 >( p );
    auto iso = sin( ix * it ) + exp( ix );
    auto iso_dx = iso.template deriv< 1 >();  // deriv w.r.t. var 1 (x)

    // Compare only monomials where the x-degree in the RESULT is < x-order (= 3).
    // For those, the deriv coefficient comes from source monomials with x-degree <= x-order,
    // which are all present in the mixed box.  Monomials with x-degree = 3 in df would
    // require x-degree 4 source terms, which are truncated in the mixed box but present
    // in the isotropic oracle — so we skip them.
    constexpr int x_order = 3;
    for ( std::size_t k = 0; k < F::Inner::nCoefficients; ++k )
    {
        const auto alpha = F::Inner::scheme::multiOf( k );
        const int x_deg = alpha[1];        // union layout: t=var0, x=var1
        if ( x_deg >= x_order ) continue;  // skip boundary: iso includes x^{x_order+1} source
        EXPECT_NEAR( df.inner()[k], iso_dx[tax::flatIndex< 2 >( alpha )], 1e-12 )
            << "coeff " << k << " alpha=(" << alpha[0] << "," << alpha[1] << ")";
    }
}

TEST( MixedNamed, IntegByAxisNameMatchesIsotropicOracle )
{
    auto x = tax::mixed::variable< "x", 3 >( 0.7 );
    auto t = tax::mixed::variable< "t", 4 >( 1.3 );
    auto f = sin( x * t ) + exp( x );
    using F = decltype( f );
    auto fi = f.template integ< "t" >();
    using FI = decltype( fi );
    static_assert( std::is_same_v< FI, F >, "integ preserves the axis set and orders" );

    // Isotropic oracle: t=var0 at 1.3, x=var1 at 0.7
    typename tax::TE< 7, 2 >::Input p{ 1.3, 0.7 };
    auto it = tax::TE< 7, 2 >::variable< 0 >( p );
    auto ix = tax::TE< 7, 2 >::variable< 1 >( p );
    auto iso = sin( ix * it ) + exp( ix );
    auto iso_it = iso.template integ< 0 >();  // integ w.r.t. var 0 (t)

    for ( std::size_t k = 0; k < F::Inner::nCoefficients; ++k )
    {
        const auto alpha = F::Inner::scheme::multiOf( k );
        EXPECT_NEAR( fi.inner()[k], iso_it[tax::flatIndex< 2 >( alpha )], 1e-12 ) << "coeff " << k;
    }
}

// slice<"x"> of an {t@4,x@3} expansion: result has only axis x;
// every monomial with t-degree>0 dropped; x-only coefficients match the source.
TEST( MixedNamed, SliceByAxisName )
{
    auto x = tax::mixed::variable< "x", 3 >( 0.7 );
    auto t = tax::mixed::variable< "t", 4 >( 1.3 );
    // f = x + x*t + t^2 (plus constant from expansion points)
    auto f = x + x * t;
    using F = decltype( f );
    // Union axes: t(var0), x(var1)

    auto sx = f.template slice< "x" >();
    using SX = decltype( sx );
    // Result should be a MixedTaylorExpansion over only axis "x" at order 3
    using ExpectedAx = tax::named::OrderedAxis< "x", 1, 3 >;
    static_assert( std::is_same_v< SX, tax::named::MixedTaylorExpansion< double, ExpectedAx > > );

    // The x-only slice should keep only monomials with t-degree = 0.
    // f = x + x*t + ... ; slicing drops the x*t term.
    // At expansion point x=0.7, t=1.3:
    // f value = 0.7 + 0.7*1.3 = 0.7*2.3 = 1.61, dx coeff = 1 + 1.3 = 2.3, dt coeff = 0.7, etc.
    // slice keeps: value=1.61, dx coeff=2.3
    // (only monomials with t-exponent=0)
    EXPECT_NEAR( sx.value(), 0.7 + 0.7 * 1.3, 1e-12 );

    // dx coefficient: from f, coeff of dt^0*dx^1 is (1 + 1.3) = 2.3
    tax::MultiIndex< 1 > alpha_x1{};
    alpha_x1[0] = 1;
    EXPECT_NEAR( sx.inner()[SX::Inner::scheme::flatOf( alpha_x1 )], 1.0 + 1.3, 1e-12 );

    // Every source monomial with t-degree > 0 must be absent in sx
    // In SX there are no t variables, so check that x*t cross terms are gone
    for ( std::size_t k = 0; k < F::Inner::nCoefficients; ++k )
    {
        const auto alpha = F::Inner::scheme::multiOf( k );
        if ( alpha[0] > 0 )  // t-degree > 0
        {
            // The source's t-containing monomials should not appear in the slice
            // (Nothing to verify in the output since they are dropped)
            (void)alpha;
        }
    }
}

// truncate<"t",2> of {t@20,x@4}: result is {t@2,x@4}; coefficients with t-degree<=2
// are preserved, t-degree>2 are dropped.
TEST( MixedNamed, TruncateByAxisName )
{
    auto x = tax::mixed::variable< "x", 4 >( 0.5 );
    auto t = tax::mixed::variable< "t", 20 >( 0.2 );
    auto f = x * t + x;  // simple enough to have known coefficients
    using F = decltype( f );
    // Union: t(var0,order20), x(var1,order4). Box: 21*5=105

    auto ft = f.truncate< "t", 2 >();
    using FT = decltype( ft );
    // Expected: {t@2, x@4} — box = numMonomials(2,1)*numMonomials(4,1) = 3*5 = 15
    using ExpAxT = tax::named::OrderedAxis< "t", 1, 2 >;
    using ExpAxX = tax::named::OrderedAxis< "x", 1, 4 >;
    static_assert(
        std::is_same_v< FT, tax::named::MixedTaylorExpansion< double, ExpAxT, ExpAxX > > );
    static_assert( FT::Inner::nCoefficients == 15 );

    // Coefficients with t-degree <= 2 must match the original.
    for ( std::size_t k = 0; k < FT::Inner::nCoefficients; ++k )
    {
        const auto alpha = FT::Inner::scheme::multiOf( k );
        // alpha[0] = t-degree, alpha[1] = x-degree; t-degree <= 2 by construction of FT
        // Find the same coefficient in the source.
        const std::size_t k_src = F::Inner::scheme::flatOf( alpha );
        EXPECT_NEAR( ft.inner()[k], f.inner()[k_src], 1e-12 ) << "coeff " << k;
    }

    // Also verify that the source's t-degree>2 terms are NOT in the truncated result.
    // The truncated result has only 15 coefficients so they cannot appear by construction.
    static_assert( FT::Inner::nCoefficients < F::Inner::nCoefficients );
}

// ---------------------------------------------------------------------------
// Task 5 coverage tests
// ---------------------------------------------------------------------------

// Dim>=2 shared axis through promotion/embed: two MixedTaylorExpansions that
// share a 2-D axis "q" (at different per-axis orders), each also carrying a
// distinct 1-D axis, are multiplied.  The result type must be
//   { "p"@3, "q"@max(2,3)=3, "r"@2 }  (sorted by name)
// and every box coefficient must match the isotropic TE<8,4> oracle evaluated
// at the same expansion point.  This exercises embedMixed / MergeOrdered for
// dim>1 shared axes (only dim==1 shared axes had prior coverage).
TEST( MixedNamed, Dim2SharedAxisPromoteEmbed )
{
    // A: "p"(dim=1,ord=3)  x  "q"(dim=2,ord=2)
    // B: "q"(dim=2,ord=3)  x  "r"(dim=1,ord=2)
    // Canonical axis order in result: "p"(var0), "q"(var1,var2), "r"(var3)
    // Expansion points: p=0.5, q0=0.3, q1=0.7, r=1.1
    std::array< double, 2 > q_pts{ 0.3, 0.7 };
    auto pa = tax::mixed::variable< "p", 3 >( 0.5 );
    auto qa = tax::mixed::variables< "q", 2, 2 >( q_pts );  // A carries q@2
    auto qb = tax::mixed::variables< "q", 3, 2 >( q_pts );  // B carries q@3
    auto rb = tax::mixed::variable< "r", 2 >( 1.1 );

    // Build A = pa * qa[0] + qa[1]   (uses "p" and "q")
    auto fa = pa * qa[0] + qa[1];
    // Build B = qb[0] * rb + qb[1]   (uses "q" and "r")
    auto fb = qb[0] * rb + qb[1];
    // Compose: merged axis set {"p"@3, "q"@3, "r"@2}
    auto f = fa * fb;
    using F = decltype( f );

    // Type check: 4 variables total (1+2+1), box =
    // numMonomials(3,1)*numMonomials(3,2)*numMonomials(2,1) = 4 * 10 * 3 = 120
    static_assert( F::vars_v == 4 );
    static_assert( F::Inner::nCoefficients == 120 );

    // Isotropic oracle: TE<8,4> with vars p(var0), q0(var1), q1(var2), r(var3)
    // at expansion points p=0.5, q0=0.3, q1=0.7, r=1.1
    typename tax::TE< 8, 4 >::Input pt{ 0.5, 0.3, 0.7, 1.1 };
    auto ip = tax::TE< 8, 4 >::variable< 0 >( pt );
    auto iq0 = tax::TE< 8, 4 >::variable< 1 >( pt );
    auto iq1 = tax::TE< 8, 4 >::variable< 2 >( pt );
    auto ir = tax::TE< 8, 4 >::variable< 3 >( pt );
    auto iso_a = ip * iq0 + iq1;
    auto iso_b = iq0 * ir + iq1;
    auto iso = iso_a * iso_b;

    // Every mixed box coefficient must match the isotropic oracle at the same multi-index.
    for ( std::size_t k = 0; k < F::Inner::nCoefficients; ++k )
    {
        const auto alpha = F::Inner::scheme::multiOf( k );
        // alpha has 4 entries: [p, q0, q1, r] matching the isotropic variable layout.
        EXPECT_NEAR( f.inner()[k], iso[tax::flatIndex< 4 >( alpha )], 1e-12 )
            << "coeff " << k << " alpha=(" << alpha[0] << "," << alpha[1] << "," << alpha[2] << ","
            << alpha[3] << ")";
    }
}

// Slice with ALL axes named (degenerate case: no axes are dropped).
// slice<"a","b">() of an {a@3, b@2} expansion must return an equivalent
// expansion of the same type, with every coefficient preserved.
TEST( MixedNamed, SliceAllAxesNoDrop )
{
    auto a = tax::mixed::variable< "a", 3 >( 0.4 );
    auto b = tax::mixed::variable< "b", 2 >( 0.9 );
    auto f = sin( a ) * exp( b ) + a * b;
    using F = decltype( f );

    // Sorted canonical order: "a"(var0,ord=3), "b"(var1,ord=2)
    auto s = f.template slice< "a", "b" >();
    using S = decltype( s );

    // The slice result must be the same type as f (no axes dropped, same orders).
    static_assert( std::is_same_v< S, F > );

    // Every coefficient must be identical.
    for ( std::size_t k = 0; k < F::Inner::nCoefficients; ++k )
    {
        EXPECT_DOUBLE_EQ( s.inner()[k], f.inner()[k] ) << "coeff " << k;
    }
}

// isotropic-superset oracle: every box coefficient of a composed mixed expansion
// must equal the same coefficient of the full isotropic TE<Σorder, vars>.
TEST( MixedNamed, IsotropicSupersetOracle )
{
    // Small per-axis orders so the isotropic super-box is affordable: x@3, t@4 (Σ=7).
    auto x = tax::mixed::variable< "x", 3 >( 0.7 );
    auto t = tax::mixed::variable< "t", 4 >( 1.3 );
    auto f = sin( x * t ) + exp( x );
    using F = decltype( f );

    // Isotropic oracle over the same two variables at the same expansion points.
    // Union axes sorted by name: t (var 0, x0=1.3), x (var 1, x0=0.7).
    typename tax::TE< 7, 2 >::Input p{ 1.3, 0.7 };
    auto it = tax::TE< 7, 2 >::variable< 0 >( p );
    auto ix = tax::TE< 7, 2 >::variable< 1 >( p );
    auto iso = sin( ix * it ) + exp( ix );

    for ( std::size_t k = 0; k < F::Inner::nCoefficients; ++k )
    {
        const auto alpha = F::Inner::scheme::multiOf( k );
        EXPECT_NEAR( f.inner()[k], iso[tax::flatIndex< 2 >( alpha )], 1e-12 );
    }
}

// embed<R>() into a target with a LOWER per-axis order must DROP out-of-box
// monomials, not write out of bounds (an unguarded out[flatOf == kNotInBox] is a
// stack OOB write at runtime and ill-formed in constant evaluation). Run in a
// constexpr context so a regression fails to COMPILE here.
TEST( MixedNamed, EmbedToLowerOrderDropsOutOfBox )
{
    using Src = tax::MixedTaylorExpansion< double, tax::OrderedAxis< "x", 1, 4 > >;
    using Tgt = tax::MixedTaylorExpansion< double, tax::OrderedAxis< "x", 1, 1 > >;  // lower order

    constexpr Tgt t = []() constexpr {
        Src s{};
        s[Src::Inner::scheme::flatOf( tax::MultiIndex< 1 >{ 0 } )] = 4.0;  // constant
        s[Src::Inner::scheme::flatOf( tax::MultiIndex< 1 >{ 1 } )] = 4.0;  // x^1
        s[Src::Inner::scheme::flatOf( tax::MultiIndex< 1 >{ 2 } )] = 1.0;  // x^2: out of Tgt box
        return s.template embed< Tgt >();
    }();

    static_assert( Tgt::nCoefficients == 2 );
    EXPECT_DOUBLE_EQ( t[0], 4.0 );  // constant kept
    EXPECT_DOUBLE_EQ( t[1], 4.0 );  // x^1 kept; x^2 (out of box) dropped, not OOB-written
}
