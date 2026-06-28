#include <gtest/gtest.h>

#include <tax/expansion/detail/algebra.hpp>
#include <tax/expansion/detail/cauchy.hpp>
#include <tax/expansion/detail/cauchy_stencil.hpp>
#include <tax/expansion/detail/recurrence_stencil.hpp>

template < int N, int M >
void runDiffMulti( double tol = 1e-12 )
{
    tax::Coeffs< double, N, M > a{}, b{}, out_loop{}, out_sten{};
    for ( std::size_t k = 0; k < a.size(); ++k )
    {
        a[k] = 0.1 * double( k + 1 ) - 0.4;
        b[k] = 0.2 - 0.05 * double( k );
    }
    tax::detail::kernels::cauchyProductLoop< double, N, M >( out_loop, a, b );
    tax::detail::kernels::cauchyProductStencil< double, N, M >( out_sten, a, b );
    for ( std::size_t k = 0; k < a.size(); ++k )
        EXPECT_NEAR( out_sten[k], out_loop[k], tol ) << "N=" << N << " M=" << M << " k=" << k;
}

TEST( CauchyStencil, Diff_N3_M2 ) { runDiffMulti< 3, 2 >(); }
TEST( CauchyStencil, Diff_N4_M3 ) { runDiffMulti< 4, 3 >(); }
TEST( CauchyStencil, Diff_N5_M4 ) { runDiffMulti< 5, 4 >(); }

// Entry count is exact: ordered pairs (beta, gamma) with |beta| + |gamma| <= N
// biject with monomials of degree <= N in 2M variables.
static_assert( tax::detail::kernels::CauchyStencil< 3, 2 >::kEntries == tax::numMonomials( 3, 4 ) );
static_assert( tax::detail::kernels::CauchyStencil< 4, 3 >::kEntries == tax::numMonomials( 4, 6 ) );

// cauchyProduct must remain usable in constant evaluation for M >= 2
// (the stencil path is runtime-only; constexpr callers take the loop kernel).
constexpr tax::Coeffs< double, 2, 2 > kConstexprProduct = [] {
    tax::Coeffs< double, 2, 2 > a{}, b{}, out{};
    a[0] = 1.0;  // 1 + 2x
    a[1] = 2.0;
    b[0] = 3.0;  // 3 + 4y
    b[2] = 4.0;
    tax::detail::kernels::cauchyProduct< double, 2, 2 >( out, a, b );
    return out;
}();
// (1 + 2x)(3 + 4y) = 3 + 6x + 4y + 8xy
static_assert( kConstexprProduct[0] == 3.0 );
static_assert( kConstexprProduct[1] == 6.0 );
static_assert( kConstexprProduct[2] == 4.0 );
static_assert( kConstexprProduct[3] == 0.0 );
static_assert( kConstexprProduct[4] == 8.0 );
static_assert( kConstexprProduct[5] == 0.0 );

TEST( CauchyStencil, ConstexprFallbackCompiles )
{
    SUCCEED();  // assertions above are compile-time
}

// cauchySelfProduct (M >= 2) must agree between constant evaluation (loop kernel)
// and runtime (stencil). Both paths route through cauchyProduct and perform the
// same ordered convolution, so they agree to floating-point round-off. We do NOT
// assert bit-identity: a compiler may contract the runtime path's a*b+c into a
// fused multiply-add (one rounding) while the constant-evaluated path is not
// contracted (two roundings), so the two can differ by a ULP (observed on Clang
// and on macOS GCC). A tight absolute tolerance captures the real guarantee
// portably — matching the runtime-vs-runtime Diff tests above.
template < int N, int M >
constexpr tax::Coeffs< double, N, M > selfProductConstexpr()
{
    tax::Coeffs< double, N, M > f{}, out{};
    for ( std::size_t k = 0; k < f.size(); ++k ) f[k] = 0.1 * double( k + 1 ) - 0.37;
    tax::cauchySelfProduct< double, tax::IsotropicScheme< N, M > >( out, f );
    return out;
}

template < int N, int M >
void runSelfProductConstevalMatchesRuntime()
{
    constexpr tax::Coeffs< double, N, M > ct = selfProductConstexpr< N, M >();

    tax::Coeffs< double, N, M > f{}, rt{};
    for ( std::size_t k = 0; k < f.size(); ++k ) f[k] = 0.1 * double( k + 1 ) - 0.37;
    tax::cauchySelfProduct< double, tax::IsotropicScheme< N, M > >( rt, f );  // runtime path

    for ( std::size_t k = 0; k < f.size(); ++k )
        EXPECT_NEAR( rt[k], ct[k], 1e-12 ) << "N=" << N << " M=" << M << " k=" << k;
}

TEST( CauchySelfProduct, ConstevalMatchesRuntime_N3_M2 )
{
    runSelfProductConstevalMatchesRuntime< 3, 2 >();
}
TEST( CauchySelfProduct, ConstevalMatchesRuntime_N4_M3 )
{
    runSelfProductConstevalMatchesRuntime< 4, 3 >();
}

// RecurrenceStencil entry count: the |beta| >= 1 decompositions are the
// Cauchy pairs minus the |beta| = 0 ones (one per gamma).
static_assert( tax::detail::kernels::RecurrenceStencil< 3, 2 >::kEntries ==
               tax::numMonomials( 3, 4 ) - tax::numMonomials( 3, 2 ) );
static_assert( tax::detail::kernels::RecurrenceStencil< 4, 3 >::kEntries ==
               tax::numMonomials( 4, 6 ) - tax::numMonomials( 4, 3 ) );

// seriesReciprocal must remain usable in constant evaluation for M >= 2
// (forEachRecurrenceRow enumerates rows on the fly when consteval).
constexpr tax::Coeffs< double, 2, 2 > kConstexprReciprocal = [] {
    tax::Coeffs< double, 2, 2 > a{}, out{};
    a[0] = 2.0;  // 2 + x
    a[1] = 1.0;
    tax::detail::kernels::seriesReciprocal< double, tax::IsotropicScheme< 2, 2 > >( out, a );
    return out;
}();
// 1/(2 + x) = 1/2 - x/4 + x^2/8 + O(x^3)
static_assert( kConstexprReciprocal[0] == 0.5 );
static_assert( kConstexprReciprocal[1] == -0.25 );
static_assert( kConstexprReciprocal[2] == 0.0 );
static_assert( kConstexprReciprocal[3] == 0.125 );

// Runtime stencil walk and constant-evaluation fallback must produce
// identical rows (same order, same entries).
TEST( RecurrenceStencil, RowsMatchConstexprEnumeration )
{
    using tax::detail::kernels::RecurrenceEntry;
    constexpr int N = 4, M = 3;
    std::vector< RecurrenceEntry > table_rows, enum_rows;
    std::vector< std::size_t > table_bounds, enum_bounds;

    const auto& st = tax::detail::kernels::recurrenceStencil< N, M >();
    for ( std::size_t ai = 1; ai < tax::numMonomials( N, M ); ++ai )
    {
        table_bounds.push_back( st.row[ai + 1] - st.row[ai] );
        for ( std::size_t e = st.row[ai]; e < st.row[ai + 1]; ++e )
            table_rows.push_back( st.entries[e] );
    }

    std::size_t ai_expected = 1;
    tax::forEachMonomial< M, N >( [&]( const tax::MultiIndex< M >& alpha ) {
        int d = 0;
        for ( int i = 0; i < M; ++i ) d += alpha[std::size_t( i )];
        if ( d == 0 ) return;
        ASSERT_EQ( tax::flatIndex< M >( alpha ), ai_expected++ );
        std::size_t count = 0;
        tax::forEachSubIndex< M >(
            alpha, [&]( const tax::MultiIndex< M >& beta, const tax::MultiIndex< M >& gamma ) {
                int db = 0;
                for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                if ( db == 0 ) return;
                enum_rows.push_back( RecurrenceEntry{ std::uint32_t( tax::flatIndex< M >( beta ) ),
                                                      std::uint32_t( tax::flatIndex< M >( gamma ) ),
                                                      std::uint32_t( db ) } );
                ++count;
            } );
        enum_bounds.push_back( count );
    } );

    ASSERT_EQ( table_bounds, enum_bounds );
    ASSERT_EQ( table_rows.size(), enum_rows.size() );
    for ( std::size_t i = 0; i < table_rows.size(); ++i )
    {
        EXPECT_EQ( table_rows[i].b_idx, enum_rows[i].b_idx ) << "entry " << i;
        EXPECT_EQ( table_rows[i].g_idx, enum_rows[i].g_idx ) << "entry " << i;
        EXPECT_EQ( table_rows[i].db, enum_rows[i].db ) << "entry " << i;
    }
}
