#include <gtest/gtest.h>
#include <tax/expansion/detail/cauchy.hpp>
#include <tax/expansion/detail/cauchy_unroll.hpp>

template < int N >
void runDiffUni( double tol = 1e-12 )
{
    tax::Coeffs< double, N, 1 > a{}, b{}, out_loop{}, out_unroll{};
    for ( int k = 0; k <= N; ++k )
    {
        a[k] = 0.3 + 0.1 * k;
        b[k] = -0.2 + 0.05 * k;
    }
    tax::detail::kernels::cauchyProductLoop< double, N, 1 >( out_loop, a, b );
    tax::detail::kernels::cauchyProductUnroll< double, N, 1 >( out_unroll, a, b );
    for ( int k = 0; k <= N; ++k )
        EXPECT_NEAR( out_unroll[k], out_loop[k], tol ) << "N=" << N << " k=" << k;
}

TEST( CauchyUnroll, DiffMatchesLoop_N3 ) { runDiffUni< 3 >(); }
TEST( CauchyUnroll, DiffMatchesLoop_N5 ) { runDiffUni< 5 >(); }
TEST( CauchyUnroll, DiffMatchesLoop_N10 ) { runDiffUni< 10 >(); }
