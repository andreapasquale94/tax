#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/storage/sparse.hpp>
#include <vector>

namespace tax::detail::kernels
{

/// Scratch state shared by the sparse Cauchy kernels (thread_local; reused across calls, so no per-call heap allocation).
template < typename T, int N, int M >
struct SparseCauchyScratch
{
    static constexpr std::size_t NC = numMonomials( N, M );
    static constexpr std::size_t kWords = ( NC + 63 ) / 64;

    std::array< T, NC > acc{};
    std::array< std::uint64_t, kWords > touched{};
    std::vector< MultiIndex< M > > alpha;  // decoded support of one operand

    [[nodiscard]] static SparseCauchyScratch& instance() noexcept
    {
        static thread_local SparseCauchyScratch s;
        return s;
    }

    /// Decode a support span into multi-indices (reuses the member buffer).
    void decode( std::span< const storage::flat_index_t > support )
    {
        alpha.clear();
        alpha.reserve( support.size() );
        for ( const storage::flat_index_t k : support )
            alpha.push_back( unflatIndex< M >( std::size_t( k ) ) );
    }

    void mark( std::size_t k, T v ) noexcept
    {
        acc[k] += v;
        touched[k / 64] |= ( std::uint64_t{ 1 } << ( k & 63 ) );
    }

    /// Emit nonzero accumulator slots in ascending flat-index order and
    /// restore the scratch invariant (touched slots reset to zero).
    /// Not noexcept: appends to the output vectors, which may allocate.
    void emit( storage::SparseContainer< T, N, M >& out )
    {
        auto& ri = out.rawIndices();
        auto& rv = out.rawValues();
        for ( std::size_t w = 0; w < kWords; ++w )
        {
            std::uint64_t bits = touched[w];
            while ( bits )
            {
                const int b = std::countr_zero( bits );
                const std::size_t k = w * 64 + std::size_t( b );
                if ( acc[k] != T{ 0 } )
                {
                    ri.push_back( storage::flat_index_t( k ) );
                    rv.push_back( acc[k] );
                }
                acc[k] = T{ 0 };
                bits &= bits - 1;
            }
            touched[w] = 0;
        }
    }
};

/// Truncated sparse Cauchy product `out += f * g`, accumulating into `out`.
// Not noexcept: decode/emit append to vectors, which may allocate (matches the
// throwing sparse kernels in sparse_subs.hpp).
template < typename T, int N, int M >
void sparseCauchyProduct( storage::SparseContainer< T, N, M >& out,
                          const storage::SparseContainer< T, N, M >& f,
                          const storage::SparseContainer< T, N, M >& g )
{
    static const DegreeOf< N, M > deg_table{};

    const auto fi = f.support();
    const auto fv = f.values();
    const auto gi = g.support();
    const auto gv = g.values();

    auto& scratch = SparseCauchyScratch< T, N, M >::instance();
    scratch.decode( gi );

    for ( std::size_t a = 0; a < fi.size(); ++a )
    {
        const std::size_t ia = fi[a];
        const int da = deg_table.value[ia];
        const auto alpha_a = unflatIndex< M >( ia );
        const T va = fv[a];
        for ( std::size_t b = 0; b < gi.size(); ++b )
        {
            if ( da + deg_table.value[gi[b]] > N )
                break;  // graded order: all later degrees are >= this one
            MultiIndex< M > sum{};
            for ( int i = 0; i < M; ++i )
                sum[std::size_t( i )] =
                    alpha_a[std::size_t( i )] + scratch.alpha[b][std::size_t( i )];
            scratch.mark( flatIndex< M >( sum ), va * gv[b] );
        }
    }

    scratch.emit( out );
}

/// Truncated sparse self-product `out = f * f`, exploiting pair symmetry.
// Not noexcept: decode/emit append to vectors, which may allocate.
template < typename T, int N, int M >
void sparseCauchySelfProduct( storage::SparseContainer< T, N, M >& out,
                              const storage::SparseContainer< T, N, M >& f )
{
    static const DegreeOf< N, M > deg_table{};

    const auto fi = f.support();
    const auto fv = f.values();

    auto& scratch = SparseCauchyScratch< T, N, M >::instance();
    scratch.decode( fi );

    for ( std::size_t a = 0; a < fi.size(); ++a )
    {
        const std::size_t ia = fi[a];
        const int da = deg_table.value[ia];
        const auto& alpha_a = scratch.alpha[a];
        const T va = fv[a];

        // Diagonal: f[ia]^2 contributes once (if 2*da <= N).
        if ( 2 * da <= N )
        {
            MultiIndex< M > sum{};
            for ( int i = 0; i < M; ++i ) sum[std::size_t( i )] = 2 * alpha_a[std::size_t( i )];
            scratch.mark( flatIndex< M >( sum ), va * va );
        }
        // Off-diagonal: pair {ia, ib} with ib > ia contributes 2*va*vb.
        for ( std::size_t b = a + 1; b < fi.size(); ++b )
        {
            if ( da + deg_table.value[fi[b]] > N )
                break;  // graded order: all later degrees are >= this one
            MultiIndex< M > sum{};
            for ( int i = 0; i < M; ++i )
                sum[std::size_t( i )] =
                    alpha_a[std::size_t( i )] + scratch.alpha[b][std::size_t( i )];
            scratch.mark( flatIndex< M >( sum ), T{ 2 } * va * fv[b] );
        }
    }

    scratch.emit( out );
}

}  // namespace tax::detail::kernels
