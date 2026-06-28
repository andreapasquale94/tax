#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/storage/sparse.hpp>
#include <tax/expansion/detail/sparse_cauchy.hpp>

namespace tax::detail::kernels
{

/// Additive closure of seed multi-indices under multi-index addition, truncated to total degree `N`. Always includes flat index `0`.
template < int N, int M >
[[nodiscard]] inline std::vector< storage::flat_index_t >
additiveClosure( const std::vector< storage::flat_index_t >& seeds )
{
    constexpr std::size_t NC = numMonomials( N, M );
    std::vector< bool >                   seen( NC, false );
    seen[0] = true;
    std::vector< storage::flat_index_t > frontier{ 0 };

    while ( !frontier.empty() )
    {
        std::vector< storage::flat_index_t > next;
        for ( storage::flat_index_t a : frontier )
        {
            const auto alpha = unflatIndex< M >( std::size_t( a ) );
            for ( storage::flat_index_t s : seeds )
            {
                const auto beta = unflatIndex< M >( std::size_t( s ) );
                MultiIndex< M > sum_idx{};
                int deg = 0;
                for ( int i = 0; i < M; ++i )
                {
                    sum_idx[std::size_t( i )] = alpha[std::size_t( i )] + beta[std::size_t( i )];
                    deg += sum_idx[std::size_t( i )];
                }
                if ( deg > N ) continue;
                const std::size_t flat = flatIndex< M >( sum_idx );
                if ( !seen[flat] )
                {
                    seen[flat] = true;
                    next.push_back( storage::flat_index_t( flat ) );
                }
            }
        }
        frontier = std::move( next );
    }

    std::vector< storage::flat_index_t > result;
    result.reserve( NC );
    for ( std::size_t k = 0; k < NC; ++k )
        if ( seen[k] ) result.push_back( storage::flat_index_t( k ) );
    return result;
}

/// Sparse reciprocal `out = 1 / f` via forward substitution. Throws std::domain_error if the constant term is zero.
template < typename T, int N, int M >
void seriesReciprocalSparse( storage::SparseContainer< T, N, M >&       out,
                              const storage::SparseContainer< T, N, M >& f )
{
    constexpr std::size_t NC = numMonomials( N, M );

    const auto fi = f.support();
    const auto fv = f.values();
    if ( fi.empty() || fi.front() != 0 || fv.front() == T{ 0 } )
        throw std::domain_error(
            "seriesReciprocalSparse: constant term must be nonzero" );

    const T inv_f0 = T{ 1 } / fv.front();

    // Seeds: f's nonzero perturbation indices (flat > 0).
    std::vector< storage::flat_index_t > seeds;
    seeds.reserve( fi.size() );
    for ( std::size_t k = 1; k < fi.size(); ++k )
        seeds.push_back( fi[k] );

    const auto support = additiveClosure< N, M >( seeds );

    // Dense scratch for O(1) lookup of out[gamma] in the recurrence.
    std::vector< T > scratch( NC, T{ 0 } );
    scratch[0] = inv_f0;

    // For each α in support[1..end]:
    //   out[α] = -inv_f0 * sum_{β in f, β>0, β<=α} f[β] * out[α-β]
    for ( std::size_t k = 1; k < support.size(); ++k )
    {
        const std::size_t ai    = support[k];
        const auto        alpha = unflatIndex< M >( ai );
        T                 acc{ 0 };
        for ( std::size_t j = 1; j < fi.size(); ++j )
        {
            const auto beta = unflatIndex< M >( std::size_t( fi[j] ) );
            MultiIndex< M > gamma{};
            bool valid = true;
            for ( int i = 0; i < M; ++i )
            {
                if ( beta[std::size_t( i )] > alpha[std::size_t( i )] )
                {
                    valid = false;
                    break;
                }
                gamma[std::size_t( i )] = alpha[std::size_t( i )] - beta[std::size_t( i )];
            }
            if ( !valid ) continue;
            const std::size_t gi = flatIndex< M >( gamma );
            acc += fv[j] * scratch[gi];
        }
        scratch[ai] = -inv_f0 * acc;
    }

    // Emit nonzero results in ascending order.
    auto& ri = out.rawIndices();
    auto& rv = out.rawValues();
    for ( storage::flat_index_t k : support )
    {
        if ( scratch[k] != T{ 0 } )
        {
            ri.push_back( k );
            rv.push_back( scratch[k] );
        }
    }
}

/// Sparse square root `out = sqrt(f)` via forward substitution. Throws std::domain_error if the constant term is <= 0.
template < typename T, int N, int M >
void seriesSqrtSparse( storage::SparseContainer< T, N, M >&       out,
                       const storage::SparseContainer< T, N, M >& f )
{
    constexpr std::size_t NC = numMonomials( N, M );

    const auto fi = f.support();
    const auto fv = f.values();
    const T    f0 = ( !fi.empty() && fi.front() == 0 ) ? fv.front() : T{ 0 };
    if ( !( f0 > T{ 0 } ) )
        throw std::domain_error(
            "seriesSqrtSparse: constant term must be strictly positive" );

    const T sqrt_f0  = std::sqrt( f0 );
    const T inv2sqrt = T{ 1 } / ( T{ 2 } * sqrt_f0 );

    std::vector< storage::flat_index_t > seeds;
    seeds.reserve( fi.size() );
    for ( std::size_t k = 1; k < fi.size(); ++k )
        seeds.push_back( fi[k] );

    const auto support = additiveClosure< N, M >( seeds );

    // Dense scratch for f and out (for O(1) inner-loop lookup).
    std::vector< T > f_dense( NC, T{ 0 } );
    for ( std::size_t k = 0; k < fi.size(); ++k )
        f_dense[fi[k]] = fv[k];

    std::vector< T > scratch( NC, T{ 0 } );
    scratch[0] = sqrt_f0;

    // For each α in support[1..end]:
    //   out[α] = (f[α] - sum_{β+γ=α, β,γ>0, both in support} out[β]*out[γ]) / (2*out[0])
    for ( std::size_t k = 1; k < support.size(); ++k )
    {
        const std::size_t ai    = support[k];
        const auto        alpha = unflatIndex< M >( ai );
        T                 acc   = f_dense[ai];
        for ( std::size_t j = 1; j < k; ++j )
        {
            const std::size_t bi   = support[j];
            const auto        beta = unflatIndex< M >( bi );
            MultiIndex< M >   gamma{};
            bool valid = true;
            for ( int i = 0; i < M; ++i )
            {
                if ( beta[std::size_t( i )] > alpha[std::size_t( i )] )
                {
                    valid = false;
                    break;
                }
                gamma[std::size_t( i )] = alpha[std::size_t( i )] - beta[std::size_t( i )];
            }
            if ( !valid ) continue;
            const std::size_t gi = flatIndex< M >( gamma );
            acc -= scratch[bi] * scratch[gi];
        }
        scratch[ai] = acc * inv2sqrt;
    }

    auto& ri = out.rawIndices();
    auto& rv = out.rawValues();
    for ( storage::flat_index_t k : support )
    {
        if ( scratch[k] != T{ 0 } )
        {
            ri.push_back( k );
            rv.push_back( scratch[k] );
        }
    }
}

/// Sparse integer power `out = f^n` via binary exponentiation. Negative exponents throw std::invalid_argument.
template < typename T, int N, int M >
void seriesPowIntSparse( storage::SparseContainer< T, N, M >&       out,
                         const storage::SparseContainer< T, N, M >& f,
                         int                                         n )
{
    if ( n < 0 )
        throw std::invalid_argument(
            "seriesPowIntSparse: negative exponent not supported; "
            "use reciprocal then pow for n<0" );

    using Container = storage::SparseContainer< T, N, M >;

    if ( n == 0 )
    {
        out.set( 0, T{ 1 } );
        return;
    }
    if ( n == 1 )
    {
        f.forEachNonzero( [&out]( std::size_t k, T v ) { out.set( k, v ); } );
        return;
    }

    // Binary exponentiation using bare containers.
    Container base_c;
    f.forEachNonzero( [&base_c]( std::size_t k, T v ) { base_c.set( k, v ); } );

    Container result_c;
    bool      result_set = false;

    while ( n > 0 )
    {
        if ( n & 1 )
        {
            if ( !result_set )
            {
                base_c.forEachNonzero(
                    [&result_c]( std::size_t k, T v ) { result_c.set( k, v ); } );
                result_set = true;
            }
            else
            {
                Container tmp;
                sparseCauchyProduct< T, N, M >( tmp, result_c, base_c );
                result_c = std::move( tmp );
            }
        }
        n >>= 1;
        if ( n > 0 )
        {
            Container sq;
            sparseCauchySelfProduct< T, N, M >( sq, base_c );
            base_c = std::move( sq );
        }
    }

    result_c.forEachNonzero( [&out]( std::size_t k, T v ) { out.set( k, v ); } );
}

}  // namespace tax::detail::kernels
