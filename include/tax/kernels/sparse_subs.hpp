#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tax/core/multi_index.hpp>
#include <tax/core/storage/sparse.hpp>
#include <tax/kernels/sparse_cauchy.hpp>
#include <vector>

namespace tax::detail::kernels
{

/// Additive closure of seed multi-indices under multi-index addition, truncated to total degree
/// `N`. Always includes flat index `0`.
template < int N, int M >
[[nodiscard]] inline std::vector< storage::flat_index_t > additiveClosure(
    const std::vector< storage::flat_index_t >& seeds )
{
    constexpr std::size_t NC = numMonomials( N, M );
    std::vector< bool > seen( NC, false );
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

/// Sparse reciprocal `out = 1 / f` via forward substitution. Throws std::domain_error if the
/// constant term is zero.
template < typename T, int N, int M >
void seriesReciprocalSparse( storage::SparseContainer< T, N, M >& out,
                             const storage::SparseContainer< T, N, M >& f )
{
    constexpr std::size_t NC = numMonomials( N, M );

    const auto fi = f.support();
    const auto fv = f.values();
    if ( fi.empty() || fi.front() != 0 || fv.front() == T{ 0 } )
        throw std::domain_error( "seriesReciprocalSparse: constant term must be nonzero" );

    const T inv_f0 = T{ 1 } / fv.front();

    // Seeds: f's nonzero perturbation indices (flat > 0).
    std::vector< storage::flat_index_t > seeds;
    seeds.reserve( fi.size() );
    for ( std::size_t k = 1; k < fi.size(); ++k ) seeds.push_back( fi[k] );

    const auto support = additiveClosure< N, M >( seeds );

    // Dense scratch for O(1) lookup of out[gamma] in the recurrence.
    std::vector< T > scratch( NC, T{ 0 } );
    scratch[0] = inv_f0;

    // For each α in support[1..end]:
    //   out[α] = -inv_f0 * sum_{β in f, β>0, β<=α} f[β] * out[α-β]
    for ( std::size_t k = 1; k < support.size(); ++k )
    {
        const std::size_t ai = support[k];
        const auto alpha = unflatIndex< M >( ai );
        T acc{ 0 };
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

/// Sparse square root `out = sqrt(f)` via forward substitution. Throws std::domain_error if the
/// constant term is <= 0.
template < typename T, int N, int M >
void seriesSqrtSparse( storage::SparseContainer< T, N, M >& out,
                       const storage::SparseContainer< T, N, M >& f )
{
    constexpr std::size_t NC = numMonomials( N, M );

    const auto fi = f.support();
    const auto fv = f.values();
    const T f0 = ( !fi.empty() && fi.front() == 0 ) ? fv.front() : T{ 0 };
    if ( !( f0 > T{ 0 } ) )
        throw std::domain_error( "seriesSqrtSparse: constant term must be strictly positive" );

    const T sqrt_f0 = std::sqrt( f0 );
    const T inv2sqrt = T{ 1 } / ( T{ 2 } * sqrt_f0 );

    std::vector< storage::flat_index_t > seeds;
    seeds.reserve( fi.size() );
    for ( std::size_t k = 1; k < fi.size(); ++k ) seeds.push_back( fi[k] );

    const auto support = additiveClosure< N, M >( seeds );

    // Dense scratch for f and out (for O(1) inner-loop lookup).
    std::vector< T > f_dense( NC, T{ 0 } );
    for ( std::size_t k = 0; k < fi.size(); ++k ) f_dense[fi[k]] = fv[k];

    std::vector< T > scratch( NC, T{ 0 } );
    scratch[0] = sqrt_f0;

    // For each α in support[1..end]:
    //   out[α] = (f[α] - sum_{β+γ=α, β,γ>0, both in support} out[β]*out[γ]) / (2*out[0])
    for ( std::size_t k = 1; k < support.size(); ++k )
    {
        const std::size_t ai = support[k];
        const auto alpha = unflatIndex< M >( ai );
        T acc = f_dense[ai];
        for ( std::size_t j = 1; j < k; ++j )
        {
            const std::size_t bi = support[j];
            const auto beta = unflatIndex< M >( bi );
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

/// Perturbation seeds (flat index > 0) of a sparse container.
template < typename T, int N, int M >
[[nodiscard]] inline std::vector< storage::flat_index_t > perturbationSeeds(
    const storage::SparseContainer< T, N, M >& f )
{
    const auto fi = f.support();
    std::vector< storage::flat_index_t > seeds;
    seeds.reserve( fi.size() );
    for ( const storage::flat_index_t k : fi )
        if ( k != 0 ) seeds.push_back( k );
    return seeds;
}

/// Scatter a sparse container into a dense scratch vector (`dense[k] = f[k]`).
template < typename T, int N, int M >
inline void scatterDense( std::vector< T >& dense, const storage::SparseContainer< T, N, M >& f )
{
    std::fill( dense.begin(), dense.end(), T{ 0 } );
    f.forEachNonzero( [&dense]( std::size_t k, T v ) { dense[k] = v; } );
}

/// If `beta` (= unflat `bflat`) is componentwise <= `alpha`, set `gflat` to
/// flatIndex(alpha - beta) and `bdeg` to |beta|, and return true.
template < int M >
[[nodiscard]] inline bool trySubMulti( const MultiIndex< M >& alpha, storage::flat_index_t bflat,
                                       std::size_t& gflat, int& bdeg )
{
    const auto beta = unflatIndex< M >( std::size_t( bflat ) );
    MultiIndex< M > g{};
    bdeg = 0;
    for ( int i = 0; i < M; ++i )
    {
        const int bi = beta[std::size_t( i )];
        if ( bi > alpha[std::size_t( i )] ) return false;
        g[std::size_t( i )] = alpha[std::size_t( i )] - bi;
        bdeg += bi;
    }
    gflat = flatIndex< M >( g );
    return true;
}

/// Emit the nonzeros of `dense` at the (ascending) `support` indices into `out`.
template < typename T, int N, int M >
inline void emitSupport( storage::SparseContainer< T, N, M >& out,
                         const std::vector< storage::flat_index_t >& support,
                         const std::vector< T >& dense )
{
    auto& ri = out.rawIndices();
    auto& rv = out.rawValues();
    for ( const storage::flat_index_t k : support )
    {
        if ( dense[k] != T{ 0 } )
        {
            ri.push_back( k );
            rv.push_back( dense[k] );
        }
    }
}

/// Sparse exponential `out = exp(f)` via the product recurrence `out' = f' * out`
/// (the multiplier aliases the output, so no auxiliary series is needed).
template < typename T, int N, int M >
void seriesExpSparse( storage::SparseContainer< T, N, M >& out,
                      const storage::SparseContainer< T, N, M >& f )
{
    using std::exp;
    constexpr std::size_t NC = numMonomials( N, M );

    const auto fi = f.support();
    const auto fv = f.values();
    const T f0 = ( !fi.empty() && fi.front() == 0 ) ? fv.front() : T{ 0 };

    const auto support = additiveClosure< N, M >( perturbationSeeds( f ) );

    std::vector< T > od( NC, T{ 0 } );
    od[0] = exp( f0 );

    for ( std::size_t s = 1; s < support.size(); ++s )
    {
        const std::size_t k = support[s];
        const auto alpha = unflatIndex< M >( k );
        int d = 0;
        for ( int i = 0; i < M; ++i ) d += alpha[std::size_t( i )];
        T rhs{ 0 };
        for ( std::size_t j = 0; j < fi.size(); ++j )
        {
            if ( fi[j] == 0 ) continue;
            std::size_t gflat;
            int bdeg;
            if ( !trySubMulti< M >( alpha, fi[j], gflat, bdeg ) ) continue;
            rhs += T( bdeg ) * fv[j] * od[gflat];
        }
        od[k] = rhs / T( d );
    }

    emitSupport< T, N, M >( out, support, od );
}

/// Sparse product-recurrence driver `out' = src' * h`, seeded `out[0] = out0`
/// (for erf; `h` is a distinct auxiliary series). Result support is the
/// additive closure of the operands' perturbations.
template < typename T, int N, int M >
void seriesDerivProductSparse( storage::SparseContainer< T, N, M >& out, T out0,
                               const storage::SparseContainer< T, N, M >& src,
                               const storage::SparseContainer< T, N, M >& h )
{
    constexpr std::size_t NC = numMonomials( N, M );

    const auto si = src.support();
    const auto sv = src.values();

    std::vector< storage::flat_index_t > seeds = perturbationSeeds( src );
    for ( const storage::flat_index_t k : perturbationSeeds( h ) ) seeds.push_back( k );
    const auto support = additiveClosure< N, M >( seeds );

    std::vector< T > hd( NC, T{ 0 } );
    scatterDense< T, N, M >( hd, h );

    std::vector< T > od( NC, T{ 0 } );
    od[0] = out0;

    for ( std::size_t s = 1; s < support.size(); ++s )
    {
        const std::size_t k = support[s];
        const auto alpha = unflatIndex< M >( k );
        int d = 0;
        for ( int i = 0; i < M; ++i ) d += alpha[std::size_t( i )];
        T rhs{ 0 };
        for ( std::size_t j = 0; j < si.size(); ++j )
        {
            if ( si[j] == 0 ) continue;
            std::size_t gflat;
            int bdeg;
            if ( !trySubMulti< M >( alpha, si[j], gflat, bdeg ) ) continue;
            rhs += T( bdeg ) * sv[j] * hd[gflat];
        }
        od[k] = rhs / T( d );
    }

    emitSupport< T, N, M >( out, support, od );
}

/// Sparse quotient-recurrence driver `h * out' = Sign * src'`, seeded
/// `out[0] = out0`. Requires `h[0] != 0`. Covers log, asin/acos/atan,
/// asinh/acosh/atanh and atan2.
template < int Sign, typename T, int N, int M >
void seriesDerivQuotientSparse( storage::SparseContainer< T, N, M >& out, T out0,
                                const storage::SparseContainer< T, N, M >& src,
                                const storage::SparseContainer< T, N, M >& h )
{
    static_assert( Sign == 1 || Sign == -1 );
    constexpr std::size_t NC = numMonomials( N, M );

    const auto hi = h.support();
    const auto hv = h.values();
    const T h0 = ( !hi.empty() && hi.front() == 0 ) ? hv.front() : T{ 0 };
    const T inv_h0 = T{ 1 } / h0;

    std::vector< storage::flat_index_t > seeds = perturbationSeeds( src );
    for ( const storage::flat_index_t k : perturbationSeeds( h ) ) seeds.push_back( k );
    const auto support = additiveClosure< N, M >( seeds );

    std::vector< T > sd( NC, T{ 0 } );
    scatterDense< T, N, M >( sd, src );

    std::vector< T > od( NC, T{ 0 } );
    od[0] = out0;

    for ( std::size_t s = 1; s < support.size(); ++s )
    {
        const std::size_t k = support[s];
        const auto alpha = unflatIndex< M >( k );
        int d = 0;
        for ( int i = 0; i < M; ++i ) d += alpha[std::size_t( i )];
        T rhs{ 0 };
        for ( std::size_t j = 0; j < hi.size(); ++j )
        {
            if ( hi[j] == 0 ) continue;
            std::size_t gflat;
            int bdeg;
            if ( !trySubMulti< M >( alpha, hi[j], gflat, bdeg ) ) continue;
            rhs += T( d - bdeg ) * hv[j] * od[gflat];
        }
        const T s_src = Sign > 0 ? sd[k] : -sd[k];
        od[k] = ( s_src - rhs / T( d ) ) * inv_h0;
    }

    emitSupport< T, N, M >( out, support, od );
}

/// Sparse real-power recurrence `a * out' = c * a' * out`, seeded `out[0] = out0`
/// (the intended real branch of `a[0]^c`). Requires `a[0] != 0`. Drives cbrt and
/// the real / rational power surface.
template < typename T, int N, int M >
void seriesPowFromSeedSparse( storage::SparseContainer< T, N, M >& out, T out0,
                              const storage::SparseContainer< T, N, M >& a, T c )
{
    constexpr std::size_t NC = numMonomials( N, M );

    const auto ai = a.support();
    const auto av = a.values();
    const T a0 = ( !ai.empty() && ai.front() == 0 ) ? av.front() : T{ 0 };
    const T inv_a0 = T{ 1 } / a0;

    const auto support = additiveClosure< N, M >( perturbationSeeds( a ) );

    std::vector< T > od( NC, T{ 0 } );
    od[0] = out0;

    for ( std::size_t s = 1; s < support.size(); ++s )
    {
        const std::size_t k = support[s];
        const auto alpha = unflatIndex< M >( k );
        int d = 0;
        for ( int i = 0; i < M; ++i ) d += alpha[std::size_t( i )];
        T rhs{ 0 };
        for ( std::size_t j = 0; j < ai.size(); ++j )
        {
            if ( ai[j] == 0 ) continue;
            std::size_t gflat;
            int bdeg;
            if ( !trySubMulti< M >( alpha, ai[j], gflat, bdeg ) ) continue;
            rhs += ( c * T( bdeg ) - T( d - bdeg ) ) * av[j] * od[gflat];
        }
        od[k] = rhs * inv_a0 / T( d );
    }

    emitSupport< T, N, M >( out, support, od );
}

/// Sparse coupled sine/cosine `s = sin(f)`, `c = cos(f)` in one pass.
template < typename T, int N, int M >
void seriesSinCosSparse( storage::SparseContainer< T, N, M >& s_out,
                         storage::SparseContainer< T, N, M >& c_out,
                         const storage::SparseContainer< T, N, M >& f )
{
    using std::cos;
    using std::sin;
    constexpr std::size_t NC = numMonomials( N, M );

    const auto fi = f.support();
    const auto fv = f.values();
    const T f0 = ( !fi.empty() && fi.front() == 0 ) ? fv.front() : T{ 0 };

    const auto support = additiveClosure< N, M >( perturbationSeeds( f ) );

    std::vector< T > sd( NC, T{ 0 } ), cd( NC, T{ 0 } );
    sd[0] = sin( f0 );
    cd[0] = cos( f0 );

    for ( std::size_t idx = 1; idx < support.size(); ++idx )
    {
        const std::size_t k = support[idx];
        const auto alpha = unflatIndex< M >( k );
        int d = 0;
        for ( int i = 0; i < M; ++i ) d += alpha[std::size_t( i )];
        T sr{ 0 }, cr{ 0 };
        for ( std::size_t j = 0; j < fi.size(); ++j )
        {
            if ( fi[j] == 0 ) continue;
            std::size_t gflat;
            int bdeg;
            if ( !trySubMulti< M >( alpha, fi[j], gflat, bdeg ) ) continue;
            const T w = T( bdeg ) * fv[j];
            sr += w * cd[gflat];
            cr += w * sd[gflat];
        }
        sd[k] = sr / T( d );
        cd[k] = -cr / T( d );
    }

    emitSupport< T, N, M >( s_out, support, sd );
    emitSupport< T, N, M >( c_out, support, cd );
}

/// Sparse integer power `out = f^n` via binary exponentiation. Negative exponents throw
/// std::invalid_argument.
template < typename T, int N, int M >
void seriesPowIntSparse( storage::SparseContainer< T, N, M >& out,
                         const storage::SparseContainer< T, N, M >& f, int n )
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
    bool result_set = false;

    while ( n > 0 )
    {
        if ( n & 1 )
        {
            if ( !result_set )
            {
                base_c.forEachNonzero(
                    [&result_c]( std::size_t k, T v ) { result_c.set( k, v ); } );
                result_set = true;
            } else
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
