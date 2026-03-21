#pragma once

#include <cmath>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/ops.hpp>

namespace tax::detail
{

template < typename T, int N, int M >
/**
 * @brief Reciprocal series solve `a * out = 1`.
 * @details Requires `a[0] != 0`.
 */
constexpr void seriesReciprocal( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
{
    out = {};
    const T inv_a0 = T{ 1 } / a[0];

    if constexpr ( M == 1 )
    {
        out[0] = inv_a0;
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k <= d; ++k ) rhs -= a[k] * out[d - k];
            out[d] = rhs * inv_a0;
        }
    } else
    {
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = ( d == 0 ) ? T{ 1 } : T{ 0 };
                forEachSubIndex< M >( alpha, 1, d,
                                      [&]( auto bi, auto gi, int ) { rhs -= a[bi] * out[gi]; } );
                out[ai] = rhs * inv_a0;
            } );
        }
    }
}

template < typename T, int N, int M >
/// @brief Square series `out = a^2`.
constexpr void seriesSquare( std::array< T, numMonomials( N, M ) >& out,
                             const std::array< T, numMonomials( N, M ) >& a ) noexcept
{
    cauchySelfProduct< T, N, M >( out, a );
}

template < typename T, int N, int M >
/// @brief Cube series `out = a^3`.
constexpr void seriesCube( std::array< T, numMonomials( N, M ) >& out,
                           const std::array< T, numMonomials( N, M ) >& a ) noexcept
{
    // Two Cauchy products: tmp = a^2 (via symmetric self-product), out = tmp * a.
    // O(N^2) for M=1 (vs the previous O(N^3) triple loop), O(S^2) for M>1.
    constexpr auto S = numMonomials( N, M );
    std::array< T, S > tmp{};
    cauchySelfProduct< T, N, M >( tmp, a );
    cauchyProduct< T, N, M >( out, tmp, a );
}

template < typename T, int N, int M >
/**
 * @brief Square-root series solve `out * out = a`.
 * @details Uses the principal branch from `sqrt(a[0])`.
 */
constexpr void seriesSqrt( std::array< T, numMonomials( N, M ) >& out,
                           const std::array< T, numMonomials( N, M ) >& a ) noexcept
{
    using std::sqrt;
    out = {};
    out[0] = sqrt( a[0] );
    const T inv2g0 = T{ 1 } / ( T{ 2 } * out[0] );

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            // Exploit symmetry: out[k]*out[d-k] == out[d-k]*out[k].
            T rhs = a[d];
            for ( int k = 1; k + k < d; ++k ) rhs -= T{ 2 } * out[k] * out[d - k];
            if ( d % 2 == 0 ) rhs -= out[d / 2] * out[d / 2];
            out[d] = rhs * inv2g0;
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = a[ai];
                forEachSubIndex< M >( alpha, 1, d - 1, [&]( auto bi, auto gi, int ) {
                    if ( bi < gi )
                        rhs -= T{ 2 } * out[bi] * out[gi];
                    else if ( bi == gi )
                        rhs -= out[bi] * out[bi];
                } );
                out[ai] = rhs * inv2g0;
            } );
        }
    }
}

template < typename T, int N, int M >
/**
 * @brief Cubic-root series solve `out * out * out = a`.
 * @details Uses the real branch from `cbrt(a[0])`. Requires `a[0] != 0`.
 *
 *          Maintains `sq = out^2` incrementally so each degree needs only O(d) work
 *          for M=1 (O(N^2) total vs the naive O(N^3)), and one Cauchy-product's worth
 *          of work across all degrees for M>1 (O(S^2) vs O(N·S^2)).
 *
 *          Recurrence at multi-index alpha with |alpha|=d (out[alpha]=0 initially):
 *            (out^3)_alpha = 3*g0^2*out_alpha
 *                          + out[0] * sq_alpha^*
 *                          + sum_{0<|beta|<d} out[beta] * sq[alpha-beta]
 *          where sq_alpha^* = sum_{0<|beta|<d} out[beta]*out[alpha-beta].
 *          Combining: rhs -= out[beta] * (out[0]*out[gamma] + sq[gamma])
 *          for each pair (beta,gamma) via forEachSubIndex(alpha, 1, d-1).
 */
constexpr void seriesCbrt( std::array< T, numMonomials( N, M ) >& out,
                           const std::array< T, numMonomials( N, M ) >& a ) noexcept
{
    using std::cbrt;
    constexpr auto S = numMonomials( N, M );

    out = {};
    out[0] = cbrt( a[0] );
    const T inv3g0sq = T{ 1 } / ( T{ 3 } * out[0] * out[0] );

    if constexpr ( M == 1 )
    {
        // Maintain sq[j] = (out^2)_j incrementally.
        std::array< T, S > sq{};
        sq[0] = out[0] * out[0];
        for ( int d = 1; d <= N; ++d )
        {
            // sq_d_partial = sum_{k=1}^{d-1} out[k]*out[d-k]  (symmetric: half iterations)
            T sq_d_partial = T{ 0 };
            for ( int k = 1; k + k < d; ++k ) sq_d_partial += T{ 2 } * out[k] * out[d - k];
            if ( d % 2 == 0 ) sq_d_partial += out[d / 2] * out[d / 2];

            // rhs = (out^3)_d with out[d]=0:
            //   out[0]*sq_d_partial  +  sum_{j=1}^{d-1} out[j]*sq[d-j]
            T rhs = out[0] * sq_d_partial;
            for ( int j = 1; j < d; ++j ) rhs += out[j] * sq[d - j];

            out[d] = ( a[d] - rhs ) * inv3g0sq;

            // Finalize sq[d] = 2*out[0]*out[d] + sq_d_partial
            sq[d] = T{ 2 } * out[0] * out[d] + sq_d_partial;
        }
    } else
    {
        // Maintain sq = out^2 incrementally (symmetric self-product for sq update).
        std::array< T, S > sq{};
        sq[0] = out[0] * out[0];
        for ( int d = 1; d <= N; ++d )
        {
            // Compute out for degree d using sq for degrees 0..d-1.
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = a[ai];
                // Subtract out[beta] * (out[0]*out[gamma] + sq[gamma])
                // for each pair (beta, gamma=alpha-beta) with 0 < |beta| < d.
                forEachSubIndex< M >( alpha, 1, d - 1, [&]( auto bi, auto gi, int ) {
                    rhs -= out[bi] * ( out[0] * out[gi] + sq[gi] );
                } );
                out[ai] = rhs * inv3g0sq;
            } );

            // Update sq for degree d using fully-updated out (symmetric enumeration).
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T val = T{ 0 };
                forEachSubIndex< M >( alpha, [&]( auto bi, auto gi ) {
                    if ( bi < gi )
                        val += T{ 2 } * out[bi] * out[gi];
                    else if ( bi == gi )
                        val += out[bi] * out[bi];
                } );
                sq[ai] = val;
            } );
        }
    }
}

}  // namespace tax::detail
