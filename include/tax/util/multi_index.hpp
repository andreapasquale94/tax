// SPDX-License-Identifier: BSD-3-Clause
//
// Multi-index utilities and packed-storage layout.
//
// All TTE coefficients live in a single flat buffer (Eigen::Matrix in the
// static path, Eigen::VectorX in the dynamic path) ordered graded
// reverse-lexicographic:
//
//   - lower total degree comes first, and
//   - within a degree, indices are sorted by reverse-lex (the rightmost
//     non-zero entry is the major axis).
//
// Concretely for M = 2, N = 2 the order is:
//   (0,0)            <- degree 0, offset 0
//   (1,0), (0,1)     <- degree 1, offset 1
//   (2,0), (1,1), (0,2)  <- degree 2, offset 3
//
// The functions below take std::span<const std::size_t> for the multi-index
// so that the same code services both the static (std::array) and dynamic
// (std::vector) callers.

#pragma once

#include <cstddef>
#include <span>

#include "tax/util/binomial.hpp"

namespace tax::util
{

// Number of monomials in M variables with total degree <= order.
[[nodiscard]] constexpr std::size_t monomialCount( std::size_t order, std::size_t nvars ) noexcept
{
    return binom( order + nvars, nvars );
}

// Number of monomials in M variables with total degree exactly d.
[[nodiscard]] constexpr std::size_t degreeSize( std::size_t d, std::size_t nvars ) noexcept
{
    if ( nvars == 0 )
    {
        return d == 0 ? 1 : 0;
    }
    return binom( d + nvars - 1, nvars - 1 );
}

// Flat offset where coefficients of total degree d begin.
[[nodiscard]] constexpr std::size_t degreeOffset( std::size_t d, std::size_t nvars ) noexcept
{
    if ( d == 0 )
    {
        return 0;
    }
    // sum_{e=0}^{d-1} C(e + M - 1, M - 1) = C(d + M - 1, M)
    if ( nvars == 0 )
    {
        return 1;
    }
    return binom( d + nvars - 1, nvars );
}

// Compute |alpha| (sum of components).
[[nodiscard]] constexpr std::size_t totalDegree( std::span< const std::size_t > alpha ) noexcept
{
    std::size_t s = 0;
    for ( auto a : alpha )
    {
        s += a;
    }
    return s;
}

// Multi-index factorial: product of alpha_i!.
[[nodiscard]] constexpr std::size_t factorial( std::span< const std::size_t > alpha ) noexcept
{
    std::size_t out = 1;
    for ( auto a : alpha )
    {
        std::size_t f = 1;
        for ( std::size_t i = 2; i <= a; ++i )
        {
            f *= i;
        }
        out *= f;
    }
    return out;
}

// Graded reverse-lex flat index of alpha.  Within a degree d, indices are
// ranked so that (d, 0, ..., 0) is first and (0, ..., 0, d) is last.
//
// The position within a degree is computed by walking from the last
// component back to the first; for a multi-index (alpha_0, ..., alpha_{M-1})
// of total degree d the rank within the degree-d block is
//
//   rank = sum_{i = M-1 down to 1} C(r_i + i - 1, i)
//
// where r_i is the residual sum_{j=0}^{i} alpha_j after stripping
// alpha_{i+1}..alpha_{M-1}.
[[nodiscard]] constexpr std::size_t flatIndex( std::span< const std::size_t > alpha ) noexcept
{
    const std::size_t M = alpha.size();
    const std::size_t d = totalDegree( alpha );
    std::size_t rank = 0;
    std::size_t residual = d;
    for ( std::size_t i = M; i-- > 1; )
    {
        // alpha_i contribution: skip multi-indices where the i-th component
        // exceeds alpha[i] (i.e. residual is smaller).
        const std::size_t a = alpha[ i ];
        // Sum_{k=0}^{a-1} C(residual - k + (i - 1), i - 1) gives # of
        // (alpha_0..alpha_{i-1}) tuples summing to (residual - k); that is
        // the count of multi-indices preceding ours where alpha[i] = k.
        for ( std::size_t k = 0; k < a; ++k )
        {
            rank += binom( ( residual - k ) + ( i - 1 ), i - 1 );
        }
        residual -= a;
    }
    return degreeOffset( d, M ) + rank;
}

// Rank of alpha within its own degree-d block.  Equivalent to
// `flatIndex(alpha) - degreeOffset(|alpha|, M)` but computed without the
// outer offset addition.
[[nodiscard]] constexpr std::size_t
flatIndexWithinDegree( std::span< const std::size_t > alpha ) noexcept
{
    const std::size_t M = alpha.size();
    const std::size_t d = totalDegree( alpha );
    std::size_t rank = 0;
    std::size_t residual = d;
    for ( std::size_t i = M; i-- > 1; )
    {
        const std::size_t a = alpha[ i ];
        for ( std::size_t k = 0; k < a; ++k )
        {
            rank += binom( ( residual - k ) + ( i - 1 ), i - 1 );
        }
        residual -= a;
    }
    return rank;
}

// Inverse of flatIndex: recover the multi-index of length M corresponding
// to flat index `idx`.  Writes into `out` which must have size M.
constexpr void unflatIndex( std::size_t idx, std::span< std::size_t > out ) noexcept
{
    const std::size_t M = out.size();
    // Locate degree d.
    std::size_t d = 0;
    while ( idx >= degreeSize( d, M ) )
    {
        idx -= degreeSize( d, M );
        ++d;
    }
    std::size_t residual = d;
    for ( std::size_t i = M; i-- > 1; )
    {
        std::size_t a = 0;
        while ( true )
        {
            const std::size_t block = binom( ( residual - a ) + ( i - 1 ), i - 1 );
            if ( idx < block )
            {
                break;
            }
            idx -= block;
            ++a;
        }
        out[ i ] = a;
        residual -= a;
    }
    if ( M > 0 )
    {
        out[ 0 ] = residual;
    }
}

// Iterate every multi-index of given degree, calling `f(span<const size_t>)`.
//
// Order matches flatIndex / unflatIndex above (graded reverse-lex).  At each
// step we identify the smallest i >= 1 where positions 0..i-1 collectively
// hold at least 1 unit of "budget", increment alpha[i] by 1, drain the rest
// of that budget back into alpha[0], and zero alpha[1..i-1].
template < class F >
inline void forEachMultiIndexOfDegree( std::size_t degree, std::size_t nvars, F&& f )
{
    if ( nvars == 0 )
    {
        if ( degree == 0 )
        {
            std::span< const std::size_t > empty{};
            f( empty );
        }
        return;
    }

    constexpr std::size_t kStackBuf = 32;
    std::size_t buf_static[ kStackBuf ];
    std::size_t* buf = buf_static;
    std::size_t* heap = nullptr;
    if ( nvars > kStackBuf )
    {
        heap = new std::size_t[ nvars ];
        buf = heap;
    }
    for ( std::size_t i = 0; i < nvars; ++i )
    {
        buf[ i ] = 0;
    }
    buf[ 0 ] = degree;

    while ( true )
    {
        f( std::span< const std::size_t >( buf, nvars ) );
        std::size_t accum = 0;
        std::size_t i = 0;
        while ( i < nvars )
        {
            if ( i >= 1 && accum >= 1 )
            {
                break;
            }
            accum += buf[ i ];
            ++i;
        }
        if ( i >= nvars )
        {
            break;
        }
        buf[ i ] += 1;
        buf[ 0 ] = accum - 1;
        for ( std::size_t j = 1; j < i; ++j )
        {
            buf[ j ] = 0;
        }
    }

    if ( heap != nullptr )
    {
        delete[] heap;
    }
}

}  // namespace tax::util
