// SPDX-License-Identifier: BSD-3-Clause
//
// Slice-aware Cauchy convolution.
//
// All buffered nodes ultimately reduce their per-degree update to one
// primitive: given two operand slices a (degree eA) and b (degree eB),
// accumulate scale * a[beta] * b[gamma] into out[beta + gamma] for every
// (beta, gamma) with the prescribed degrees.  Slices are passed as
// Eigen-compatible views (a `coeff(i)` and `coeffRef(i)` interface),
// which lets buffered nodes mix dense materialised inputs (other
// buffered nodes, leaf storage) with lazy view-like inputs (Add, Sub,
// scalar-affine).
//
// Helpers also live here for "out += a * a" (self) and "out += scale * (a * b)".

#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <span>

#include "tax/util/multi_index.hpp"

namespace tax::kernels
{

// out[α-rank] += scale * a[β-rank] * b[γ-rank] for all (β, γ) with
// |β| = eA, |γ| = eB, α = β + γ.  Indices are within-degree ranks (the
// rank of the multi-index inside its own degree slice), so a, b, out are
// slices not full buffers.
//
// `a` and `b` only need `.coeff(i)`; `out` needs `.coeffRef(i)`.
template < class T, class A, class B, class Out >
inline void cauchyAccumulateSlice( std::size_t eA, std::size_t eB, std::size_t nvars,
                                   T scale, const A& a, const B& b, Out& out )
{
    using namespace tax::util;
    constexpr std::size_t kStackBuf = 32;
    std::size_t alpha_static[ kStackBuf ];
    std::size_t* alpha_buf = alpha_static;
    std::size_t* heap = nullptr;
    if ( nvars > kStackBuf )
    {
        heap = new std::size_t[ nvars ];
        alpha_buf = heap;
    }

    forEachMultiIndexOfDegree( eA, nvars, [ & ]( std::span< const std::size_t > beta ) {
        const std::size_t bi = flatIndexWithinDegree( beta );
        const T av =
            static_cast< T >( a.coeff( static_cast< Eigen::Index >( bi ) ) ) * scale;
        if ( av == T{} )
        {
            return;
        }
        forEachMultiIndexOfDegree( eB, nvars, [ & ]( std::span< const std::size_t > gamma ) {
            for ( std::size_t k = 0; k < nvars; ++k )
            {
                alpha_buf[ k ] = beta[ k ] + gamma[ k ];
            }
            const std::size_t ai =
                flatIndexWithinDegree( std::span< const std::size_t >( alpha_buf, nvars ) );
            const std::size_t gi = flatIndexWithinDegree( gamma );
            out.coeffRef( static_cast< Eigen::Index >( ai ) ) +=
                av * static_cast< T >( b.coeff( static_cast< Eigen::Index >( gi ) ) );
        } );
    } );

    if ( heap != nullptr )
    {
        delete[] heap;
    }
}

// out_d = sum_{e=0}^{d} A_e * B_{d-e}.  Caller supplies operand expressions
// `a, b` providing `slice(e)` returning a coeff-readable view, and the
// destination's degree-d slice `out_d` (writable).
template < class T, class A, class B, class OutSlice >
inline void cauchyMulComputeDegree( std::size_t d, std::size_t nvars,
                                    const A& a, const B& b, OutSlice& out_d )
{
    out_d.setZero();
    for ( std::size_t e = 0; e <= d; ++e )
    {
        auto a_e = a.slice( e );
        auto b_de = b.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, T{ 1 }, a_e, b_de, out_d );
    }
}

// out_d = (a^2)_d.
template < class T, class A, class OutSlice >
inline void squareComputeDegree( std::size_t d, std::size_t nvars, const A& a,
                                 OutSlice& out_d )
{
    out_d.setZero();
    for ( std::size_t e = 0; e <= d; ++e )
    {
        auto a_e = a.slice( e );
        auto a_de = a.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, T{ 1 }, a_e, a_de, out_d );
    }
}

// out = a / b, evaluated degree-by-degree via b * out = a:
//   out_d = (a_d - sum_{e=1}^{d} b_e * out_{d-e}) / b_0
// `out` is a slice-providing object (slice(e) returning a writable view
// for e <= d) so the recurrence can read its own lower-degree slices.
template < class T, class A, class B, class OutObj >
inline void divComputeDegree( std::size_t d, std::size_t nvars, const A& a, const B& b,
                              const OutObj& out )
{
    using namespace tax::util;
    auto out_d = out.slice( d );
    auto a_d = a.slice( d );
    const std::size_t dsize = degreeSize( d, nvars );

    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) =
            static_cast< T >( a_d.coeff( static_cast< Eigen::Index >( i ) ) );
    }

    for ( std::size_t e = 1; e <= d; ++e )
    {
        auto b_e = b.slice( e );
        auto out_de = out.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, T{ -1 }, b_e, out_de, out_d );
    }

    const T b0 = static_cast< T >( b.slice( 0 ).coeff( 0 ) );
    const T inv_b0 = T{ 1 } / b0;
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) *= inv_b0;
    }
}

// out = 1/b.
template < class T, class B, class OutObj >
inline void reciprocalComputeDegree( std::size_t d, std::size_t nvars, const B& b,
                                     const OutObj& out )
{
    using namespace tax::util;
    if ( d == 0 )
    {
        out.slice( 0 ).coeffRef( 0 ) =
            T{ 1 } / static_cast< T >( b.slice( 0 ).coeff( 0 ) );
        return;
    }
    auto out_d = out.slice( d );
    out_d.setZero();
    for ( std::size_t e = 1; e <= d; ++e )
    {
        auto b_e = b.slice( e );
        auto out_de = out.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, T{ -1 }, b_e, out_de, out_d );
    }
    const T b0 = static_cast< T >( b.slice( 0 ).coeff( 0 ) );
    const T inv_b0 = T{ 1 } / b0;
    const std::size_t dsize = degreeSize( d, nvars );
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) *= inv_b0;
    }
}

}  // namespace tax::kernels
