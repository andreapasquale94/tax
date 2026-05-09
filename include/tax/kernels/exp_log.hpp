// SPDX-License-Identifier: BSD-3-Clause
//
// Exponential and logarithm recurrences over slice-providing operands.

#pragma once

#include <cmath>
#include <cstddef>

#include "tax/kernels/cauchy.hpp"
#include "tax/util/multi_index.hpp"

namespace tax::kernels
{

// out = exp(u).  Recurrence:
//   out_alpha = (1/d) sum_{e=1}^{d} e * (u_gamma * out_beta) for |gamma|=e, |beta|=d-e.
template < class T, class U, class OutObj >
inline void expComputeDegree( std::size_t d, std::size_t nvars, const U& u, const OutObj& out )
{
    if ( d == 0 )
    {
        out.degreeSlice( 0 ).coeffRef( 0 ) =
            std::exp( static_cast< T >( u.degreeSlice( 0 ).coeff( 0 ) ) );
        return;
    }
    auto out_d = out.degreeSlice( d );
    out_d.setZero();
    const T inv_d = T{ 1 } / static_cast< T >( d );
    for ( std::size_t e = 1; e <= d; ++e )
    {
        const T scale = static_cast< T >( e ) * inv_d;
        auto u_e = u.degreeSlice( e );
        auto out_de = out.degreeSlice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, scale, u_e, out_de, out_d );
    }
}

// out = log(u).  Recurrence (from u * E[F] = E[u]):
//   out_alpha = u_alpha / u_0
//             - (1/(d u_0)) sum_{e=1}^{d-1} (d - e) * (u_beta * out_gamma)
template < class T, class U, class OutObj >
inline void logComputeDegree( std::size_t d, std::size_t nvars, const U& u, const OutObj& out )
{
    using namespace tax::util;
    if ( d == 0 )
    {
        out.degreeSlice( 0 ).coeffRef( 0 ) =
            std::log( static_cast< T >( u.degreeSlice( 0 ).coeff( 0 ) ) );
        return;
    }
    const T u0 = static_cast< T >( u.degreeSlice( 0 ).coeff( 0 ) );
    const T inv_u0 = T{ 1 } / u0;
    auto out_d = out.degreeSlice( d );
    auto u_d = u.degreeSlice( d );
    const std::size_t dsize = degreeSize( d, nvars );
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) =
            static_cast< T >( u_d.coeff( static_cast< Eigen::Index >( i ) ) ) * inv_u0;
    }
    const T inv_d_u0 = inv_u0 / static_cast< T >( d );
    for ( std::size_t e = 1; e + 1 <= d; ++e )
    {
        const T scale = -static_cast< T >( d - e ) * inv_d_u0;
        auto u_e = u.degreeSlice( e );
        auto out_de = out.degreeSlice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, scale, u_e, out_de, out_d );
    }
}

}  // namespace tax::kernels
