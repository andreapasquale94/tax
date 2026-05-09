// SPDX-License-Identifier: BSD-3-Clause
//
// Elementary algebraic recurrences: sqrt.

#pragma once

#include <cmath>
#include <cstddef>

#include "tax/kernels/cauchy.hpp"
#include "tax/util/multi_index.hpp"

namespace tax::kernels
{

// out = sqrt(u).  Recurrence from out * out = u:
//   out_d = (u_d - sum_{e=1}^{d-1} out_e * out_{d-e}) / (2 out_0)
template < class T, class U, class OutObj >
inline void sqrtComputeDegree( std::size_t d, std::size_t nvars, const U& u, const OutObj& out )
{
    using namespace tax::util;
    if ( d == 0 )
    {
        out.degreeSlice( 0 ).coeffRef( 0 ) =
            std::sqrt( static_cast< T >( u.degreeSlice( 0 ).coeff( 0 ) ) );
        return;
    }
    auto out_d = out.degreeSlice( d );
    auto u_d = u.degreeSlice( d );
    const std::size_t dsize = degreeSize( d, nvars );
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) =
            static_cast< T >( u_d.coeff( static_cast< Eigen::Index >( i ) ) );
    }
    for ( std::size_t e = 1; e + 1 <= d; ++e )
    {
        auto out_e = out.degreeSlice( e );
        auto out_de = out.degreeSlice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, T{ -1 }, out_e, out_de, out_d );
    }
    const T out0 = static_cast< T >( out.degreeSlice( 0 ).coeff( 0 ) );
    const T inv_2f0 = T{ 1 } / ( T{ 2 } * out0 );
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) *= inv_2f0;
    }
}

}  // namespace tax::kernels
