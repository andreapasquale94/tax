// SPDX-License-Identifier: BSD-3-Clause
//
// sin / cos and sinh / cosh paired recurrences over slice-providing operands.

#pragma once

#include <cmath>
#include <cstddef>

#include "tax/kernels/cauchy.hpp"
#include "tax/util/multi_index.hpp"

namespace tax::kernels
{

// (sin(u))_d, (cos(u))_d computed jointly:
//   sin_alpha = (1/d) sum_{e=1}^{d} e * u_gamma * cos_beta
//   cos_alpha = -(1/d) sum_{e=1}^{d} e * u_gamma * sin_beta
// `sin_obj` and `cos_obj` are slice-providing storage; the function
// updates both their degree-d slices.
template < class T, class U, class SinObj, class CosObj >
inline void sinCosComputeDegree( std::size_t d, std::size_t nvars, const U& u,
                                 const SinObj& sin_obj, const CosObj& cos_obj )
{
    if ( d == 0 )
    {
        const T u0 = static_cast< T >( u.slice( 0 ).coeff( 0 ) );
        sin_obj.slice( 0 ).coeffRef( 0 ) = std::sin( u0 );
        cos_obj.slice( 0 ).coeffRef( 0 ) = std::cos( u0 );
        return;
    }
    auto sin_d = sin_obj.slice( d );
    auto cos_d = cos_obj.slice( d );
    sin_d.setZero();
    cos_d.setZero();
    const T inv_d = T{ 1 } / static_cast< T >( d );
    for ( std::size_t e = 1; e <= d; ++e )
    {
        const T scale = static_cast< T >( e ) * inv_d;
        auto u_e = u.slice( e );
        auto cos_de = cos_obj.slice( d - e );
        auto sin_de = sin_obj.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, scale, u_e, cos_de, sin_d );
        cauchyAccumulateSlice< T >( e, d - e, nvars, -scale, u_e, sin_de, cos_d );
    }
}

// sinh / cosh: same pattern with no sign flip on the second equation.
template < class T, class U, class SinhObj, class CoshObj >
inline void sinhCoshComputeDegree( std::size_t d, std::size_t nvars, const U& u,
                                   const SinhObj& sinh_obj, const CoshObj& cosh_obj )
{
    if ( d == 0 )
    {
        const T u0 = static_cast< T >( u.slice( 0 ).coeff( 0 ) );
        sinh_obj.slice( 0 ).coeffRef( 0 ) = std::sinh( u0 );
        cosh_obj.slice( 0 ).coeffRef( 0 ) = std::cosh( u0 );
        return;
    }
    auto sinh_d = sinh_obj.slice( d );
    auto cosh_d = cosh_obj.slice( d );
    sinh_d.setZero();
    cosh_d.setZero();
    const T inv_d = T{ 1 } / static_cast< T >( d );
    for ( std::size_t e = 1; e <= d; ++e )
    {
        const T scale = static_cast< T >( e ) * inv_d;
        auto u_e = u.slice( e );
        auto cosh_de = cosh_obj.slice( d - e );
        auto sinh_de = sinh_obj.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, scale, u_e, cosh_de, sinh_d );
        cauchyAccumulateSlice< T >( e, d - e, nvars, scale, u_e, sinh_de, cosh_d );
    }
}

}  // namespace tax::kernels
