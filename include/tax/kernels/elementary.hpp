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
        out.slice( 0 ).coeffRef( 0 ) =
            std::sqrt( static_cast< T >( u.slice( 0 ).coeff( 0 ) ) );
        return;
    }
    auto out_d = out.slice( d );
    auto u_d = u.slice( d );
    const std::size_t dsize = degreeSize( d, nvars );
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) =
            static_cast< T >( u_d.coeff( static_cast< Eigen::Index >( i ) ) );
    }
    for ( std::size_t e = 1; e + 1 <= d; ++e )
    {
        auto out_e = out.slice( e );
        auto out_de = out.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, T{ -1 }, out_e, out_de, out_d );
    }
    const T out0 = static_cast< T >( out.slice( 0 ).coeff( 0 ) );
    const T inv_2f0 = T{ 1 } / ( T{ 2 } * out0 );
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) *= inv_2f0;
    }
}

// out = cbrt(u).  Solve F^3 = u, maintaining G = F^2 alongside.  At
// degree d (with d >= 1):
//   F_d = (u_d
//        - sum_{1 <= |beta| <= d-1} G_beta * F_{alpha-beta}
//        - F_0 * sum_{1 <= |beta| <= d-1} F_beta * F_{alpha-beta}
//         ) / (3 F_0^2)
//   G_d = 2 F_0 F_d + sum_{1 <= |beta| <= d-1} F_beta * F_{alpha-beta}
template < class T, class U, class OutObj, class AuxObj >
inline void cbrtComputeDegree( std::size_t d, std::size_t nvars, const U& u,
                               const OutObj& out, const AuxObj& aux )
{
    using namespace tax::util;
    if ( d == 0 )
    {
        const T u0 = static_cast< T >( u.slice( 0 ).coeff( 0 ) );
        const T f0 = std::cbrt( u0 );
        out.slice( 0 ).coeffRef( 0 ) = f0;
        aux.slice( 0 ).coeffRef( 0 ) = f0 * f0;
        return;
    }
    auto out_d = out.slice( d );
    auto u_d = u.slice( d );
    const std::size_t dsize = degreeSize( d, nvars );
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) =
            static_cast< T >( u_d.coeff( static_cast< Eigen::Index >( i ) ) );
    }
    // Subtract sum_{|beta|=1..d-1} G_beta F_{d-|beta|}.
    for ( std::size_t e = 1; e + 1 <= d; ++e )
    {
        auto aux_e = aux.slice( e );
        auto out_de = out.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, T{ -1 }, aux_e, out_de, out_d );
    }
    // Subtract F_0 * sum_{|beta|=1..d-1} F_beta F_{d-|beta|}.
    const T f0 = static_cast< T >( out.slice( 0 ).coeff( 0 ) );
    const T neg_f0 = -f0;
    for ( std::size_t e = 1; e + 1 <= d; ++e )
    {
        auto out_e = out.slice( e );
        auto out_de = out.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, neg_f0, out_e, out_de, out_d );
    }
    const T inv_3f0sq = T{ 1 } / ( T{ 3 } * f0 * f0 );
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) *= inv_3f0sq;
    }

    // Maintain G_d = 2 F_0 F_d + sum_{|beta|=1..d-1} F_beta F_{d-|beta|}.
    auto aux_d = aux.slice( d );
    const T two_f0 = T{ 2 } * f0;
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        aux_d.coeffRef( static_cast< Eigen::Index >( i ) ) =
            two_f0 * out_d.coeff( static_cast< Eigen::Index >( i ) );
    }
    for ( std::size_t e = 1; e + 1 <= d; ++e )
    {
        auto out_e = out.slice( e );
        auto out_de = out.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, T{ 1 }, out_e, out_de, aux_d );
    }
}

// out = u^p (real exponent).  Recurrence from u F' = p u' F:
//   F_0 = pow(u_0, p)
//   F_alpha = (p * F_0 / u_0) * u_alpha
//           + (1 / (d * u_0)) * sum_{1 <= |beta| <= d-1, gamma = alpha - beta}
//             (d - |beta|) * (p * F_beta * u_gamma - u_beta * F_gamma)
template < class T, class U, class OutObj >
inline void powRealComputeDegree( std::size_t d, std::size_t nvars, T p, const U& u,
                                  const OutObj& out )
{
    using namespace tax::util;
    if ( d == 0 )
    {
        const T u0 = static_cast< T >( u.slice( 0 ).coeff( 0 ) );
        out.slice( 0 ).coeffRef( 0 ) = std::pow( u0, p );
        return;
    }
    const T u0 = static_cast< T >( u.slice( 0 ).coeff( 0 ) );
    const T inv_u0 = T{ 1 } / u0;
    const T f0 = static_cast< T >( out.slice( 0 ).coeff( 0 ) );

    auto out_d = out.slice( d );
    auto u_d = u.slice( d );
    const std::size_t dsize = degreeSize( d, nvars );
    const T leading = p * f0 * inv_u0;
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) =
            leading * static_cast< T >( u_d.coeff( static_cast< Eigen::Index >( i ) ) );
    }

    const T inv_d_u0 = inv_u0 / static_cast< T >( d );
    for ( std::size_t e = 1; e + 1 <= d; ++e )
    {
        const T weight = static_cast< T >( d - e ) * inv_d_u0;
        // + p * weight * F_e * u_{d-e}
        auto out_e = out.slice( e );
        auto u_de = u.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, p * weight, out_e, u_de, out_d );
        // - weight * u_e * F_{d-e}
        auto u_e = u.slice( e );
        auto out_de = out.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, -weight, u_e, out_de, out_d );
    }
}

}  // namespace tax::kernels
