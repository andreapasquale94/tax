// SPDX-License-Identifier: BSD-3-Clause
//
// Recurrences for the inverse trigonometric and inverse hyperbolic
// functions.  Every one of {atan, atanh, asin, acos, asinh, acosh}
// reduces to the same shape:
//
//     G(u) * E[F] = sign * E[u]
//
// where G(u) is an auxiliary function of u (built independently with
// the existing kernels) and `sign` is +1 (for atan/atanh/asin/asinh/
// acosh) or -1 (for acos).  Splitting the Cauchy product on the left
// gives the per-degree recurrence
//
//     F_alpha = sign * u_alpha / G_0
//             - (1/(d * G_0)) * sum_{1 <= |beta| <= d-1} G_beta * (d - |beta|) * F_{alpha - beta}
//
// for d = |alpha| >= 1, with F_0 set from the closed-form value of
// the inverse function at u_0.
//
// Callers maintain G in their own auxiliary buffer (slice-providing
// object) and pass it here; this header has no opinion on how G itself
// is computed.

#pragma once

#include <cmath>
#include <cstddef>

#include "tax/kernels/cauchy.hpp"
#include "tax/util/multi_index.hpp"

namespace tax::kernels
{

// out_d is filled in-place from u, the auxiliary G, and out's lower slices.
//   * d == 0 -> caller must already have populated out.slice(0).coeffRef(0)
//     with the closed-form inverse-function value at u_0.
//   * d >= 1 -> applies the Euler-operator recurrence above.
template < class T, class U, class GObj, class OutObj >
inline void eulerAuxRecurrenceComputeDegree( std::size_t d, std::size_t nvars, T sign,
                                             const U& u, const GObj& g,
                                             const OutObj& out )
{
    using namespace tax::util;
    if ( d == 0 )
    {
        return;
    }
    const T g0 = static_cast< T >( g.slice( 0 ).coeff( 0 ) );
    const T inv_g0 = T{ 1 } / g0;

    auto out_d = out.slice( d );
    auto u_d = u.slice( d );
    const std::size_t dsize = degreeSize( d, nvars );
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) =
            sign * static_cast< T >( u_d.coeff( static_cast< Eigen::Index >( i ) ) ) * inv_g0;
    }

    const T inv_d_g0 = inv_g0 / static_cast< T >( d );
    for ( std::size_t e = 1; e + 1 <= d; ++e )
    {
        const T scale = -static_cast< T >( d - e ) * inv_d_g0;
        auto g_e = g.slice( e );
        auto out_de = out.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, scale, g_e, out_de, out_d );
    }
}

// out = atan2(y, x).  Recurrence from (x^2 + y^2) * E[F] = x*E[y] - y*E[x]:
//
//   F_alpha = (1/(d * G_0)) * (
//                 sum_{beta + gamma = alpha} (x_beta |gamma| y_gamma - y_beta |gamma| x_gamma)
//               - sum_{1 <= |beta| <= d-1} G_beta * (d - |beta|) * F_{alpha - beta} )
//
// where G = x^2 + y^2 is supplied through `g` (slice-provider).  The
// caller must pre-fill out.slice(0).coeffRef(0) with std::atan2(y_0, x_0).
template < class T, class X, class Y, class GObj, class OutObj >
inline void atan2ComputeDegree( std::size_t d, std::size_t nvars, const X& x, const Y& y,
                                const GObj& g, const OutObj& out )
{
    using namespace tax::util;
    if ( d == 0 )
    {
        return;
    }
    const T g0 = static_cast< T >( g.slice( 0 ).coeff( 0 ) );
    const T inv_g0 = T{ 1 } / g0;

    auto out_d = out.slice( d );
    const std::size_t dsize = degreeSize( d, nvars );
    for ( std::size_t i = 0; i < dsize; ++i )
    {
        out_d.coeffRef( static_cast< Eigen::Index >( i ) ) = T{ 0 };
    }

    // Numerator: x_beta |gamma| y_gamma - y_beta |gamma| x_gamma over all
    // (beta, gamma) with beta + gamma = alpha, |alpha| = d.  The |gamma|
    // factor zeroes out gamma = 0, so we only need |gamma| >= 1.
    const T inv_d_g0 = inv_g0 / static_cast< T >( d );
    for ( std::size_t e = 1; e <= d; ++e )
    {
        // |beta| = d - e, |gamma| = e
        const T scale = static_cast< T >( e ) * inv_d_g0;
        auto x_be = x.slice( d - e );
        auto y_be = y.slice( d - e );
        auto x_ge = x.slice( e );
        auto y_ge = y.slice( e );
        cauchyAccumulateSlice< T >( d - e, e, nvars, scale, x_be, y_ge, out_d );
        cauchyAccumulateSlice< T >( d - e, e, nvars, -scale, y_be, x_ge, out_d );
    }

    // Subtract sum_{|beta|=1..d-1} G_beta * (d-|beta|) * F_{d-|beta|}.
    for ( std::size_t e = 1; e + 1 <= d; ++e )
    {
        const T scale = -static_cast< T >( d - e ) * inv_d_g0;
        auto g_e = g.slice( e );
        auto out_de = out.slice( d - e );
        cauchyAccumulateSlice< T >( e, d - e, nvars, scale, g_e, out_de, out_d );
    }
}

}  // namespace tax::kernels
