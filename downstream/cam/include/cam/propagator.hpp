#pragma once

#include <cam/anomaly.hpp>
#include <cam/constants.hpp>
#include <cam/elements.hpp>
#include <cam/linalg.hpp>
#include <cmath>

namespace cam
{

/// @brief Averaged J2 right-hand side in Delaunay variables.
template < typename T >
[[nodiscard]] Vec6< T > averagedJ2rhs( const Vec6< T >& x, double mu, double J2, double rE )
{
    using std::acos;
    using std::sin;

    const T& L = x[3];
    const T& G = x[4];
    const T& H = x[5];

    const T eta = G / L;
    const T ci = H / G;
    const T si = sin( acos( ci ) );

    const double mu2 = mu * mu;
    const double mu4 = mu2 * mu2;
    const double rE2 = rE * rE;

    const T L2 = L * L;
    const T L3 = L2 * L;
    const T L6 = L3 * L3;
    const T L7 = L6 * L;
    const T eta2 = eta * eta;
    const T eta3 = eta2 * eta;
    const T eta4 = eta2 * eta2;
    const T si2 = si * si;
    const T ci2 = ci * ci;

    const T dldt = mu2 / L3 + ( ( 3.0 * J2 * rE2 * mu4 ) / ( 2.0 * L7 * eta3 ) -
                                ( 9.0 * J2 * si2 * rE2 * mu4 ) / ( 4.0 * L7 * eta3 ) );

    const T dgdt = ( ( 3.0 * J2 * rE2 * mu4 ) / ( 2.0 * L7 * eta4 ) -
                     ( 9.0 * J2 * si2 * rE2 * mu4 ) / ( 4.0 * L7 * eta4 ) +
                     ( 3.0 * ci2 * J2 * rE2 * mu4 ) / ( 2.0 * G * L6 * eta3 ) );

    const T dhdt = -( 3.0 * ci2 * J2 * rE2 * mu4 ) / ( 2.0 * H * L6 * eta3 );

    Vec6< T > ff;
    ff << dldt, dgdt, dhdt, T{ 0.0 }, T{ 0.0 }, T{ 0.0 };
    return ff;
}

/// @brief Analytical averaged-J2 propagation of a cartesian state by time-of-flight `tof`.
template < typename T >
[[nodiscard]] Vec6< T > propJ2An( const Vec6< T >& xx0, const T& tof, double mu, double rE,
                                  double J2 )
{
    Vec6< T > kep0 = cart2kep( xx0, mu );
    Vec6< T > hill0 = kep2hill( kep0, mu );
    Vec6< T > hill0Mean = osculating2meanHill( hill0, mu, J2, rE );

    Vec6< T > kep0Mean = hill2kep( hill0Mean, mu );
    kep0Mean[5] = true2meanAnomaly( kep0Mean[5], kep0Mean[1] );

    Vec6< T > del0Mean = kep2delaunay( kep0Mean, mu );
    Vec6< T > delRate = averagedJ2rhs( del0Mean, mu, J2, rE );
    Vec6< T > delfMean;
    for ( int i = 0; i < 6; ++i ) delfMean[i] = delRate[i] * tof + del0Mean[i];

    Vec6< T > kepfMean = delaunay2kep( delfMean, mu );
    kepfMean[5] = mean2trueAnomaly( kepfMean[5], kepfMean[1] );

    Vec6< T > hillfMean = kep2hill( kepfMean, mu );
    Vec6< T > hillf = mean2osculatingHill( hillfMean, mu, J2, rE );
    return hill2cart( hillf, mu );
}

/// @brief Analytical Keplerian propagation using Lagrange coefficients and Kepler's equation.
/// @param xx0 Initial cartesian state.
/// @param t   Time of flight.
/// @param mu  Gravitational parameter.
/// @param iterations Number of Newton iterations on Kepler's equation (default 30).
template < typename T >
[[nodiscard]] Vec6< T > propKepAn( const Vec6< T >& xx0, const T& t, double mu,
                                   int iterations = 30 )
{
    using std::atan2;
    using std::cos;
    using std::cosh;
    using std::sin;
    using std::sinh;
    using std::sqrt;
    using std::tan;

    Vec3< T > rr0, vv0;
    rr0 << xx0[0], xx0[1], xx0[2];
    vv0 << xx0[3], xx0[4], xx0[5];

    const Vec3< T > hh = rr0.cross( vv0 );
    const T h = vnorm( hh );
    const T r0 = vnorm( rr0 );
    const T v0sq = vv0[0] * vv0[0] + vv0[1] * vv0[1] + vv0[2] * vv0[2];

    const T a = mu / ( 2.0 * mu / r0 - v0sq );
    const T p = h * h / mu;
    const T sigma0 = rr0.dot( vv0 ) / sqrt( T{ mu } );

    T F, Ft, G, Gt;

    if ( cons( a ) > 0.0 )
    {
        const T MmM0 = t * sqrt( T{ mu } / ( a * a * a ) );
        T EmE0 = T{ cons( MmM0 ) };
        for ( int i = 0; i < iterations; ++i )
        {
            const T fx0 = -MmM0 + EmE0 + sigma0 / sqrt( a ) * ( 1.0 - cos( EmE0 ) ) -
                          ( 1.0 - r0 / a ) * sin( EmE0 );
            const T fxp = 1.0 + sigma0 / sqrt( a ) * sin( EmE0 ) -
                          ( 1.0 - r0 / a ) * cos( EmE0 );
            EmE0 = EmE0 - fx0 / fxp;
        }

        const T theta = 2.0 * atan2( sqrt( a * p ) * tan( EmE0 / 2.0 ),
                                     r0 + sigma0 * sqrt( a ) * tan( EmE0 / 2.0 ) );
        const T r = p * r0 / ( r0 + ( p - r0 ) * cos( theta ) - sqrt( p ) * sigma0 * sin( theta ) );

        F = 1.0 - a / r0 * ( 1.0 - cos( EmE0 ) );
        G = a * sigma0 / sqrt( T{ mu } ) * ( 1.0 - cos( EmE0 ) ) +
            r0 * sqrt( a / mu ) * sin( EmE0 );
        Ft = -sqrt( mu * a ) / ( r * r0 ) * sin( EmE0 );
        Gt = 1.0 - a / r * ( 1.0 - cos( EmE0 ) );
    }
    else
    {
        const T NmN0 = t * sqrt( T{ mu } / ( ( -a ) * ( -a ) * ( -a ) ) );
        T HmH0 = T{ 0.0 };
        for ( int i = 0; i < iterations; ++i )
        {
            const T fx0 = -NmN0 - HmH0 + sigma0 / sqrt( -a ) * ( -1.0 + cosh( HmH0 ) ) +
                          ( 1.0 - r0 / a ) * sinh( HmH0 );
            const T fxp = -1.0 + sigma0 / sqrt( -a ) * sinh( HmH0 ) +
                          ( 1.0 - r0 / a ) * cosh( HmH0 );
            HmH0 = HmH0 - fx0 / fxp;
        }

        F = 1.0 - a / r0 * ( 1.0 - cosh( HmH0 ) );
        G = a * sigma0 / sqrt( T{ mu } ) * ( 1.0 - cosh( HmH0 ) ) +
            r0 * sqrt( -a / mu ) * sinh( HmH0 );
        Vec3< T > rv_temp;
        for ( int i = 0; i < 3; ++i ) rv_temp[i] = F * rr0[i] + G * vv0[i];
        const T r = vnorm( rv_temp );
        Ft = -sqrt( mu * ( -a ) ) / ( r * r0 ) * sinh( HmH0 );
        Gt = 1.0 - a / r * ( 1.0 - cosh( HmH0 ) );
    }

    Vec6< T > xxf;
    for ( int i = 0; i < 3; ++i )
    {
        xxf[i] = F * rr0[i] + G * vv0[i];
        xxf[i + 3] = Ft * rr0[i] + Gt * vv0[i];
    }
    return xxf;
}

/// @brief Right-hand side for zonal-harmonic (J2, J3, J4) dynamics in scaled units.
/// @details Scaling is baked in: `mu` and `rE` are internally rescaled to mirror the original
/// implementation. Input `xx` is assumed scaled so that `Lsc = rE`, `Vsc = sqrt(mu/rE)`.
template < typename T >
[[nodiscard]] Vec6< T > rhsJ234( const Vec6< T >& xx, double /*t*/ )
{
    const Scaling s = Scaling::make( EarthPhysics::mu, EarthPhysics::rE );
    const double mu = s.muSc;
    const double rE = s.rESc;
    const double J2 = EarthPhysics::J2;
    const double J3 = EarthPhysics::J3;
    const double J4 = EarthPhysics::J4;

    Vec3< T > pos, vel;
    pos << xx[0], xx[1], xx[2];
    vel << xx[3], xx[4], xx[5];

    Vec6< T > res;
    res[0] = xx[3];
    res[1] = xx[4];
    res[2] = xx[5];

    const T r = vnorm( pos );
    const T x = pos[0];
    const T y = pos[1];
    const T z = pos[2];

    const T mur3 = mu / ( r * r * r );
    const T z2r2 = ( z / r ) * ( z / r );

    const T J2rEr = 1.5 * J2 * rE / r * rE / r;
    const T mur7Er3 = 5.0 / 2.0 * mur3 * J3 * rE / r * rE / r * rE / r / r;
    const T mur7Er4 = 15.0 / 8.0 * mur3 * J4 * rE / r * rE / r * rE / r * rE / r;

    res[3] = -pos[0] * mur3 * ( 1.0 + J2rEr * ( 1.0 - 5.0 * z2r2 ) );
    res[4] = -pos[1] * mur3 * ( 1.0 + J2rEr * ( 1.0 - 5.0 * z2r2 ) );
    res[5] = -pos[2] * mur3 * ( 1.0 + J2rEr * ( 3.0 - 5.0 * z2r2 ) );

    res[3] = res[3] + ( mur7Er3 * x * z * ( 7.0 * z2r2 - 3.0 ) +
                        mur7Er4 * x * ( 1.0 - 14.0 * z2r2 + 21.0 * z2r2 * z2r2 ) );
    res[4] = res[4] + ( mur7Er3 * y * z * ( 7.0 * z2r2 - 3.0 ) +
                        mur7Er4 * y * ( 1.0 - 14.0 * z2r2 + 21.0 * z2r2 * z2r2 ) );
    res[5] = res[5] +
             ( mur7Er3 * r * r * ( 3.0 / 5.0 - 6.0 * z2r2 + 7.0 * z2r2 * z2r2 ) +
               mur7Er4 * z * ( 5.0 - 70.0 / 3.0 * z2r2 + 21.0 * z2r2 * z2r2 ) );

    return res;
}

/// @brief Backward-integration RHS (negates `rhsJ234`).
template < typename T >
[[nodiscard]] Vec6< T > rhsJ234Back( const Vec6< T >& xx, double t )
{
    return -rhsJ234( xx, t );
}

}  // namespace cam
