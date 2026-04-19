#pragma once

#include <cam/anomaly.hpp>
#include <cam/linalg.hpp>
#include <cmath>
#include <limits>

namespace cam
{

/// @brief Keplerian elements (a, e, i, RAAN, omega, true_anomaly) -> Cartesian ECI.
template < typename T >
[[nodiscard]] Vec6< T > kep2cart( const Vec6< T >& kep, double mu ) noexcept
{
    using std::cos;
    using std::sin;
    using std::sqrt;

    const T p = kep[0] * ( 1.0 - kep[1] * kep[1] );

    Vec3< T > rm, vm;
    rm[0] = p * cos( kep[5] ) / ( 1.0 + kep[1] * cos( kep[5] ) );
    rm[1] = p * sin( kep[5] ) / ( 1.0 + kep[1] * cos( kep[5] ) );
    rm[2] = T{ 0.0 };
    vm[0] = -sin( kep[5] ) * sqrt( mu / p );
    vm[1] = ( kep[1] + cos( kep[5] ) ) * sqrt( mu / p );
    vm[2] = T{ 0.0 };

    const T cRA = cos( kep[3] );
    const T sRA = sin( kep[3] );
    const T cPA = cos( kep[4] );
    const T sPA = sin( kep[4] );
    const T ci = cos( kep[2] );
    const T si = sin( kep[2] );

    T RR[3][3];
    RR[0][0] = cRA * cPA - sRA * ci * sPA;
    RR[0][1] = -cRA * sPA - sRA * ci * cPA;
    RR[0][2] = sRA * si;
    RR[1][0] = sRA * cPA + cRA * ci * sPA;
    RR[1][1] = -sRA * sPA + cRA * ci * cPA;
    RR[1][2] = -cRA * si;
    RR[2][0] = si * sPA;
    RR[2][1] = si * cPA;
    RR[2][2] = ci;

    Vec3< T > rr = Vec3< T >::Zero();
    Vec3< T > vv = Vec3< T >::Zero();
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
        {
            rr[i] = rr[i] + RR[i][j] * rm[j];
            vv[i] = vv[i] + RR[i][j] * vm[j];
        }

    Vec6< T > out;
    out << rr[0], rr[1], rr[2], vv[0], vv[1], vv[2];
    return out;
}

/// @brief Cartesian ECI -> Keplerian elements.
template < typename T >
[[nodiscard]] Vec6< T > cart2kep( const Vec6< T >& rv, double mu )
{
    using std::acos;
    using std::asin;
    using std::atan2;
    using std::cos;
    using std::sin;
    using std::sqrt;

    Vec3< T > rr, vv;
    rr << rv[0], rv[1], rv[2];
    vv << rv[3], rv[4], rv[5];

    const T r = vnorm( rr );
    const T v2 = vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2];

    Vec3< T > h = rr.cross( vv );

    Vec6< T > kep;
    kep[0] = mu / ( 2.0 * ( mu / r - v2 / 2.0 ) );

    const T h1sqr = h[0] * h[0];
    const T h2sqr = h[1] * h[1];

    T RAAN;
    if ( cons( h1sqr + h2sqr ) == 0.0 )
    {
        RAAN = T{ 0.0 };
    }
    else
    {
        const T denom = sqrt( h1sqr + h2sqr );
        const T sinO = h[0] / denom;
        const T cosO = -h[1] / denom;
        if ( cons( cosO ) >= 0.0 )
        {
            if ( cons( sinO ) >= 0.0 )
                RAAN = asin( sinO );
            else
                RAAN = 2.0 * M_PI + asin( sinO );
        }
        else
        {
            if ( cons( sinO ) >= 0.0)
                RAAN = acos( cosO );
            else
                RAAN = 2.0 * M_PI - acos( cosO );
        }
    }

    const Vec3< T > vc = vv.cross( h );
    Vec3< T > ee;
    ee[0] = vc[0] / mu - rr[0] / r;
    ee[1] = vc[1] / mu - rr[1] / r;
    ee[2] = vc[2] / mu - rr[2] / r;
    const T e = vnorm( ee );

    const T i = acos( h[2] / vnorm( h ) );

    kep[1] = e;
    kep[2] = i;
    kep[3] = RAAN;

    T omega;
    T theta;
    if ( cons( e ) <= 1.0e-8 && cons( i ) < 1.0e-8 )
    {
        omega = atan2( rr[1], rr[0] );
        theta = T{ 0.0 };
        kep[4] = omega;
        kep[5] = theta;
        return kep;
    }

    if ( cons( e ) <= 1.0e-8 && cons( i ) >= 1.0e-8 )
    {
        omega = T{ 0.0 };
        Vec3< T > P, Q, W;
        P[0] = cos( omega ) * cos( RAAN ) - sin( omega ) * sin( i ) * sin( RAAN );
        P[1] = -sin( omega ) * cos( RAAN ) - cos( omega ) * cos( i ) * sin( RAAN );
        P[2] = sin( RAAN ) * sin( i );
        Q[0] = cos( omega ) * sin( RAAN ) + sin( omega ) * cos( i ) * cos( RAAN );
        Q[1] = -sin( omega ) * sin( RAAN ) + cos( omega ) * cos( i ) * cos( RAAN );
        Q[2] = -cos( RAAN ) * sin( i );
        W[0] = sin( omega ) * sin( i );
        W[1] = cos( omega ) * sin( i );
        W[2] = cos( i );
        Vec3< T > rrt;
        rrt[0] = P[0] * rr[0] + Q[0] * rr[1] + W[0] * rr[2];
        rrt[1] = P[1] * rr[0] + Q[1] * rr[1] + W[1] * rr[2];
        rrt[2] = P[2] * rr[0] + Q[2] * rr[1] + W[2] * rr[2];
        theta = atan2( rrt[1], rrt[0] );
        kep[4] = omega;
        kep[5] = theta;
        return kep;
    }

    T dotRxE = rr[0] * ee[0] + rr[1] * ee[1] + rr[2] * ee[2];
    T RxE = vnorm( rr ) * vnorm( ee );
    const double tiny = std::numeric_limits< double >::epsilon() * std::abs( cons( dotRxE ) );
    if ( std::abs( cons( dotRxE ) ) > std::abs( cons( RxE ) ) &&
         std::abs( cons( dotRxE ) ) - std::abs( cons( RxE ) ) < std::abs( tiny ) )
    {
        dotRxE = dotRxE - ( dotRxE * std::numeric_limits< double >::epsilon() );
    }
    theta = acos( dotRxE / RxE );

    if ( cons( rr.dot( vv ) ) < 0.0 ) theta = 2.0 * M_PI - theta;

    if ( cons( i ) <= 1.0e-8 && cons( e ) >= 1.0e-8 )
    {
        omega = atan2( ee[1], ee[0] );
        kep[4] = omega;
        kep[5] = theta;
        return kep;
    }

    const T sino = rr[2] / r / sin( i );
    const T coso = ( rr[0] * cos( RAAN ) + rr[1] * sin( RAAN ) ) / r;
    T argLat;
    if ( cons( coso ) >= 0.0 )
    {
        if ( cons( sino ) >= 0.0 )
            argLat = asin( sino );
        else
            argLat = 2.0 * M_PI + asin( sino );
    }
    else
    {
        if ( cons( sino ) >= 0.0 )
            argLat = acos( coso );
        else
            argLat = 2.0 * M_PI - acos( coso );
    }
    omega = argLat - theta;
    if ( cons( omega ) < 0.0 ) omega = omega + 2.0 * M_PI;

    kep[4] = omega;
    kep[5] = theta;
    return kep;
}

/// @brief Keplerian -> Delaunay canonical elements (l, g, h, L, G, H).
template < typename T >
[[nodiscard]] Vec6< T > kep2delaunay( const Vec6< T >& kep, double mu )
{
    using std::cos;
    using std::sqrt;
    Vec6< T > d;
    d[0] = kep[5];                         // l (mean anomaly)
    d[1] = kep[4];                         // g
    d[2] = kep[3];                         // h
    d[3] = sqrt( mu * kep[0] );            // L
    d[4] = sqrt( 1.0 - kep[1] * kep[1] ) * d[3];  // G
    d[5] = cos( kep[2] ) * d[4];           // H
    return d;
}

/// @brief Delaunay -> Keplerian. The 6th element is returned as mean anomaly.
template < typename T >
[[nodiscard]] Vec6< T > delaunay2kep( const Vec6< T >& del, double mu )
{
    using std::acos;
    using std::sqrt;
    Vec6< T > kep;
    kep[0] = del[3] * del[3] / mu;
    kep[1] = sqrt( 1.0 - ( del[4] / del[3] ) * ( del[4] / del[3] ) );
    kep[2] = acos( del[5] / del[4] );
    kep[3] = del[2];
    kep[4] = del[1];
    kep[5] = del[0];
    return kep;
}

/// @brief Keplerian -> Hill elements (r, theta, nu, R, Theta, N).
template < typename T >
[[nodiscard]] Vec6< T > kep2hill( const Vec6< T >& kep, double mu )
{
    using std::cos;
    using std::sin;
    using std::sqrt;
    Vec6< T > h;
    const T p = kep[0] * ( 1.0 - kep[1] * kep[1] );
    const T f = kep[5];
    h[4] = sqrt( mu * p );
    h[0] = p / ( 1.0 + kep[1] * cos( f ) );
    h[1] = f + kep[4];
    h[2] = kep[3];
    h[3] = ( h[4] / p ) * kep[1] * sin( f );
    h[5] = h[4] * cos( kep[2] );
    return h;
}

/// @brief Hill -> Cartesian ECI.
template < typename T >
[[nodiscard]] Vec6< T > hill2cart( const Vec6< T >& hill, double /*mu*/ )
{
    using std::cos;
    using std::sin;
    using std::sqrt;

    const T r = hill[0];
    const T th = hill[1];
    const T nu = hill[2];
    const T R = hill[3];
    const T Th = hill[4];
    const T ci = hill[5] / hill[4];
    const T si = sqrt( 1.0 - ci * ci );

    Vec3< T > u;
    u[0] = cos( th ) * cos( nu ) - ci * sin( th ) * sin( nu );
    u[1] = cos( th ) * sin( nu ) + ci * sin( th ) * cos( nu );
    u[2] = si * sin( th );

    Vec6< T > c;
    c[0] = r * u[0];
    c[1] = r * u[1];
    c[2] = r * u[2];
    c[3] = ( R * cos( th ) - Th * sin( th ) / r ) * cos( nu ) -
           ( R * sin( th ) + Th * cos( th ) / r ) * sin( nu ) * ci;
    c[4] = ( R * cos( th ) - Th * sin( th ) / r ) * sin( nu ) +
           ( R * sin( th ) + Th * cos( th ) / r ) * cos( nu ) * ci;
    c[5] = ( R * sin( th ) + Th * cos( th ) / r ) * si;
    return c;
}

/// @brief Hill -> Keplerian. Output 6th element is true anomaly.
template < typename T >
[[nodiscard]] Vec6< T > hill2kep( const Vec6< T >& hill, double mu )
{
    using std::acos;
    using std::cos;
    using std::sin;
    using std::sqrt;

    const T r = hill[0];
    const T th = hill[1];
    const T nu = hill[2];
    const T R = hill[3];
    const T Th = hill[4];
    const T Nu = hill[5];

    const T i = acos( Nu / Th );
    const T cs =
        ( -1.0 + ( Th * Th ) / ( mu * r ) ) * cos( th ) + ( R * Th * sin( th ) ) / mu;
    const T ss =
        -( ( R * Th * cos( th ) ) / mu ) + ( -1.0 + ( Th * Th ) / ( mu * r ) ) * sin( th );
    const T e = sqrt( cs * cs + ss * ss );
    const T p = Th * Th / mu;
    const T costrue = ( 1.0 / e ) * ( p / r - 1.0 );
    T f = acos( costrue );
    if ( cons( R ) < 0.0 ) f = 2.0 * M_PI - f;
    const T a = p / ( 1.0 - e * e );

    Vec6< T > kep;
    kep[0] = a;
    kep[1] = e;
    kep[2] = i;
    kep[3] = nu;
    kep[4] = th - f;
    kep[5] = f;
    return kep;
}

/// @brief Osculating Hill -> mean Hill (Brouwer-style J2 short-periodic removal).
template < typename T >
[[nodiscard]] Vec6< T > osculating2meanHill( const Vec6< T >& hill, double mu, double J2,
                                             double rE )
{
    using std::acos;
    using std::cos;
    using std::sin;
    using std::sqrt;

    const T r = hill[0];
    const T th = hill[1];
    const T nu = hill[2];
    const T R = hill[3];
    const T Th = hill[4];
    const T Nu = hill[5];

    const T ci = Nu / Th;
    const T si = sqrt( 1.0 - ci * ci );
    const T cs =
        ( -1.0 + ( Th * Th ) / ( mu * r ) ) * cos( th ) + ( R * Th * sin( th ) ) / mu;
    const T ss =
        -( ( R * Th * cos( th ) ) / mu ) + ( -1.0 + ( Th * Th ) / ( mu * r ) ) * sin( th );
    const T e = sqrt( cs * cs + ss * ss );
    const T eta = sqrt( 1.0 - e * e );
    const T beta = 1.0 / ( 1.0 + eta );
    const T p = Th * Th / mu;
    const T costrue = ( 1.0 / e ) * ( p / r - 1.0 );
    T f = acos( costrue );
    if ( cons( R ) < 0.0 ) f = 2.0 * M_PI - f;
    const T M = true2meanAnomaly( f, e );
    const T phi = f - M;

    const double rE2 = rE * rE;
    const T si2 = si * si;
    const T Th2 = Th * Th;
    const T Th3 = Th2 * Th;
    const T Th4 = Th2 * Th2;
    const T mu2 = T{ mu * mu };
    const T r2 = r * r;

    const T rMean =
        r + ( ( rE2 * beta * J2 ) / ( 2.0 * r ) - ( 3.0 * rE2 * beta * J2 * si2 ) / ( 4.0 * r ) +
              ( rE2 * eta * J2 * mu2 * r ) / Th4 -
              ( 3.0 * rE2 * eta * J2 * mu2 * r * si2 ) / ( 2.0 * Th4 ) +
              ( rE2 * J2 * mu ) / ( 2.0 * Th2 ) - ( rE2 * beta * J2 * mu ) / ( 2.0 * Th2 ) -
              ( 3.0 * rE2 * J2 * mu * si2 ) / ( 4.0 * Th2 ) +
              ( 3.0 * rE2 * beta * J2 * mu * si2 ) / ( 4.0 * Th2 ) -
              ( rE2 * J2 * mu * si2 * cos( 2.0 * th ) ) / ( 4.0 * Th2 ) );

    const T thMean =
        th + ( ( -3.0 * rE2 * J2 * mu2 * phi ) / Th4 +
               ( 15.0 * rE2 * J2 * mu2 * phi * si2 ) / ( 4.0 * Th4 ) -
               ( 5.0 * rE2 * J2 * mu * R ) / ( 2.0 * Th3 ) -
               ( rE2 * beta * J2 * mu * R ) / ( 2.0 * Th3 ) +
               ( 3.0 * rE2 * J2 * mu * R * si2 ) / Th3 +
               ( 3.0 * rE2 * beta * J2 * mu * R * si2 ) / ( 4.0 * Th3 ) -
               ( rE2 * beta * J2 * R ) / ( 2.0 * r * Th ) +
               ( 3.0 * rE2 * beta * J2 * R * si2 ) / ( 4.0 * r * Th ) +
               ( -( rE2 * J2 * mu * R ) / ( 2.0 * Th3 ) +
                 ( rE2 * J2 * mu * R * si2 ) / Th3 ) *
                   cos( 2.0 * th ) +
               ( -( rE2 * J2 * mu2 ) / ( 4.0 * Th4 ) +
                 ( 5.0 * rE2 * J2 * mu2 * si2 ) / ( 8.0 * Th4 ) +
                 ( rE2 * J2 * mu ) / ( r * Th2 ) -
                 ( 3.0 * rE2 * J2 * mu * si2 ) / ( 2.0 * r * Th2 ) ) *
                   sin( 2.0 * th ) );

    const T nuMean =
        nu + ( ( 3.0 * rE2 * ci * J2 * mu2 * phi ) / ( 2.0 * Th4 ) +
               ( 3.0 * rE2 * ci * J2 * mu * R ) / ( 2.0 * Th3 ) +
               ( rE2 * ci * J2 * mu * R * cos( 2.0 * th ) ) / ( 2.0 * Th3 ) +
               ( ( rE2 * ci * J2 * mu2 ) / ( 4.0 * Th4 ) -
                 ( rE2 * ci * J2 * mu ) / ( r * Th2 ) ) *
                   sin( 2.0 * th ) );

    const T RMean = R + ( -( rE2 * beta * J2 * R ) / ( 2.0 * r2 ) +
                          ( 3.0 * rE2 * beta * J2 * R * si2 ) / ( 4.0 * r2 ) -
                          ( rE2 * eta * J2 * mu2 * R ) / ( 2.0 * Th4 ) +
                          ( 3.0 * rE2 * eta * J2 * mu2 * R * si2 ) / ( 4.0 * Th4 ) +
                          ( rE2 * J2 * mu * si2 * sin( 2.0 * th ) ) / ( 2.0 * r2 * Th ) );

    const T ThMean = Th + ( ( ( rE2 * J2 * mu2 * si2 ) / ( 4.0 * Th3 ) -
                              ( rE2 * J2 * mu * si2 ) / ( r * Th ) ) *
                                cos( 2.0 * th ) -
                            ( rE2 * J2 * mu * R * si2 * sin( 2.0 * th ) ) / ( 2.0 * Th2 ) );

    const T NuMean = Nu;

    Vec6< T > out;
    out << rMean, thMean, nuMean, RMean, ThMean, NuMean;
    return out;
}

/// @brief Mean Hill -> osculating Hill (inverse of osculating2meanHill).
template < typename T >
[[nodiscard]] Vec6< T > mean2osculatingHill( const Vec6< T >& hill, double mu, double J2,
                                             double rE )
{
    using std::acos;
    using std::cos;
    using std::sin;
    using std::sqrt;

    const T r = hill[0];
    const T th = hill[1];
    const T nu = hill[2];
    const T R = hill[3];
    const T Th = hill[4];
    const T Nu = hill[5];

    const T ci = Nu / Th;
    const T si = sqrt( 1.0 - ci * ci );
    const T cs =
        ( -1.0 + ( Th * Th ) / ( mu * r ) ) * cos( th ) + ( R * Th * sin( th ) ) / mu;
    const T ss =
        -( ( R * Th * cos( th ) ) / mu ) + ( -1.0 + ( Th * Th ) / ( mu * r ) ) * sin( th );
    const T e = sqrt( cs * cs + ss * ss );
    const T eta = sqrt( 1.0 - e * e );
    const T beta = 1.0 / ( 1.0 + eta );
    const T p = Th * Th / mu;
    const T costrue = ( 1.0 / e ) * ( p / r - 1.0 );
    T f = acos( costrue );
    if ( cons( R ) < 0.0 ) f = 2.0 * M_PI - f;
    const T M = true2meanAnomaly( f, e );
    const T phi = f - M;

    const double rE2 = rE * rE;
    const T si2 = si * si;
    const T Th2 = Th * Th;
    const T Th3 = Th2 * Th;
    const T Th4 = Th2 * Th2;
    const T mu2 = T{ mu * mu };
    const T r2 = r * r;

    const T rOsc =
        r - ( ( rE2 * beta * J2 ) / ( 2.0 * r ) - ( 3.0 * rE2 * beta * J2 * si2 ) / ( 4.0 * r ) +
              ( rE2 * eta * J2 * mu2 * r ) / Th4 -
              ( 3.0 * rE2 * eta * J2 * mu2 * r * si2 ) / ( 2.0 * Th4 ) +
              ( rE2 * J2 * mu ) / ( 2.0 * Th2 ) - ( rE2 * beta * J2 * mu ) / ( 2.0 * Th2 ) -
              ( 3.0 * rE2 * J2 * mu * si2 ) / ( 4.0 * Th2 ) +
              ( 3.0 * rE2 * beta * J2 * mu * si2 ) / ( 4.0 * Th2 ) -
              ( rE2 * J2 * mu * si2 * cos( 2.0 * th ) ) / ( 4.0 * Th2 ) );

    const T thOsc =
        th - ( ( -3.0 * rE2 * J2 * mu2 * phi ) / Th4 +
               ( 15.0 * rE2 * J2 * mu2 * phi * si2 ) / ( 4.0 * Th4 ) -
               ( 5.0 * rE2 * J2 * mu * R ) / ( 2.0 * Th3 ) -
               ( rE2 * beta * J2 * mu * R ) / ( 2.0 * Th3 ) +
               ( 3.0 * rE2 * J2 * mu * R * si2 ) / Th3 +
               ( 3.0 * rE2 * beta * J2 * mu * R * si2 ) / ( 4.0 * Th3 ) -
               ( rE2 * beta * J2 * R ) / ( 2.0 * r * Th ) +
               ( 3.0 * rE2 * beta * J2 * R * si2 ) / ( 4.0 * r * Th ) +
               ( -( rE2 * J2 * mu * R ) / ( 2.0 * Th3 ) +
                 ( rE2 * J2 * mu * R * si2 ) / Th3 ) *
                   cos( 2.0 * th ) +
               ( -( rE2 * J2 * mu2 ) / ( 4.0 * Th4 ) +
                 ( 5.0 * rE2 * J2 * mu2 * si2 ) / ( 8.0 * Th4 ) +
                 ( rE2 * J2 * mu ) / ( r * Th2 ) -
                 ( 3.0 * rE2 * J2 * mu * si2 ) / ( 2.0 * r * Th2 ) ) *
                   sin( 2.0 * th ) );

    const T nuOsc =
        nu - ( ( 3.0 * rE2 * ci * J2 * mu2 * phi ) / ( 2.0 * Th4 ) +
               ( 3.0 * rE2 * ci * J2 * mu * R ) / ( 2.0 * Th3 ) +
               ( rE2 * ci * J2 * mu * R * cos( 2.0 * th ) ) / ( 2.0 * Th3 ) +
               ( ( rE2 * ci * J2 * mu2 ) / ( 4.0 * Th4 ) -
                 ( rE2 * ci * J2 * mu ) / ( r * Th2 ) ) *
                   sin( 2.0 * th ) );

    const T ROsc = R - ( -( rE2 * beta * J2 * R ) / ( 2.0 * r2 ) +
                         ( 3.0 * rE2 * beta * J2 * R * si2 ) / ( 4.0 * r2 ) -
                         ( rE2 * eta * J2 * mu2 * R ) / ( 2.0 * Th4 ) +
                         ( 3.0 * rE2 * eta * J2 * mu2 * R * si2 ) / ( 4.0 * Th4 ) +
                         ( rE2 * J2 * mu * si2 * sin( 2.0 * th ) ) / ( 2.0 * r2 * Th ) );

    const T ThOsc = Th - ( ( ( rE2 * J2 * mu2 * si2 ) / ( 4.0 * Th3 ) -
                             ( rE2 * J2 * mu * si2 ) / ( r * Th ) ) *
                               cos( 2.0 * th ) -
                           ( rE2 * J2 * mu * R * si2 * sin( 2.0 * th ) ) / ( 2.0 * Th2 ) );

    const T NuOsc = Nu;

    Vec6< T > out;
    out << rOsc, thOsc, nuOsc, ROsc, ThOsc, NuOsc;
    return out;
}

}  // namespace cam
