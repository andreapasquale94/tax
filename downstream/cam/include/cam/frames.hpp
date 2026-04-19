#pragma once

#include <cam/linalg.hpp>

namespace cam
{

/// @brief Rotation matrix mapping ECI to the RTN (radial/tangential/normal) frame of state `xx`.
template < typename T >
[[nodiscard]] Mat3< T > rtn( const Vec6< T >& xx )
{
    Vec3< T > rr, vv;
    rr << xx[0], xx[1], xx[2];
    vv << xx[3], xx[4], xx[5];

    const T rn = vnorm( rr );
    const T vn = vnorm( vv );
    rr[0] = rr[0] / rn;
    rr[1] = rr[1] / rn;
    rr[2] = rr[2] / rn;
    vv[0] = vv[0] / vn;
    vv[1] = vv[1] / vn;
    vv[2] = vv[2] / vn;

    Vec3< T > nn = rr.cross( vv );
    const T nnn = vnorm( nn );
    nn[0] = nn[0] / nnn;
    nn[1] = nn[1] / nnn;
    nn[2] = nn[2] / nnn;

    Vec3< T > tt = nn.cross( rr );
    const T ttn = vnorm( tt );
    tt[0] = tt[0] / ttn;
    tt[1] = tt[1] / ttn;
    tt[2] = tt[2] / ttn;

    Mat3< T > A;
    A.row( 0 ) = rr.transpose();
    A.row( 1 ) = tt.transpose();
    A.row( 2 ) = nn.transpose();
    return A;
}

/// @brief Encounter-plane axes built from primary (`vp`) and secondary (`vs`) velocities.
/// @param axis 0: xi (relative angular momentum), 1: eta (relative velocity), 2: zeta.
template < typename T >
[[nodiscard]] Vec3< T > encounterPlane( const Vec3< T >& vp, const Vec3< T >& vs, int axis )
{
    Vec3< T > relVel = vp - vs;
    const T reln = vnorm( relVel );
    Vec3< T > eta;
    eta[0] = relVel[0] / reln;
    eta[1] = relVel[1] / reln;
    eta[2] = relVel[2] / reln;

    Vec3< T > cv = vs.cross( vp );
    const T cvn = vnorm( cv );
    Vec3< T > xi;
    xi[0] = cv[0] / cvn;
    xi[1] = cv[1] / cvn;
    xi[2] = cv[2] / cvn;

    Vec3< T > zeta = xi.cross( eta );
    const T zn = vnorm( zeta );
    zeta[0] = zeta[0] / zn;
    zeta[1] = zeta[1] / zn;
    zeta[2] = zeta[2] / zn;

    if ( axis == 0 ) return xi;
    if ( axis == 1 ) return eta;
    return zeta;
}

/// @brief B-plane rotation matrix (rows are xi, eta, zeta axes).
template < typename T >
[[nodiscard]] Mat3< T > bplane( const Vec3< T >& vs, const Vec3< T >& vd )
{
    const Vec3< T > xi = encounterPlane( vs, vd, 0 );
    const Vec3< T > eta = encounterPlane( vs, vd, 1 );
    const Vec3< T > zeta = encounterPlane( vs, vd, 2 );

    Mat3< T > R;
    R.row( 0 ) = xi.transpose();
    R.row( 1 ) = eta.transpose();
    R.row( 2 ) = zeta.transpose();
    return R;
}

}  // namespace cam
