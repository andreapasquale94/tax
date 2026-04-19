// =========================================================================
// state_prop.cpp
// tax port of multi-impulsive-cam/cpp/stateProp.cpp
//
// Single-step DA state propagation driver. Reads debris and spacecraft
// initial states plus nominal time-to-TCA from runtime/xd0.dat and
// runtime/xs0.dat, propagates both under Kepler and averaged-J2 dynamics
// using tax DA variables, refines the TCA via polynomial map inversion,
// and writes propagated states to runtime/xdfKep.dat, xsfKep.dat,
// xdfJ2.dat, xsfJ2.dat.
// =========================================================================

#include <Eigen/Dense>
#include <cam/cam.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tax/tax.hpp>

namespace
{

constexpr int kOrder = 1;
constexpr int kVars = 7;  // 6 spacecraft state perturbations + 1 time perturbation
using DA = tax::TEn< kOrder, kVars >;

bool read_state( const std::string& path, Eigen::Matrix< double, 6, 1 >& x, double& tdum )
{
    std::ifstream infile( path );
    if ( !infile.is_open() )
    {
        std::cerr << "Input file not found: " << path << '\n';
        return false;
    }
    for ( int i = 0; i < 6; ++i ) infile >> x[i];
    infile >> tdum;
    return true;
}

void write_state( const std::string& path, const Eigen::Matrix< double, 6, 1 >& x, double Lsc,
                  double Vsc )
{
    std::ofstream out( path );
    out << std::setprecision( 16 );
    for ( int i = 0; i < 3; ++i ) out << x[i] * Lsc << '\n';
    for ( int i = 3; i < 6; ++i ) out << x[i] * Vsc << '\n';
}

template < int I >
DA axis( double offset )
{
    typename DA::Input x0{};
    return DA::template variable< I >( x0 ) + offset;
}

}  // namespace

int main()
{
    const double mu = cam::EarthPhysics::mu;
    const double rE = cam::EarthPhysics::rE;
    const double J2 = cam::EarthPhysics::J2;

    const cam::Scaling s = cam::Scaling::make( mu, rE );

    Eigen::Matrix< double, 6, 1 > xd_raw, xs_raw;
    double td_nom = 0.0, ts_nom = 0.0;
    if ( !read_state( "runtime/xd0.dat", xd_raw, td_nom ) ) return 1;
    if ( !read_state( "runtime/xs0.dat", xs_raw, ts_nom ) ) return 1;

    // Debris: constant DA (no DA dependence). Spacecraft: DA variables for state.
    cam::Vec6< DA > xd0, xs0;
    for ( int i = 0; i < 3; ++i ) xd0[i] = DA{ xd_raw[i] / s.Lsc };
    for ( int i = 3; i < 6; ++i ) xd0[i] = DA{ xd_raw[i] / s.Vsc };

    xs0[0] = axis< 0 >( xs_raw[0] / s.Lsc );
    xs0[1] = axis< 1 >( xs_raw[1] / s.Lsc );
    xs0[2] = axis< 2 >( xs_raw[2] / s.Lsc );
    xs0[3] = axis< 3 >( xs_raw[3] / s.Vsc );
    xs0[4] = axis< 4 >( xs_raw[4] / s.Vsc );
    xs0[5] = axis< 5 >( xs_raw[5] / s.Vsc );

    const DA t2c = axis< 6 >( ts_nom ) * ( 1.0 / s.Tsc );

    const cam::Vec6< DA > xdfJ2 = cam::propJ2An( xd0, t2c, s.muSc, s.rESc, J2 );
    const cam::Vec6< DA > xdfKep = cam::propKepAn( xd0, t2c, s.muSc );
    const cam::Vec6< DA > xsfJ2 = cam::propJ2An( xs0, t2c, s.muSc, s.rESc, J2 );
    const cam::Vec6< DA > xsfKep = cam::propKepAn( xs0, t2c, s.muSc );

    // Relative state under J2 at nominal TCA
    cam::Vec6< DA > xrel;
    for ( int i = 0; i < 6; ++i ) xrel[i] = xsfJ2[i] - xdfJ2[i];

    cam::Vec3< DA > rr, vv;
    rr << xrel[0], xrel[1], xrel[2];
    vv << xrel[3], xrel[4], xrel[5];

    std::cout << "nominal dot(r,v)                = " << rr.dot( vv ).value() << '\n';
    std::cout << "nominal relative distance [km]  = " << cam::vnorm( rr ).value() * s.Lsc << '\n';

    const DA tca = cam::findTCA< kOrder, kVars >( xrel );

    // Evaluate the propagated states at the refined TCA.
    typename DA::Input dx{};
    for ( int i = 0; i < 6; ++i ) dx[i] = 0.0;
    dx[6] = tca.value();

    cam::Vec6< double > xdfJ2_v, xsfJ2_v, xdfKep_v, xsfKep_v;
    for ( int i = 0; i < 6; ++i )
    {
        xdfJ2_v[i] = xdfJ2[i].eval( dx );
        xsfJ2_v[i] = xsfJ2[i].eval( dx );
        xdfKep_v[i] = xdfKep[i].eval( dx );
        xsfKep_v[i] = xsfKep[i].eval( dx );
    }

    cam::Vec6< double > xrel_v;
    for ( int i = 0; i < 6; ++i ) xrel_v[i] = xsfJ2_v[i] - xdfJ2_v[i];
    cam::Vec3< double > rr_v{ xrel_v[0], xrel_v[1], xrel_v[2] };
    cam::Vec3< double > vv_v{ xrel_v[3], xrel_v[4], xrel_v[5] };
    std::cout << "dot(r,v) at refined TCA         = " << rr_v.dot( vv_v ) << '\n';
    std::cout << "relative distance at TCA [km]   = " << cam::vnorm( rr_v ) * s.Lsc << '\n';
    std::cout << "refined TCA (s from epoch0)     = " << tca.value() * s.Tsc << '\n';

    write_state( "runtime/xdfKep.dat", xdfKep_v, s.Lsc, s.Vsc );
    write_state( "runtime/xsfKep.dat", xsfKep_v, s.Lsc, s.Vsc );
    write_state( "runtime/xdfJ2.dat", xdfJ2_v, s.Lsc, s.Vsc );
    write_state( "runtime/xsfJ2.dat", xsfJ2_v, s.Lsc, s.Vsc );
    return 0;
}
