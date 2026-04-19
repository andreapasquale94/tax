// =========================================================================
// state_back_prop.cpp
// tax port of multi-impulsive-cam/cpp/stateBackProp.cpp
//
// Backward DA propagation of debris and spacecraft states from TCA to the
// initial epoch using both averaged-J2 analytical dynamics and the Kepler
// analytical propagator. Reads runtime/xdTCA.dat and runtime/xsTCA.dat,
// writes runtime/xd0Kep.dat, xs0Kep.dat, xd0J2.dat, xs0J2.dat.
// =========================================================================

#include <Eigen/Dense>
#include <cam/cam.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tax/tax.hpp>

namespace
{

constexpr int kOrder = 1;
constexpr int kVars = 6;
using DA = tax::TEn< kOrder, kVars >;

bool read_state( const std::string& path, Eigen::Matrix< double, 6, 1 >& x, double& tdum )
{
    std::ifstream in( path );
    if ( !in.is_open() )
    {
        std::cerr << "Input file not found: " << path << '\n';
        return false;
    }
    for ( int i = 0; i < 6; ++i ) in >> x[i];
    in >> tdum;
    return true;
}

void write_state( const std::string& path, const cam::Vec6< DA >& x, double Lsc, double Vsc )
{
    std::ofstream out( path );
    out << std::setprecision( 16 );
    for ( int i = 0; i < 3; ++i ) out << x[i].value() * Lsc << '\n';
    for ( int i = 3; i < 6; ++i ) out << x[i].value() * Vsc << '\n';
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

    Eigen::Matrix< double, 6, 1 > xd_tca, xs_tca;
    double td = 0.0, ts = 0.0;
    if ( !read_state( "runtime/xdTCA.dat", xd_tca, td ) ) return 1;
    if ( !read_state( "runtime/xsTCA.dat", xs_tca, ts ) ) return 1;

    cam::Vec6< DA > xdTCA, xsTCA;
    xdTCA[0] = axis< 0 >( xd_tca[0] / s.Lsc );
    xdTCA[1] = axis< 1 >( xd_tca[1] / s.Lsc );
    xdTCA[2] = axis< 2 >( xd_tca[2] / s.Lsc );
    xdTCA[3] = axis< 3 >( xd_tca[3] / s.Vsc );
    xdTCA[4] = axis< 4 >( xd_tca[4] / s.Vsc );
    xdTCA[5] = axis< 5 >( xd_tca[5] / s.Vsc );

    for ( int i = 0; i < 3; ++i ) xsTCA[i] = DA{ xs_tca[i] / s.Lsc };
    for ( int i = 3; i < 6; ++i ) xsTCA[i] = DA{ xs_tca[i] / s.Vsc };

    const DA negT2c = DA{ -td / s.Tsc };

    const cam::Vec6< DA > xd0J2 = cam::propJ2An( xdTCA, negT2c, s.muSc, s.rESc, J2 );
    const cam::Vec6< DA > xs0J2 = cam::propJ2An( xsTCA, negT2c, s.muSc, s.rESc, J2 );
    const cam::Vec6< DA > xd0Kep = cam::propKepAn( xdTCA, negT2c, s.muSc );
    const cam::Vec6< DA > xs0Kep = cam::propKepAn( xsTCA, negT2c, s.muSc );

    write_state( "runtime/xd0Kep.dat", xd0Kep, s.Lsc, s.Vsc );
    write_state( "runtime/xs0Kep.dat", xs0Kep, s.Lsc, s.Vsc );
    write_state( "runtime/xd0J2.dat", xd0J2, s.Lsc, s.Vsc );
    write_state( "runtime/xs0J2.dat", xs0J2, s.Lsc, s.Vsc );
    return 0;
}
