// DA-based sanity tests: propagate, find TCA via polynomial map inversion.
#include <Eigen/Dense>
#include <cam/cam.hpp>
#include <cmath>
#include <iostream>
#include <tax/tax.hpp>

namespace
{

constexpr int kOrder = 2;
constexpr int kVars = 7;
using DA = tax::TEn< kOrder, kVars >;

template < int I >
DA axis( double offset )
{
    typename DA::Input x0{};
    return DA::template variable< I >( x0 ) + offset;
}

int check( const char* what, double a, double b, double tol )
{
    if ( std::fabs( a - b ) > tol )
    {
        std::cerr << "[FAIL] " << what << ": " << a << " vs " << b
                  << " (|diff|=" << std::fabs( a - b ) << ")\n";
        return 1;
    }
    return 0;
}

}  // namespace

int main()
{
    const double mu = cam::EarthPhysics::mu;
    const double rE = cam::EarthPhysics::rE;
    const cam::Scaling s = cam::Scaling::make( mu, rE );

    // Transverse conjunction: debris on equatorial prograde orbit, spacecraft
    // on polar orbit at nearly the same radius. They cross at the ascending
    // node producing a clean transverse encounter.
    cam::Vec6< double > xd_raw, xs_raw;
    const double r0 = 7000.0;
    const double v0 = std::sqrt( mu / r0 );
    xd_raw << r0, 0.0, 0.0, 0.0, v0, 0.0;
    xs_raw << r0 + 0.010, 0.0, 0.0, 0.0, 0.0, v0;
    const double t_nom = 0.0;  // nominal TCA right at epoch

    cam::Vec6< DA > xd0, xs0;
    for ( int i = 0; i < 3; ++i ) xd0[i] = DA{ xd_raw[i] / s.Lsc };
    for ( int i = 3; i < 6; ++i ) xd0[i] = DA{ xd_raw[i] / s.Vsc };

    xs0[0] = axis< 0 >( xs_raw[0] / s.Lsc );
    xs0[1] = axis< 1 >( xs_raw[1] / s.Lsc );
    xs0[2] = axis< 2 >( xs_raw[2] / s.Lsc );
    xs0[3] = axis< 3 >( xs_raw[3] / s.Vsc );
    xs0[4] = axis< 4 >( xs_raw[4] / s.Vsc );
    xs0[5] = axis< 5 >( xs_raw[5] / s.Vsc );

    const DA t2c = axis< 6 >( t_nom ) * ( 1.0 / s.Tsc );

    const cam::Vec6< DA > xdf = cam::propKepAn( xd0, t2c, s.muSc );
    const cam::Vec6< DA > xsf = cam::propKepAn( xs0, t2c, s.muSc );

    cam::Vec6< DA > xrel;
    for ( int i = 0; i < 6; ++i ) xrel[i] = xsf[i] - xdf[i];

    const DA tca = cam::findTCA< kOrder, kVars >( xrel );

    int failures = 0;

    typename DA::Input dx{};
    dx[6] = tca.value();
    cam::Vec6< double > xrel_v;
    for ( int i = 0; i < 6; ++i ) xrel_v[i] = xrel[i].eval( dx );

    cam::Vec3< double > r{ xrel_v[0], xrel_v[1], xrel_v[2] };
    cam::Vec3< double > v{ xrel_v[3], xrel_v[4], xrel_v[5] };

    std::cout << "dot(r,v) at TCA (scaled)      = " << r.dot( v ) << '\n';
    std::cout << "miss distance at TCA [km]     = " << cam::vnorm( r ) * s.Lsc << '\n';
    std::cout << "refined TCA [s]               = " << tca.value() * s.Tsc << '\n';

    failures += check( "dot(r,v) at TCA", r.dot( v ), 0.0, 1e-10 );
    // miss distance is driven by the 10 m radial offset
    failures += check( "miss distance [km]", cam::vnorm( r ) * s.Lsc, 0.010, 5e-3 );

    if ( failures == 0 )
    {
        std::cout << "TCA test passed.\n";
        return 0;
    }
    return 1;
}
