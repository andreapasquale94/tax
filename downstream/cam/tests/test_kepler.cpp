// Round-trip sanity tests for cam:: element conversions (double only).
#include <cam/cam.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

constexpr double kTol = 1e-9;

int check( const char* what, double a, double b, double tol = kTol )
{
    if ( !std::isfinite( a ) || !std::isfinite( b ) || std::fabs( a - b ) > tol )
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
    int failures = 0;

    // Low-earth-orbit-like state in km, km/s (unscaled).
    cam::Vec6< double > rv;
    rv << 7000.0, 0.0, 0.0, 0.0, 7.5, 1.5;

    const double mu = cam::EarthPhysics::mu;
    cam::Vec6< double > kep = cam::cart2kep( rv, mu );
    cam::Vec6< double > rv_back = cam::kep2cart( kep, mu );

    for ( int i = 0; i < 6; ++i )
        failures += check( i < 3 ? "pos round-trip" : "vel round-trip", rv_back[i], rv[i] );

    // Kepler analytical prop with t=0 is identity
    cam::Vec6< double > rv_t0 = cam::propKepAn( rv, 0.0, mu );
    for ( int i = 0; i < 6; ++i )
        failures +=
            check( i < 3 ? "propKep(t=0) pos" : "propKep(t=0) vel", rv_t0[i], rv[i], 1e-6 );

    // Round-trip: forward then backward by same time should return the initial state
    const double tof = 1500.0;  // seconds
    cam::Vec6< double > rvf = cam::propKepAn( rv, tof, mu );
    cam::Vec6< double > rvb = cam::propKepAn( rvf, -tof, mu );
    for ( int i = 0; i < 6; ++i )
        failures += check( i < 3 ? "propKep round-trip pos" : "propKep round-trip vel", rvb[i],
                           rv[i], 1e-6 );

    // kep <-> delaunay round-trip (uses mean anomaly convention)
    cam::Vec6< double > kep_M = kep;
    kep_M[5] = cam::true2meanAnomaly( kep[5], kep[1] );
    cam::Vec6< double > del = cam::kep2delaunay( kep_M, mu );
    cam::Vec6< double > kep_back = cam::delaunay2kep( del, mu );
    for ( int i = 0; i < 6; ++i )
        failures += check( "kep<->delaunay", kep_back[i], kep_M[i] );

    if ( failures == 0 )
    {
        std::cout << "All kepler round-trip tests passed.\n";
        return 0;
    }
    std::cerr << failures << " failure(s).\n";
    return 1;
}
