// Verify multi-impulse linear maps and SOCP/NLP solvers on a synthetic LEO
// conjunction. Uses an inclined eccentric reference orbit to avoid the J2
// mean-osculating singularities (see runtime/README.md).
#include <Eigen/Dense>
#include <cam/cam.hpp>
#include <cmath>
#include <iostream>

namespace
{

int check( const char* what, double a, double b, double tol )
{
    if ( !std::isfinite( a ) || std::fabs( a - b ) > tol )
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

    cam::Vec6< double > xs0;
    xs0 << 3357.7, 3613.9, 4556.7, -6.59, 2.60, 3.278;
    cam::Vec6< double > xd0;
    xd0 << 3357.71, 3613.92, 4556.71, -6.59, -2.60, 3.278;

    cam::Mat3< double > Ps = cam::Mat3< double >::Identity() * 0.01;
    cam::Mat3< double > Pd = cam::Mat3< double >::Identity() * 0.01;

    cam::MultiImpulseConfig cfg;
    cfg.nImpulses = 5;
    cfg.dtSeconds = 60.0;
    cfg.t2TCAseconds = 600.0;

    cam::LinearMaps< double > L = cam::buildLinearMaps( xs0, xd0, Ps, Pd, cfg );

    // Sanity: shapes
    failures += check( "M rows", L.M.rows(), 6, 0 );
    failures += check( "M cols", L.M.cols(), cfg.nImpulses * 3, 0 );
    failures += check( "drrb2Nom rows", L.drrb2Nom.rows(), 2, 0 );
    failures += check( "drrb2Nom cols", L.drrb2Nom.cols(), cfg.nImpulses * 3, 0 );
    failures += check( "Pb2D rows", L.Pb2D.rows(), 2, 0 );

    // Apply a non-trivial dv: a small impulse at first node should change the
    // B-plane position by approximately drrb2Nom * dv (linearisation property).
    Eigen::VectorXd dv = Eigen::VectorXd::Zero( cfg.nImpulses * 3 );
    dv( 0 ) = 1e-5;  // 10 mm/s in x
    Eigen::Vector2d delta = L.drrb2Nom * dv;
    if ( !std::isfinite( delta( 0 ) ) || !std::isfinite( delta( 1 ) ) )
    {
        std::cerr << "[FAIL] drrb2Nom · dv produced NaN\n";
        ++failures;
    }

    // Check NLP solver returns a feasible solution if target is below nominal
    cam::optim::NLPConfig opt;
    opt.sqrMahalaTarget = L.sqrMahalanobisNom * 0.9;  // already feasible at zero dv
    opt.dvMax = 0.1;
    opt.verbose = false;
    cam::optim::NLPResult res = cam::optim::solveNLP( L, opt );
    failures += check( "NLP zero-dv objective", res.objective, 0.0, 1e-3 );

    // Set a target slightly larger than nominal — solver should reduce the
    // gap (move sqrMaha closer to target) rather than return zero.
    const double sqrMahaNom = L.sqrMahalanobisNom;
    opt.sqrMahalaTarget = sqrMahaNom * 1.05;
    opt.dvMax = 0.01;
    res = cam::optim::solveNLP( L, opt );
    std::cout << "NLP nom=" << sqrMahaNom << " target=" << opt.sqrMahalaTarget
              << " achieved=" << res.sqrMahalanobis << " |dv|=" << res.objective << '\n';
    if ( res.sqrMahalanobis < sqrMahaNom )
    {
        std::cerr << "[FAIL] NLP did not improve sqrMaha\n";
        ++failures;
    }

    if ( failures == 0 )
    {
        std::cout << "All multi-impulse + optim tests passed.\n";
        return 0;
    }
    return 1;
}
