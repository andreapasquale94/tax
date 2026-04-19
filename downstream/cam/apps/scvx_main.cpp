// =========================================================================
// scvx_main.cpp
// tax port of multi-impulsive-cam/main/mainSCVX.m
//
// Multi-impulse collision-avoidance manoeuvre optimisation. Reads a
// conjunction scenario from runtime/cam.dat, builds the linearised maps via
// tax DA, and solves both the SCVX (sequential convex) and a smoothed NLP
// formulation. Prints the resulting Δv sequence and avoidance metrics.
// =========================================================================

#include <Eigen/Dense>
#include <cam/cam.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace
{

bool read_scenario( const std::string& path, cam::Vec6< double >& xs0, cam::Vec6< double >& xd0,
                    cam::Mat3< double >& Ps, cam::Mat3< double >& Pd, double& t2tca,
                    int& nImpulses, double& dt, double& sqrMahaTarget, double& dvMax )
{
    std::ifstream in( path );
    if ( !in.is_open() )
    {
        std::cerr << "Input file not found: " << path << '\n';
        return false;
    }
    for ( int i = 0; i < 6; ++i ) in >> xs0[i];
    for ( int i = 0; i < 6; ++i ) in >> xd0[i];
    // Diagonal RTN covariance (km^2 for position) — six values per object:
    // sigma_R, sigma_T, sigma_N, then off-diagonals RT, RN, TN.
    double sR, sT, sN, sRT, sRN, sTN;
    in >> sR >> sT >> sN >> sRT >> sRN >> sTN;
    Ps << sR, sRT, sRN, sRT, sT, sTN, sRN, sTN, sN;
    in >> sR >> sT >> sN >> sRT >> sRN >> sTN;
    Pd << sR, sRT, sRN, sRT, sT, sTN, sRN, sTN, sN;
    in >> t2tca >> nImpulses >> dt >> sqrMahaTarget >> dvMax;
    return true;
}

}  // namespace

int main()
{
    cam::Vec6< double > xs0, xd0;
    cam::Mat3< double > Ps, Pd;
    double t2tca, dt, sqrMahaTarget, dvMax;
    int nImpulses;

    if ( !read_scenario( "runtime/cam.dat", xs0, xd0, Ps, Pd, t2tca, nImpulses, dt,
                         sqrMahaTarget, dvMax ) )
        return 1;

    cam::MultiImpulseConfig cfg;
    cfg.nImpulses = nImpulses;
    cfg.dtSeconds = dt;
    cfg.t2TCAseconds = t2tca;

    std::cout << "--- Multi-impulse CAM scenario ---\n";
    std::cout << "  N impulses : " << nImpulses << '\n';
    std::cout << "  dt         : " << dt << " s\n";
    std::cout << "  t2TCA      : " << t2tca << " s\n";
    std::cout << "  dvMax/comp : " << dvMax << " km/s\n";
    std::cout << "  target d^2 : " << sqrMahaTarget << '\n';

    cam::LinearMaps< double > L = cam::buildLinearMaps( xs0, xd0, Ps, Pd, cfg );
    std::cout << "  nominal miss distance      : " << L.rrb2Nom.norm() << " km\n";
    std::cout << "  nominal sqrMahalanobis     : " << L.sqrMahalanobisNom << '\n';
    std::cout << "  nominal TCA correction (s) : "
              << L.tcaScaled * cam::Scaling::make( cfg.mu, cfg.rE ).Tsc - t2tca << "\n\n";

    // --- NLP via smoothed penalty ---
    cam::optim::NLPConfig nlpOpt;
    nlpOpt.sqrMahalaTarget = sqrMahaTarget;
    nlpOpt.dvMax = dvMax;
    nlpOpt.verbose = true;
    cam::optim::NLPResult nlp = cam::optim::solveNLP( L, nlpOpt );

    std::cout << "\n[NLP result]\n";
    std::cout << "  Total Δv          : " << nlp.objective * 1e3 << " m/s\n";
    std::cout << "  Final sqrMaha     : " << nlp.sqrMahalanobis << '\n';
    std::cout << "  Iterations        : " << nlp.iterations << '\n';
    std::cout << "  Converged         : " << ( nlp.converged ? "yes" : "no" ) << '\n';

    // --- SCVX with single linearisation pass ---
    cam::optim::SCVXConfig scvxOpt;
    scvxOpt.sqrMahalaTarget = sqrMahaTarget;
    scvxOpt.dvMax = dvMax;
    scvxOpt.verbose = true;
    cam::optim::SCVXResult scvx = cam::optim::solveSCVX( xs0, xd0, Ps, Pd, cfg, scvxOpt );

    std::cout << "\n[SCVX result]\n";
    std::cout << "  Total Δv          : " << scvx.totalDeltaV * 1e3 << " m/s\n";
    std::cout << "  Final sqrMaha     : " << scvx.sqrMahalanobis << '\n';
    std::cout << "  Outer iterations  : " << scvx.outerIterations << '\n';
    std::cout << "  Miss distance     : " << scvx.missDistance << " km\n";
    std::cout << "  Converged         : " << ( scvx.converged ? "yes" : "no" ) << '\n';

    // --- Save Δv sequence ---
    std::ofstream outDv( "runtime/dv.dat" );
    outDv << std::setprecision( 16 );
    for ( int k = 0; k < nImpulses; ++k )
        outDv << scvx.dv( k * 3 ) << ' ' << scvx.dv( k * 3 + 1 ) << ' '
              << scvx.dv( k * 3 + 2 ) << '\n';

    return 0;
}
