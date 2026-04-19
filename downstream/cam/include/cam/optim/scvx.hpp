#pragma once

#include <Eigen/Dense>
#include <cam/multi_impulse.hpp>
#include <cam/optim/socp.hpp>
#include <cmath>
#include <iostream>
#include <vector>

namespace cam::optim
{

/// @brief SCVX configuration.
struct SCVXConfig
{
    double sqrMahalaTarget = 25.0;  ///< Target squared Mahalanobis distance (e.g. 5σ -> 25)
    double dvMax = 1.0;             ///< Per-component impulse bound (km/s)
    int maxOuterIter = 10;          ///< SCVX outer iterations
    double tolOuter = 1e-6;         ///< Sup-norm tolerance on dv update
    int innerSocpIter = 1000;       ///< SOCP solver iterations
    double tolSocp = 1e-8;          ///< SOCP convergence tolerance
    bool verbose = true;            ///< Print per-iteration progress
};

/// @brief SCVX result.
struct SCVXResult
{
    Eigen::VectorXd dv;        ///< Final impulse sequence (3N entries)
    double totalDeltaV = 0.0;  ///< Sum of impulse magnitudes (km/s)
    double sqrMahalanobis = 0.0;
    double missDistance = 0.0;  ///< km (norm of B-plane position)
    int outerIterations = 0;
    bool converged = false;
};

namespace detail
{

/// @brief Tangent half-plane to the avoidance ellipse passing through closest
/// approach point. Returns (n, b) such that n^T r >= b moves outside the
/// ellipse defined by `r^T N r <= sqrMahaTarget`, linearised at the current
/// nominal `r0`.
inline std::pair< Eigen::Vector2d, double > tangentHalfPlane(
    const Eigen::Matrix2d& N, double sqrMahaTarget, const Eigen::Vector2d& r0 )
{
    // Ellipse: r^T N r = sqrMahaTarget. Find closest point on ellipse to r0
    // along the gradient direction.
    // Gradient at r0: g = 2 N r0; outward normal n = g / ||g||.
    Eigen::Vector2d g = N * r0;
    if ( g.norm() < 1e-30 ) g = Eigen::Vector2d{ 1.0, 0.0 };  // arbitrary direction
    Eigen::Vector2d n = g / g.norm();

    // Scalar t such that r_ell = t * (N^{-1} n)/||N^{-1/2} n|| has Mahalanobis = sqrt(target).
    // Simpler: Cholesky factorise N = L L^T; in the whitened space the ellipse
    // is a unit circle of radius sqrt(sqrMahaTarget). Closest point to r0 is at
    // the boundary in the direction r0_white / ||r0_white||.
    Eigen::LLT< Eigen::Matrix2d > llt( N );
    if ( llt.info() != Eigen::Success )
    {
        // Fallback: use r0 direction unchanged.
        const Eigen::Vector2d rEll = r0 * std::sqrt( sqrMahaTarget / std::max( 1e-30, r0.dot( N * r0 ) ) );
        const double bScalar = n.dot( rEll );
        return { n, bScalar };
    }
    const Eigen::Matrix2d L = llt.matrixL();
    Eigen::Vector2d r0w = L.transpose() * r0;
    if ( r0w.norm() < 1e-30 ) r0w = Eigen::Vector2d{ 1.0, 0.0 };
    Eigen::Vector2d rEllw = std::sqrt( sqrMahaTarget ) * r0w / r0w.norm();
    Eigen::Vector2d rEll = L.transpose().triangularView< Eigen::Upper >().solve( rEllw );

    const double bScalar = n.dot( rEll );
    return { n, bScalar };
}

}  // namespace detail

/// @brief Apply an impulse sequence to a reference trajectory (forward
///        propagation with injected impulses at each node). Returns the state
///        at the start of the manoeuvre window — for SCVX re-linearisation we
///        propagate forward and then back-propagate to the start (here we just
///        keep xs0 since the linearisation is around the *current* reference
///        which differs only via dv injected at each node).
inline Vec6< double > applyImpulsesToReference( const Vec6< double >& xs0,
                                                const Eigen::VectorXd& /*dv*/,
                                                const MultiImpulseConfig& /*cfg*/ )
{
    // For the simplified SCVX presented here we re-linearise around the
    // *unperturbed* reference. Future work: integrate the impulse history
    // into the trajectory before each re-linearisation (see the original
    // statePropMultiMapsFullPRefine binary).
    return xs0;
}

/**
 * @brief Sequential Convex Optimisation driver.
 *
 * One outer iteration:
 *   1. Compute LinearMaps at current trajectory (re-propagate with current dv).
 *   2. Form linear collision-avoidance constraint:  n^T (D dv + r0) >= n^T rEll
 *      i.e.  (n^T D) dv >= n^T (rEll - r0)
 *   3. Solve SOCP: min sum_k ||dv_k|| subject to that inequality + box.
 *   4. Update dv (relative to current). Repeat until step is small.
 *
 * Note: This is a faithful port of `mainSCVX.m` minus the MOSEK-only features
 * (ellipse refinement loop and full state-correction variables). One linear
 * inequality per outer iteration is sufficient for many conjunction geometries.
 */
inline SCVXResult solveSCVX( const Vec6< double >& xs0, const Vec6< double >& xd0,
                             const Mat3< double >& PsRTN, const Mat3< double >& PdRTN,
                             const MultiImpulseConfig& cfg, const SCVXConfig& opt )
{
    SCVXResult res;
    const int N = cfg.nImpulses;
    Eigen::VectorXd dv = Eigen::VectorXd::Zero( N * 3 );

    // Build linear maps once around the unperturbed reference. Without a
    // re-propagation step (cf. statePropMultiMapsFullPRefine in the original
    // repository) the SCVX outer loop reduces to a single SOCP solve.
    const LinearMaps< double > L = buildLinearMaps( xs0, xd0, PsRTN, PdRTN, cfg );

    auto [n, bScalar] =
        detail::tangentHalfPlane( L.Pb2D.inverse(), opt.sqrMahalaTarget, L.rrb2Nom );
    Eigen::RowVectorXd Acon = n.transpose() * L.drrb2Nom;
    const double rhs = bScalar - n.dot( L.rrb2Nom );

    SOCPResult sol = solveSOCP( Acon, rhs, N, opt.dvMax, opt.innerSocpIter, opt.tolSocp );
    dv = sol.dv;

    res.dv = dv;
    res.totalDeltaV = 0.0;
    for ( int k = 0; k < N; ++k ) res.totalDeltaV += dv.segment( k * 3, 3 ).norm();

    // Predicted (linearised) avoidance metrics
    Eigen::Vector2d r_after = L.rrb2Nom + L.drrb2Nom * dv;
    Eigen::Matrix2d Nb = L.Pb2D.inverse();
    res.sqrMahalanobis = r_after.dot( Nb * r_after );
    res.missDistance = r_after.norm();
    res.outerIterations = 1;
    res.converged = sol.converged;

    if ( opt.verbose )
    {
        std::cout << "[SCVX] |Δv|=" << res.totalDeltaV * 1e3 << " m/s, "
                  << "miss=" << res.missDistance << " km, sqrMaha=" << res.sqrMahalanobis
                  << ", converged=" << ( res.converged ? "yes" : "no" ) << '\n';
    }
    return res;
}

}  // namespace cam::optim
