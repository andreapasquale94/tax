#pragma once

#include <Eigen/Dense>
#include <cam/multi_impulse.hpp>
#include <cmath>
#include <iostream>

namespace cam::optim
{

/// @brief Configuration for the penalty-NLP solver.
struct NLPConfig
{
    double sqrMahalaTarget = 25.0;  ///< Target squared Mahalanobis distance
    double dvMax = 1.0;             ///< Per-component impulse bound
    int maxIter = 200;
    double tol = 1e-8;
    double rho0 = 1.0;       ///< Initial penalty weight
    double rhoGrowth = 4.0;
    int outerLoops = 8;
    bool verbose = false;
};

struct NLPResult
{
    Eigen::VectorXd dv;
    double objective = 0.0;
    double sqrMahalanobis = 0.0;
    int iterations = 0;
    bool converged = false;
};

/**
 * @brief Solve the multi-impulse CAM problem as an NLP via quadratic penalty +
 *        projected L-BFGS-B-like gradient descent on a smoothed objective.
 *
 * Mirrors the role of `mainNLP.m` (which calls `fmincon` with `nncon`):
 *   minimize    sum_k ||dv_k||
 *   subject to  d^2(dv) >= sqrMahaTarget        (nonlinear constraint)
 *               |dv_i| <= dvMax                  componentwise
 *
 * The Mahalanobis distance squared as a function of dv is taken from the
 * quadratic form provided by `LinearMaps`:
 *      d²(dv) = c + 2·L·dv + dv'·Q·dv
 * where Q = M' B' Nb B M, L = c1' C M + c2' B M, c = c1' c2 (precomputed).
 */
inline NLPResult solveNLP( const LinearMaps< double >& maps, const NLPConfig& opt )
{
    const Eigen::Index dim = maps.M.cols();
    const int N = static_cast< int >( dim / 3 );

    // Quadratic form: d²(dv) = c + 2 L·dv + dv'·Q·dv
    Eigen::Matrix< double, 2, 2 > Nb = maps.Pb2D.inverse();
    Eigen::Matrix< double, 2, Eigen::Dynamic > BM = maps.drrb2Nom;  // 2 x 3N
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > Q = BM.transpose() * Nb * BM;
    Eigen::Matrix< double, 1, Eigen::Dynamic > Lvec = maps.rrb2Nom.transpose() * Nb * BM;
    const double cConst = maps.sqrMahalanobisNom;

    NLPResult res;
    res.dv = Eigen::VectorXd::Zero( dim );
    Eigen::VectorXd dv = res.dv;

    auto sqrMaha = [&]( const Eigen::VectorXd& x ) {
        return cConst + 2.0 * Lvec.dot( x ) + x.dot( Q * x );
    };
    auto sqrMahaGrad = [&]( const Eigen::VectorXd& x ) {
        return 2.0 * Lvec.transpose() + 2.0 * Q * x;
    };

    const double eps = 1e-9;
    double rho = opt.rho0;
    int totalIter = 0;

    auto project = [&]( Eigen::VectorXd& x ) {
        for ( int i = 0; i < x.size(); ++i ) x( i ) = std::clamp( x( i ), -opt.dvMax, opt.dvMax );
    };

    for ( int outer = 0; outer < opt.outerLoops; ++outer )
    {
        double alpha = 1.0;
        for ( int it = 0; it < opt.maxIter; ++it )
        {
            ++totalIter;
            // Smoothed objective: sum_k sqrt(||dv_k||^2 + eps^2)
            // + (rho/2) * max(0, sqrMahaTarget - sqrMaha(dv))^2
            const double dminus = opt.sqrMahalaTarget - sqrMaha( dv );

            Eigen::VectorXd grad( dim );
            for ( int k = 0; k < N; ++k )
            {
                Eigen::Vector3d dvk = dv.segment( k * 3, 3 );
                const double n = std::sqrt( dvk.squaredNorm() + eps * eps );
                grad.segment( k * 3, 3 ) = dvk / n;
            }
            if ( dminus > 0.0 ) grad += rho * dminus * sqrMahaGrad( dv );

            // Backtracking line search
            double objCurrent = 0.0;
            for ( int k = 0; k < N; ++k )
                objCurrent +=
                    std::sqrt( dv.segment( k * 3, 3 ).squaredNorm() + eps * eps );
            if ( dminus > 0.0 ) objCurrent += 0.5 * rho * dminus * dminus;

            Eigen::VectorXd dvTrial;
            bool accepted = false;
            for ( int ls = 0; ls < 30; ++ls )
            {
                dvTrial = dv - alpha * grad;
                project( dvTrial );

                double objTrial = 0.0;
                for ( int k = 0; k < N; ++k )
                    objTrial +=
                        std::sqrt( dvTrial.segment( k * 3, 3 ).squaredNorm() + eps * eps );
                const double dminusT = std::max( 0.0, opt.sqrMahalaTarget - sqrMaha( dvTrial ) );
                objTrial += 0.5 * rho * dminusT * dminusT;

                if ( objTrial <= objCurrent - 1e-4 * alpha * grad.squaredNorm() )
                {
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if ( !accepted ) break;

            const double step = ( dvTrial - dv ).norm();
            dv = dvTrial;
            if ( step < opt.tol ) break;
            alpha = std::min( 1.0, 2.0 * alpha );
        }

        const double sm = sqrMaha( dv );
        if ( sm >= opt.sqrMahalaTarget - 1e-4 ) break;
        rho *= opt.rhoGrowth;
    }

    res.dv = dv;
    res.iterations = totalIter;
    for ( int k = 0; k < N; ++k ) res.objective += dv.segment( k * 3, 3 ).norm();
    res.sqrMahalanobis = sqrMaha( dv );
    res.converged = res.sqrMahalanobis >= opt.sqrMahalaTarget - 1e-3;

    if ( opt.verbose )
        std::cout << "[NLP] Δv=" << res.objective << " sqrMaha=" << res.sqrMahalanobis
                  << " iters=" << res.iterations << '\n';
    return res;
}

}  // namespace cam::optim
