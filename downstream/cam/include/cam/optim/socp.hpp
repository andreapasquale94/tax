#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace cam::optim
{

/// @brief Result of `solveSOCP`.
struct SOCPResult
{
    Eigen::VectorXd dv;            ///< Optimal decision (3N entries)
    double objective = 0.0;        ///< sum_k ||dv_k||
    double constraintViolation = 0.0;  ///< max(0, b - a·dv)
    int iterations = 0;
    bool converged = false;
};

/**
 * @brief Solve a structured second-order cone program by smoothed projected
 *        gradient with augmented-Lagrangian penalty on the linear inequality.
 *
 * Problem:
 *   minimize    sum_{k=1..N} ||dv_k||_2          (dv_k in R^3)
 *   subject to  A * dv >= b                       (single linear inequality)
 *               |dv_i| <= dvmax  componentwise
 *
 * `A` has shape (1, 3N); `b` is a scalar. The cone constraint
 * `||dv_k|| <= dv_mag_k` is folded into the smoothed L2 objective; box
 * constraints are enforced by component-wise projection.
 */
inline SOCPResult solveSOCP( const Eigen::RowVectorXd& A, double b, int N, double dvmax,
                             int maxIter = 1000, double tol = 1e-8 )
{
    const int dim = N * 3;
    SOCPResult result;
    result.dv = Eigen::VectorXd::Zero( dim );

    if ( A.size() != dim )
    {
        // Trivial: return zero.
        result.objective = 0.0;
        result.constraintViolation = std::max( 0.0, b );
        return result;
    }

    const double eps = 1e-9;        // Smoothing parameter for ||dv_k||
    double rho = 1.0;               // Penalty weight (grows over outer loop)
    const double rhoGrowth = 4.0;
    const int outerIters = 8;

    Eigen::VectorXd dv = result.dv;

    auto project = [&]( Eigen::VectorXd& x ) {
        for ( int i = 0; i < x.size(); ++i )
            x( i ) = std::clamp( x( i ), -dvmax, dvmax );
    };

    auto sumOfNorms = [&]( const Eigen::VectorXd& x ) {
        double s = 0.0;
        for ( int k = 0; k < N; ++k ) s += x.segment( k * 3, 3 ).norm();
        return s;
    };

    int totalIter = 0;

    for ( int outer = 0; outer < outerIters; ++outer )
    {
        // Inner: minimize  sum_k sqrt(||dv_k||^2 + eps^2) + rho/2 * max(0, b - A dv)^2
        //        with box projection.
        double alpha = 1.0;  // step size

        for ( int inner = 0; inner < maxIter; ++inner )
        {
            ++totalIter;

            // Gradient of smoothed objective
            Eigen::VectorXd grad( dim );
            for ( int k = 0; k < N; ++k )
            {
                Eigen::Vector3d dvk = dv.segment( k * 3, 3 );
                const double n = std::sqrt( dvk.squaredNorm() + eps * eps );
                grad.segment( k * 3, 3 ) = dvk / n;
            }

            // Penalty gradient
            const double Adv = A * dv;
            const double slack = b - Adv;
            if ( slack > 0.0 ) grad -= rho * slack * A.transpose();

            // Backtracking projected-gradient step
            Eigen::VectorXd dvTrial;
            const double objCurrent = [&] {
                double f = 0.0;
                for ( int k = 0; k < N; ++k )
                    f += std::sqrt( dv.segment( k * 3, 3 ).squaredNorm() + eps * eps );
                if ( slack > 0.0 ) f += 0.5 * rho * slack * slack;
                return f;
            }();

            double objTrial = 0.0;
            bool accepted = false;
            for ( int ls = 0; ls < 30; ++ls )
            {
                dvTrial = dv - alpha * grad;
                project( dvTrial );

                objTrial = 0.0;
                for ( int k = 0; k < N; ++k )
                    objTrial +=
                        std::sqrt( dvTrial.segment( k * 3, 3 ).squaredNorm() + eps * eps );
                const double slackT = std::max( 0.0, b - A * dvTrial );
                objTrial += 0.5 * rho * slackT * slackT;

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

            if ( step < tol ) break;

            // Reset step size for next iteration
            alpha = std::min( 1.0, 2.0 * alpha );
        }

        const double slackFinal = std::max( 0.0, b - A * dv );
        if ( slackFinal < tol ) break;
        rho *= rhoGrowth;
    }

    result.dv = dv;
    result.objective = sumOfNorms( dv );
    result.constraintViolation = std::max( 0.0, b - A * dv );
    result.iterations = totalIter;
    result.converged = result.constraintViolation < 1e-4;
    return result;
}

}  // namespace cam::optim
