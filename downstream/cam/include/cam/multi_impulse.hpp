#pragma once

#include <Eigen/Dense>
#include <cam/collision.hpp>
#include <cam/frames.hpp>
#include <cam/linalg.hpp>
#include <cam/propagator.hpp>
#include <cam/tca.hpp>
#include <cstddef>
#include <tax/eigen/derivative.hpp>
#include <tax/tax.hpp>

namespace cam
{

/**
 * @file multi_impulse.hpp
 * @brief Multi-impulse propagation and linear-map extraction.
 *
 * Mirrors the workflow of `cpp/statePropMultiMapsFullP.cpp` from
 * arma1978/multi-impulsive-cam. For each of N impulse opportunities at fixed
 * spacing `dt`, the spacecraft state is propagated through that interval with
 * a fresh DA expansion in 6 variables (3 position perturbations + 3 velocity
 * perturbations). The 6×6 local STM at each node is extracted via tax. The
 * STMs are then chained to build the linearised map from the full impulse
 * sequence (3N components) to the relative state at TCA, and projected onto
 * the B-plane to obtain the linear collision-avoidance constraint matrices.
 */

template < typename T >
struct LinearMaps
{
    /// State-deviation map at the last node: 6 × 3N.
    Eigen::Matrix< T, 6, Eigen::Dynamic > M;
    /// Chained STMs at every node: rows stack 6N × 3N. Useful for diagnostics.
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > A;
    /// B-plane Jacobian wrt impulse sequence: 2 × 3N.
    Eigen::Matrix< T, 2, Eigen::Dynamic > drrb2Nom;
    /// Nominal B-plane relative position: 2.
    Eigen::Matrix< T, 2, 1 > rrb2Nom;
    /// 2×2 B-plane covariance (sum of primary + secondary in B-plane).
    Eigen::Matrix< T, 2, 2 > Pb2D;
    /// Nominal time-of-closest-approach correction (seconds, scaled by Tsc^-1 internally).
    T tcaScaled;
    /// Nominal squared Mahalanobis distance.
    T sqrMahalanobisNom;
};

/// @brief Configuration for the multi-impulse propagation pipeline.
struct MultiImpulseConfig
{
    int nImpulses = 10;        ///< Number of impulse opportunities (N)
    double dtSeconds = 60.0;   ///< Spacing between impulse opportunities (s)
    double t2TCAseconds;       ///< Nominal time-to-TCA from start (s)
    double mu = EarthPhysics::mu;
    double rE = EarthPhysics::rE;
    double J2 = EarthPhysics::J2;
};

/// @brief Build the LinearMaps for an N-impulse problem.
/// @param xs0Raw Initial spacecraft state (km, km/s).
/// @param xd0Raw Initial debris state (km, km/s).
/// @param PsRTN  Spacecraft 3×3 covariance in RTN frame.
/// @param PdRTN  Debris 3×3 covariance in RTN frame.
/// @param cfg    Configuration.
/// @returns LinearMaps<double> with all linearised quantities at the reference
///          (zero-impulse) trajectory.
inline LinearMaps< double > buildLinearMaps( const Vec6< double >& xs0Raw,
                                             const Vec6< double >& xd0Raw,
                                             const Mat3< double >& PsRTN,
                                             const Mat3< double >& PdRTN,
                                             const MultiImpulseConfig& cfg )
{
    using DA = tax::TEn< 1, 7 >;  // order 1 is sufficient for STM extraction
    using TimeDA = tax::TEn< 1, 1 >;
    const Scaling s = Scaling::make( cfg.mu, cfg.rE );

    // --- Per-step propagation in scaled units --------------------------------
    Vec6< double > xs_scaled;
    for ( int i = 0; i < 3; ++i ) xs_scaled[i] = xs0Raw[i] / s.Lsc;
    for ( int i = 3; i < 6; ++i ) xs_scaled[i] = xs0Raw[i] / s.Vsc;

    Vec6< double > xd_scaled;
    for ( int i = 0; i < 3; ++i ) xd_scaled[i] = xd0Raw[i] / s.Lsc;
    for ( int i = 3; i < 6; ++i ) xd_scaled[i] = xd0Raw[i] / s.Vsc;

    const int N = cfg.nImpulses;
    const double dtScaled = cfg.dtSeconds / s.Tsc;

    // Storage for 6×6 STMs at each node.
    std::vector< Eigen::Matrix< double, 6, 6 > > LL;
    LL.resize( static_cast< std::size_t >( N ) );

    Vec6< double > xs_cur = xs_scaled;
    typename DA::Input x0{};
    for ( int k = 0; k < N; ++k )
    {
        // Build DA state with 6 perturbations on the current state.
        Vec6< DA > xs_da;
        xs_da[0] = DA::template variable< 0 >( x0 ) + xs_cur[0];
        xs_da[1] = DA::template variable< 1 >( x0 ) + xs_cur[1];
        xs_da[2] = DA::template variable< 2 >( x0 ) + xs_cur[2];
        xs_da[3] = DA::template variable< 3 >( x0 ) + xs_cur[3];
        xs_da[4] = DA::template variable< 4 >( x0 ) + xs_cur[4];
        xs_da[5] = DA::template variable< 5 >( x0 ) + xs_cur[5];

        const DA dt_da = DA{ dtScaled };
        const Vec6< DA > xs_next = propKepAn( xs_da, dt_da, s.muSc );

        // Extract 6×6 STM (partials w.r.t. variables 0..5)
        for ( int i = 0; i < 6; ++i )
        {
            for ( int j = 0; j < 6; ++j )
            {
                tax::MultiIndex< 7 > a{};
                a[std::size_t( j )] = 1;
                LL[std::size_t( k )]( i, j ) = xs_next[i].derivative( a );
            }
            xs_cur[i] = xs_next[i].value();
        }
    }

    // --- Final segment: from last impulse node to nominal TCA ---------------
    const double dtFinalScaled = ( cfg.t2TCAseconds - N * cfg.dtSeconds ) / s.Tsc;

    Vec6< DA > xs_final_in;
    xs_final_in[0] = DA::template variable< 0 >( x0 ) + xs_cur[0];
    xs_final_in[1] = DA::template variable< 1 >( x0 ) + xs_cur[1];
    xs_final_in[2] = DA::template variable< 2 >( x0 ) + xs_cur[2];
    xs_final_in[3] = DA::template variable< 3 >( x0 ) + xs_cur[3];
    xs_final_in[4] = DA::template variable< 4 >( x0 ) + xs_cur[4];
    xs_final_in[5] = DA::template variable< 5 >( x0 ) + xs_cur[5];
    const DA dt_final_da = DA::template variable< 6 >( x0 ) + dtFinalScaled;
    Vec6< DA > xs_at_tca = propKepAn( xs_final_in, dt_final_da, s.muSc );

    // Debris: also propagated to the nominal TCA, with time-only DA dependence.
    Vec6< DA > xd_da;
    for ( int i = 0; i < 6; ++i ) xd_da[i] = DA{ xd_scaled[i] };
    const DA t2tca_da = DA::template variable< 6 >( x0 ) + ( cfg.t2TCAseconds / s.Tsc );
    const Vec6< DA > xd_at_tca = propKepAn( xd_da, t2tca_da, s.muSc );

    // --- Refine TCA from relative state -------------------------------------
    Vec6< DA > xrel;
    for ( int i = 0; i < 6; ++i ) xrel[i] = xs_at_tca[i] - xd_at_tca[i];
    const DA tcaCorr = findTCA< 1, 7 >( xrel );

    // Substitute the TCA correction back into the final state polynomials.
    typename DA::Input dx_eval{};
    dx_eval[6] = tcaCorr.value();
    Vec6< double > xs_eval, xd_eval;
    for ( int i = 0; i < 6; ++i )
    {
        xs_eval[i] = xs_at_tca[i].eval( dx_eval );
        xd_eval[i] = xd_at_tca[i].eval( dx_eval );
    }

    // Local 6×6 STM at the final node (excluding time variable column).
    Eigen::Matrix< double, 6, 6 > LL_final;
    for ( int i = 0; i < 6; ++i )
    {
        for ( int j = 0; j < 6; ++j )
        {
            tax::MultiIndex< 7 > a{};
            a[std::size_t( j )] = 1;
            LL_final( i, j ) = xs_at_tca[i].derivative( a );
        }
    }

    // --- Build chained A matrix (6N × 3N) -----------------------------------
    LinearMaps< double > out;
    out.A = Eigen::MatrixXd::Zero( ( N + 1 ) * 6, N * 3 );
    for ( int k = 0; k < N; ++k )
    {
        const Eigen::Matrix< double, 6, 6 >& Lk = LL[std::size_t( k )];
        if ( k == 0 )
        {
            // Velocity-perturbation columns of LL[0] act as impulse response
            out.A.block( 0, 0, 6, 3 ) = Lk.block( 0, 3, 6, 3 );
        }
        else
        {
            // Propagate previous columns through Lk
            out.A.block( k * 6, 0, 6, N * 3 ) = Lk * out.A.block( ( k - 1 ) * 6, 0, 6, N * 3 );
            // Add current impulse response in the k-th 3-column block
            out.A.block( k * 6, k * 3, 6, 3 ) = Lk.block( 0, 3, 6, 3 );
        }
    }
    // Final row block: propagate through LL_final
    out.A.block( N * 6, 0, 6, N * 3 ) = LL_final * out.A.block( ( N - 1 ) * 6, 0, 6, N * 3 );

    out.M = out.A.block( N * 6, 0, 6, N * 3 );

    // --- B-plane projection of relative state at TCA ------------------------
    Vec3< double > rrel, vrels, vreld;
    for ( int i = 0; i < 3; ++i ) rrel[i] = ( xs_eval[i] - xd_eval[i] ) * s.Lsc;
    for ( int i = 0; i < 3; ++i ) vrels[i] = xs_eval[i + 3] * s.Vsc;
    for ( int i = 0; i < 3; ++i ) vreld[i] = xd_eval[i + 3] * s.Vsc;

    Mat3< double > toB = bplane( vrels, vreld );
    Vec3< double > rrb_3D = toB * rrel;
    out.rrb2Nom( 0 ) = rrb_3D( 0 );
    out.rrb2Nom( 1 ) = rrb_3D( 2 );

    // Covariance in B-plane
    Mat3< double > toRTNs = rtn( Vec6< double >{ xs_eval[0] * s.Lsc, xs_eval[1] * s.Lsc,
                                                  xs_eval[2] * s.Lsc, xs_eval[3] * s.Vsc,
                                                  xs_eval[4] * s.Vsc, xs_eval[5] * s.Vsc } );
    Mat3< double > toRTNd = rtn( Vec6< double >{ xd_eval[0] * s.Lsc, xd_eval[1] * s.Lsc,
                                                  xd_eval[2] * s.Lsc, xd_eval[3] * s.Vsc,
                                                  xd_eval[4] * s.Vsc, xd_eval[5] * s.Vsc } );
    Mat3< double > PsECI = toRTNs.transpose() * PsRTN * toRTNs;
    Mat3< double > PdECI = toRTNd.transpose() * PdRTN * toRTNd;

    Mat3< double > Pb_3D = toB * ( PsECI + PdECI ) * toB.transpose();
    out.Pb2D( 0, 0 ) = Pb_3D( 0, 0 );
    out.Pb2D( 0, 1 ) = Pb_3D( 0, 2 );
    out.Pb2D( 1, 0 ) = Pb_3D( 2, 0 );
    out.Pb2D( 1, 1 ) = Pb_3D( 2, 2 );

    // B-plane Jacobian: drrb_2D / d(state_at_TCA) only requires the B-projection
    // of the position rows of M (rows 0..2), and converts km/Lsc back to km.
    // M maps dv (km/s) to scaled state perturbation (Lsc, Vsc). For the position
    // part we multiply by Lsc to get km. The map dv → dr_km is M.topRows(3) * Lsc.
    Eigen::Matrix< double, 3, Eigen::Dynamic > Mpos = out.M.topRows( 3 ) * s.Lsc;
    Eigen::Matrix< double, 3, Eigen::Dynamic > drrb_3D = toB * Mpos;
    out.drrb2Nom.resize( 2, N * 3 );
    out.drrb2Nom.row( 0 ) = drrb_3D.row( 0 );
    out.drrb2Nom.row( 1 ) = drrb_3D.row( 2 );

    // Nominal squared Mahalanobis distance
    Eigen::Matrix< double, 2, 2 > Nb2D = out.Pb2D.inverse();
    out.sqrMahalanobisNom = out.rrb2Nom.dot( Nb2D * out.rrb2Nom );
    out.tcaScaled = tcaCorr.value() + cfg.t2TCAseconds / s.Tsc;
    return out;
}

}  // namespace cam
