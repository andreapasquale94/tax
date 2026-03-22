#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <tax/ads/ads_tree.hpp>
#include <tax/ads/box.hpp>
#include <tax/ode/step.hpp>
#include <tax/tte.hpp>
#include <tax/utils/combinatorics.hpp>

namespace tax::ode
{

// =============================================================================
// FlowMap: state-vector polynomial stored in each ADS leaf
// =============================================================================

/**
 * @brief Polynomial flow map from initial conditions to propagated state.
 *
 * Each leaf of the ADS tree holds one FlowMap.  The `state` member is a
 * vector of multivariate Taylor polynomials in the normalised initial-
 * condition deviations δ ∈ [−1, 1]^D.
 *
 * @tparam P  DA expansion order in the initial-condition variables.
 * @tparam D  State-space dimension (= number of DA variables).
 */
template < int P, int D >
struct FlowMap
{
    using DA    = TEn< P, D >;
    using Input = std::array< double, D >;

    Eigen::Matrix< DA, D, 1 > state{};
};

// =============================================================================
// Internal helpers for DA-valued time-Taylor coefficients
// =============================================================================

namespace detail
{

/// @brief Infinity norm of a DA polynomial (max absolute coefficient).
template < typename T, int P, int M >
[[nodiscard]] double infNorm( const TruncatedTaylorExpansionT< T, P, M >& x ) noexcept
{
    double n = 0.0;
    for ( std::size_t i = 0; i < TruncatedTaylorExpansionT< T, P, M >::nCoefficients; ++i )
        n = std::max( n, std::abs( x[i] ) );
    return n;
}

/// @brief Adaptive step size from the last two DA-valued Taylor coefficients.
/// Generalises the Jorba–Zou (2005) criterion to polynomial-valued
/// coefficients by replacing the scalar absolute value with the infinity norm.
template < int P, int M, int N >
[[nodiscard]] double stepsizeDa(
    const TruncatedTaylorExpansionT< TEn< P, M >, N, 1 >& x, double abstol ) noexcept
{
    double h = std::numeric_limits< double >::infinity();

    if constexpr ( N >= 2 )
    {
        const double c = infNorm( x[N - 1] );
        if ( c > 0.0 ) h = std::min( h, std::pow( abstol / c, 1.0 / ( N - 1 ) ) );
    }
    {
        const double c = infNorm( x[N] );
        if ( c > 0.0 ) h = std::min( h, std::pow( abstol / c, 1.0 / N ) );
    }
    return h;
}

/// @brief Vector version: minimum step size across all state components.
template < int P, int M, int N, int D >
[[nodiscard]] double stepsizeDa(
    const Eigen::Matrix< TruncatedTaylorExpansionT< TEn< P, M >, N, 1 >, D, 1 >& x,
    double abstol ) noexcept
{
    double h = std::numeric_limits< double >::infinity();
    for ( Eigen::Index i = 0; i < x.size(); ++i )
        h = std::min( h, stepsizeDa< P, M, N >( x( i ), abstol ) );
    return h;
}

/// @brief Evaluate a DA-valued time-TTE at a scalar displacement (Horner).
/// Each coefficient is a DA polynomial; the displacement is a plain double,
/// so the computation is polynomial-scalar multiply + polynomial addition.
template < typename DA, int N >
[[nodiscard]] DA evalAtScalar(
    const TruncatedTaylorExpansionT< DA, N, 1 >& poly, double dt ) noexcept
{
    DA result = poly[N];
    for ( int i = N - 1; i >= 0; --i )
    {
        result *= typename DA::scalar_type( dt );
        result += poly[i];
    }
    return result;
}

/// @brief Vector version: evaluate each component at the same scalar dt.
template < typename DA, int N, int D >
[[nodiscard]] Eigen::Matrix< DA, D, 1 > evalAtScalar(
    const Eigen::Matrix< TruncatedTaylorExpansionT< DA, N, 1 >, D, 1 >& poly,
    double dt ) noexcept
{
    Eigen::Matrix< DA, D, 1 > result( poly.size() );
    for ( Eigen::Index i = 0; i < poly.size(); ++i )
        result( i ) = evalAtScalar< DA, N >( poly( i ), dt );
    return result;
}

/// @brief Truncation error of a DA state vector.
///
/// Returns the infinity norm of all degree-P coefficients across every
/// component of the state.  Large values indicate that the polynomial
/// approximation of the flow map is degrading.
template < int P, int D >
[[nodiscard]] double truncationError(
    const Eigen::Matrix< TEn< P, D >, D, 1 >& state ) noexcept
{
    double err = 0.0;
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        const auto& poly = state( i );
        for ( std::size_t j = 0; j < TEn< P, D >::nCoefficients; ++j )
        {
            const auto alpha = tax::detail::unflatIndex< D >( j );
            if ( tax::detail::totalDegree< D >( alpha ) == P )
                err = std::max( err, std::abs( poly[j] ) );
        }
    }
    return err;
}

/// @brief Choose the initial-condition variable that contributes most to the
///        truncation error, following Wittig et al. (2015).
template < int P, int D >
[[nodiscard]] int bestSplitDim(
    const Eigen::Matrix< TEn< P, D >, D, 1 >& state ) noexcept
{
    std::array< double, D > scores{};
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        const auto& poly = state( i );
        for ( std::size_t j = 0; j < TEn< P, D >::nCoefficients; ++j )
        {
            const auto alpha = tax::detail::unflatIndex< D >( j );
            if ( tax::detail::totalDegree< D >( alpha ) == P )
                for ( int k = 0; k < D; ++k )
                    if ( alpha[k] > 0 )
                        scores[k] += std::abs( poly[j] );
        }
    }
    return static_cast< int >(
        std::max_element( scores.begin(), scores.end() ) - scores.begin() );
}

}  // namespace detail

// =============================================================================
// Single Taylor step with DA-valued state
// =============================================================================

/**
 * @brief Compute one Taylor step for a vector ODE with DA-expanded state.
 *
 * The state components are multivariate polynomials (TEn<P,D>) representing
 * a neighbourhood of initial conditions.  Time is a plain scalar.
 *
 * @tparam N  Taylor expansion order in time.
 * @tparam P  DA expansion order in the initial-condition variables.
 * @tparam D  State-space dimension (= number of DA variables).
 * @param f      Right-hand side `f(dx, x, t)`.
 * @param x0     Current DA state vector.
 * @param tc     Current time (scalar).
 * @param abstol Absolute tolerance for step-size control.
 */
template < int N, int P, int D, typename F >
[[nodiscard]] StepResult<
    Eigen::Matrix< TruncatedTaylorExpansionT< TEn< P, D >, N, 1 >, D, 1 >, double >
stepDa( F&& f, const Eigen::Matrix< TEn< P, D >, D, 1 >& x0, double tc, double abstol )
{
    using DA     = TEn< P, D >;
    using TTE    = TruncatedTaylorExpansionT< DA, N, 1 >;
    using VecTTE = Eigen::Matrix< TTE, D, 1 >;

    const Eigen::Index dim = x0.size();

    // Time variable: t(τ) = tc + τ, with constant-DA coefficients.
    TTE t_da{};
    t_da[0] = DA( tc );
    if constexpr ( N >= 1 ) t_da[1] = DA( 1.0 );

    // State variables: x_da[0] = x0 (DA polynomial per component).
    VecTTE x_da( dim );
    for ( Eigen::Index i = 0; i < dim; ++i )
    {
        x_da( i ) = TTE{};
        x_da( i )[0] = x0( i );
    }

    // Picard iteration: x_da[k+1] = f(x_da, t_da)[k] / (k+1).
    VecTTE dx( dim );
    for ( int k = 0; k < N; ++k )
    {
        f( dx, x_da, t_da );
        for ( Eigen::Index i = 0; i < dim; ++i )
        {
            x_da( i )[k + 1] = dx( i )[k];
            x_da( i )[k + 1] /= double( k + 1 );
        }
    }

    auto h = detail::stepsizeDa< P, D, N >( x_da, abstol );
    return { std::move( x_da ), h };
}

// =============================================================================
// Build DA initial conditions from a box
// =============================================================================

/**
 * @brief Create DA-expanded initial state from a box of initial conditions.
 *
 * Component `i` is the polynomial `center[i] + halfWidth[i] * δ_i`
 * where δ ∈ [−1, 1]^D is the normalised deviation.
 */
template < int P, int D >
[[nodiscard]] Eigen::Matrix< TEn< P, D >, D, 1 >
makeDaState( const Box< double, D >& box )
{
    using DA = TEn< P, D >;

    Eigen::Matrix< DA, D, 1 > x0( D );
    for ( int i = 0; i < D; ++i )
    {
        typename DA::Data c{};
        c[0] = box.center[i];
        if constexpr ( P >= 1 )
        {
            MultiIndex< D > ei{};
            ei[i] = 1;
            c[tax::detail::flatIndex< D >( ei )] = box.halfWidth[i];
        }
        x0( i ) = DA{ c };
    }
    return x0;
}

// =============================================================================
// Propagate a single subdomain from t0 to tmax
// =============================================================================

/**
 * @brief Integrate a vector ODE with DA-expanded state over one subdomain.
 *
 * Builds the DA initial conditions from @p box and advances from @p t0 to
 * @p tmax with adaptive Taylor stepping.  Returns the final DA state vector.
 *
 * @tparam N  Taylor expansion order in time.
 * @tparam P  DA expansion order in state.
 * @tparam D  State-space dimension.
 */
template < int N, int P, int D, typename F >
[[nodiscard]] Eigen::Matrix< TEn< P, D >, D, 1 >
propagateBox( F&& f, const Box< double, D >& box, double t0, double tmax,
              double abstol, int maxsteps = 500 )
{
    auto xc        = makeDaState< P, D >( box );
    double tc      = t0;
    const double s = tmax >= t0 ? 1.0 : -1.0;

    for ( int step = 0; step < maxsteps; ++step )
    {
        if ( s * ( tmax - tc ) <= 0.0 ) break;

        auto [p, h] = stepDa< N, P, D >( f, xc, tc, abstol );
        if ( h <= 0.0 ) break;

        const double dt = s * std::min( h, std::abs( tmax - tc ) );
        xc = detail::evalAtScalar< TEn< P, D >, N >( p, dt );
        tc += dt;
    }

    return xc;
}

// =============================================================================
// ADS-integrated ODE propagation
// =============================================================================

/**
 * @brief Integrate a vector ODE with Automatic Domain Splitting.
 *
 * The initial-condition domain @p x0_box is propagated from @p t0 to @p tmax.
 * If the DA approximation of the flow map degrades beyond @p ads_tol, the
 * domain is bisected along the variable that contributes most to the
 * truncation error (Wittig et al. 2015), and each half is re-propagated
 * independently.
 *
 * @tparam N  Taylor expansion order in time.
 * @tparam P  DA expansion order in the initial-condition variables.
 * @tparam D  State-space dimension (= number of DA variables).
 * @param f             Right-hand side `f(dx, x, t)`.
 * @param x0_box        Initial-condition domain (centre + half-widths).
 * @param t0            Initial time.
 * @param tmax          Final time.
 * @param step_tol      Absolute tolerance for adaptive time stepping.
 * @param ads_tol       Truncation-error tolerance for ADS splitting.
 * @param ads_max_depth Maximum number of recursive bisections from root.
 * @param maxsteps      Maximum integration steps per subdomain.
 * @return ADS tree whose done leaves contain the piecewise flow map.
 */
template < int N, int P, typename F, int D >
[[nodiscard]] AdsTree< FlowMap< P, D > > integrateAds(
    F&& f, const Box< double, D >& x0_box, double t0, double tmax,
    double step_tol, double ads_tol, int ads_max_depth = 30, int maxsteps = 500 )
{
    using FM   = FlowMap< P, D >;
    using Tree = AdsTree< FM >;

    auto evaluate_box = [&]( const Box< double, D >& box ) -> FM {
        return FM{ propagateBox< N, P, D >( f, box, t0, tmax, step_tol, maxsteps ) };
    };

    Tree tree;
    tree.addLeaf( evaluate_box( x0_box ), x0_box );

    std::vector< int > depth( 1, 0 );

    while ( !tree.empty() )
    {
        const int    idx = tree.pop();
        const auto&  lf  = tree.node( idx ).leaf();
        const double err = detail::truncationError< P, D >( lf.tte.state );
        const int    d   = depth[idx];

        if ( err < ads_tol || d >= ads_max_depth )
        {
            tree.markDone( idx );
        }
        else
        {
            const int dim = detail::bestSplitDim< P, D >( lf.tte.state );
            auto [lb, rb] = lf.box.split( dim );
            auto lt        = evaluate_box( lb );
            auto rt        = evaluate_box( rb );

            auto [li, ri] = tree.split( idx, dim, std::move( lt ), std::move( rt ) );

            if ( static_cast< int >( depth.size() ) <= ri )
                depth.resize( ri + 1, 0 );
            depth[li] = d + 1;
            depth[ri] = d + 1;
        }
    }

    return tree;
}

}  // namespace tax::ode
