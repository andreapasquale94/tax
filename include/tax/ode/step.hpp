#pragma once

#include <utility>

#include <tax/ode/stepsize.hpp>
#include <tax/eigen/eval.hpp>

namespace tax::ode
{

/**
 * @brief Result of a single Taylor integration step.
 * @tparam Poly Taylor polynomial type: `TE<N>` for scalar, `Eigen::Matrix<TE<N>,D,1>` for vector.
 * @tparam T Scalar coefficient type.
 */
template < typename Poly, typename T >
struct StepResult
{
    Poly p;  ///< Taylor polynomial of the solution, centred at the current time.
    T h;     ///< Recommended step-size magnitude (positive).
};

// =============================================================================
// Scalar step
// =============================================================================

/**
 * @brief Compute a single Taylor step for a scalar ODE dx/dt = f(x, t).
 * @details Builds the Taylor polynomial of x(tc + τ) by iteratively applying
 *   x[k+1] = f(x,t)[k] / (k+1), then computes the adaptive step size.
 *   The returned polynomial can be evaluated at any τ via `result.p.eval(τ)`.
 * @tparam N Taylor expansion order.
 * @param f Right-hand side: callable `f(x, t)` returning the derivative.
 * @param x0 Current state value.
 * @param tc Current time.
 * @param abstol Absolute tolerance for step-size control.
 * @return StepResult with the solution polynomial and recommended step size.
 */
template < int N, typename F, typename T = double >
[[nodiscard]] StepResult< TruncatedExpansionT< T, N, 1 >, T >
step( F&& f, T x0, T tc, T abstol )
{
    using TTE = TruncatedExpansionT< T, N, 1 >;

    TTE t_da{};
    t_da[0] = tc;
    if constexpr ( N >= 1 ) t_da[1] = T{ 1 };

    TTE x_da{};
    x_da[0] = x0;

    for ( int k = 0; k < N; ++k )
    {
        TTE dx = f( x_da, t_da );
        x_da[k + 1] = dx[k] / T( k + 1 );
    }

    auto h = stepsize( x_da, abstol );
    return { std::move( x_da ), h };
}

// =============================================================================
// Vector step
// =============================================================================

/**
 * @brief Compute a single Taylor step for a vector ODE f(dx, x, t).
 * @details Builds Eigen vector of Taylor polynomials representing x(tc + τ),
 *   then computes the adaptive step size (minimum across components).
 *   Evaluate the result at any τ via `tax::eval(result.p, τ)`.
 * @tparam N Taylor expansion order.
 * @param f Right-hand side: callable `f(dx, x, t)` writing derivatives into dx.
 * @param x0 Current state vector.
 * @param tc Current time.
 * @param abstol Absolute tolerance for step-size control.
 * @return StepResult with the solution polynomial vector and recommended step size.
 */
template < int N, typename F, typename T, int D >
[[nodiscard]] StepResult< Eigen::Matrix< TruncatedExpansionT< T, N, 1 >, D, 1 >, T >
step( F&& f, const Eigen::Matrix< T, D, 1 >& x0, T tc, T abstol )
{
    using TTE = TruncatedExpansionT< T, N, 1 >;
    using VecTTE = Eigen::Matrix< TTE, D, 1 >;

    const Eigen::Index dim = x0.size();

    TTE t_da{};
    t_da[0] = tc;
    if constexpr ( N >= 1 ) t_da[1] = T{ 1 };

    VecTTE x_da( dim );
    for ( Eigen::Index i = 0; i < dim; ++i )
    {
        x_da( i ) = TTE{};
        x_da( i )[0] = x0( i );
    }

    VecTTE dx( dim );
    for ( int k = 0; k < N; ++k )
    {
        f( dx, x_da, t_da );
        for ( Eigen::Index i = 0; i < dim; ++i ) x_da( i )[k + 1] = dx( i )[k] / T( k + 1 );
    }

    auto h = stepsize( x_da, abstol );
    return { std::move( x_da ), h };
}

}  // namespace tax::ode
