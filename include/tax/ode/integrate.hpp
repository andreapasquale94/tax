#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include <tax/ode/solution.hpp>
#include <tax/ode/step.hpp>

namespace tax::ode
{

// =============================================================================
// Scalar ODE – adaptive stepping
// =============================================================================

/**
 * @brief Integrate scalar ODE dx/dt = f(x, t) with adaptive step size.
 * @tparam N Taylor expansion order.
 * @param f Right-hand side: callable `f(x, t)` returning the derivative.
 * @param x0 Initial state.
 * @param t0 Initial time.
 * @param tmax Final time.
 * @param abstol Absolute tolerance for step-size control.
 * @param maxsteps Maximum number of integration steps.
 * @return TaylorSolution with dense-output polynomials.
 */
template < int N, typename F, typename T = double >
[[nodiscard]] TaylorSolution< N, T, T > integrate( F&& f, T x0, T t0, T tmax, T abstol,
                                                    int maxsteps = 500 )
{
    TaylorSolution< N, T, T > sol;
    sol.t.reserve( std::size_t( maxsteps + 1 ) );
    sol.x.reserve( std::size_t( maxsteps + 1 ) );
    sol.p.reserve( std::size_t( maxsteps + 1 ) );
    sol.t.push_back( t0 );
    sol.x.push_back( x0 );

    const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };
    T tc = t0;
    T xc = x0;

    for ( int s = 0; s < maxsteps; ++s )
    {
        if ( sign * ( tmax - tc ) <= T{} ) break;

        auto [p, h] = step< N >( f, xc, tc, abstol );
        if ( h <= T{} ) break;

        const T dt = sign * std::min( h, std::abs( tmax - tc ) );
        xc = p.eval( dt );
        sol.p.push_back( std::move( p ) );
        tc += dt;

        sol.t.push_back( tc );
        sol.x.push_back( xc );
    }

    return sol;
}

// =============================================================================
// Scalar ODE – output at specified times
// =============================================================================

/**
 * @brief Integrate scalar ODE at specific output times.
 * @param trange Monotonic sequence of output times (first element = t0).
 */
template < int N, typename F, typename T = double >
[[nodiscard]] TaylorSolution< N, T, T > integrate( F&& f, T x0, const std::vector< T >& trange,
                                                    T abstol, int maxsteps = 500 )
{
    const T t0 = trange.front();
    const T tmax = trange.back();
    const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };

    TaylorSolution< N, T, T > sol;
    sol.t = trange;
    sol.x.resize( trange.size() );
    sol.x[0] = x0;

    T tc = t0;
    T xc = x0;
    std::size_t nout = 1;
    int nsteps = 0;

    while ( nout < trange.size() && nsteps < maxsteps )
    {
        auto [p, h] = step< N >( f, xc, tc, abstol );
        if ( h <= T{} ) break;

        const T dt = sign * std::min( h, std::abs( tmax - tc ) );
        const T tc_new = tc + dt;

        while ( nout < trange.size() && sign * ( trange[nout] - tc_new ) <= T{} )
        {
            sol.x[nout] = p.eval( trange[nout] - tc );
            ++nout;
        }

        xc = p.eval( dt );
        tc = tc_new;
        ++nsteps;
    }

    return sol;
}

// =============================================================================
// Vector ODE – adaptive stepping
// =============================================================================

/**
 * @brief Integrate vector ODE f(dx, x, t) with adaptive step size.
 * @tparam N Taylor expansion order.
 * @param f Right-hand side: callable `f(dx, x, t)` writing derivatives into dx.
 * @param x0 Initial state vector.
 * @param t0 Initial time.
 * @param tmax Final time.
 * @param abstol Absolute tolerance for step-size control.
 * @param maxsteps Maximum number of integration steps.
 * @return TaylorSolution with dense-output polynomials.
 */
template < int N, typename F, typename T, int D >
[[nodiscard]] TaylorSolution< N, Eigen::Matrix< T, D, 1 >, T > integrate(
    F&& f, const Eigen::Matrix< T, D, 1 >& x0, T t0, T tmax, T abstol, int maxsteps = 500 )
{
    using Vec = Eigen::Matrix< T, D, 1 >;

    TaylorSolution< N, Vec, T > sol;
    sol.t.reserve( std::size_t( maxsteps + 1 ) );
    sol.x.reserve( std::size_t( maxsteps + 1 ) );
    sol.p.reserve( std::size_t( maxsteps + 1 ) );
    sol.t.push_back( t0 );
    sol.x.push_back( x0 );

    const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };
    T tc = t0;
    Vec xc = x0;

    for ( int s = 0; s < maxsteps; ++s )
    {
        if ( sign * ( tmax - tc ) <= T{} ) break;

        auto [p, h] = step< N >( f, xc, tc, abstol );
        if ( h <= T{} ) break;

        const T dt = sign * std::min( h, std::abs( tmax - tc ) );
        xc = eval( p, dt );
        sol.p.push_back( std::move( p ) );
        tc += dt;

        sol.t.push_back( tc );
        sol.x.push_back( xc );
    }

    return sol;
}

// =============================================================================
// Vector ODE – output at specified times
// =============================================================================

/**
 * @brief Integrate vector ODE at specific output times.
 * @param trange Monotonic sequence of output times (first element = t0).
 */
template < int N, typename F, typename T, int D >
[[nodiscard]] TaylorSolution< N, Eigen::Matrix< T, D, 1 >, T > integrate(
    F&& f, const Eigen::Matrix< T, D, 1 >& x0, const std::vector< T >& trange, T abstol,
    int maxsteps = 500 )
{
    using Vec = Eigen::Matrix< T, D, 1 >;

    const T t0 = trange.front();
    const T tmax = trange.back();
    const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };

    TaylorSolution< N, Vec, T > sol;
    sol.t = trange;
    sol.x.resize( trange.size() );
    sol.x[0] = x0;

    T tc = t0;
    Vec xc = x0;
    std::size_t nout = 1;
    int nsteps = 0;

    while ( nout < trange.size() && nsteps < maxsteps )
    {
        auto [p, h] = step< N >( f, xc, tc, abstol );
        if ( h <= T{} ) break;

        const T dt = sign * std::min( h, std::abs( tmax - tc ) );
        const T tc_new = tc + dt;

        while ( nout < trange.size() && sign * ( trange[nout] - tc_new ) <= T{} )
        {
            sol.x[nout] = eval( p, T( trange[nout] - tc ) );
            ++nout;
        }

        xc = eval( p, dt );
        tc = tc_new;
        ++nsteps;
    }

    return sol;
}

}  // namespace tax::ode
