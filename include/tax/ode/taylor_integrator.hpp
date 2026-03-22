#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include <tax/tte.hpp>
#include <tax/eigen/eval.hpp>

namespace tax::ode
{

// =============================================================================
// Solution container
// =============================================================================

/// @brief Solution returned by taylorinteg.
/// @tparam State Scalar `T` for scalar ODEs, `Eigen::Matrix<T,D,1>` for vector ODEs.
/// @tparam T Scalar coefficient type.
template < typename State, typename T = double >
struct TaylorSolution
{
    std::vector< T > t;
    std::vector< State > x;
};

// =============================================================================
// Step-size control
// =============================================================================

/// @brief Adaptive step size from the last two Taylor coefficients.
/// @details Implements the criterion from Jorba & Zou (2005), Eq. 3-3:
///   h = min( (ε/|x[N-1]|)^{1/(N-1)}, (ε/|x[N]|)^{1/N} ).
template < typename T, int N >
[[nodiscard]] T stepsize( const TruncatedTaylorExpansionT< T, N, 1 >& x, T abstol )
{
    using std::abs;
    using std::min;
    using std::pow;

    T h = std::numeric_limits< T >::infinity();

    if constexpr ( N >= 2 )
    {
        const T c = abs( x[N - 1] );
        if ( c > T{} ) h = min( h, pow( abstol / c, T{ 1 } / T( N - 1 ) ) );
    }
    {
        const T c = abs( x[N] );
        if ( c > T{} ) h = min( h, pow( abstol / c, T{ 1 } / T( N ) ) );
    }
    return h;
}

/// @brief Adaptive step size for a vector ODE (minimum across components).
template < typename T, int N, int D >
[[nodiscard]] T stepsize( const Eigen::Matrix< TruncatedTaylorExpansionT< T, N, 1 >, D, 1 >& x,
                          T abstol )
{
    T h = std::numeric_limits< T >::infinity();
    for ( Eigen::Index i = 0; i < x.size(); ++i ) h = std::min( h, stepsize( x( i ), abstol ) );
    return h;
}

// =============================================================================
// Jet-coefficient computation
// =============================================================================

/// @brief Compute Taylor coefficients for scalar ODE dx/dt = f(x, t).
/// @details Iteratively builds up the Taylor expansion of x(t) by
///   x[k+1] = f(x,t)[k] / (k+1)  for k = 0, ..., N-1.
template < typename T, int N, typename F >
void jetcoeffs( TruncatedTaylorExpansionT< T, N, 1 >& x,
                const TruncatedTaylorExpansionT< T, N, 1 >& t, F&& f )
{
    using TTE = TruncatedTaylorExpansionT< T, N, 1 >;
    for ( int k = 0; k < N; ++k )
    {
        TTE dx = f( x, t );
        x[k + 1] = dx[k] / T( k + 1 );
    }
}

/// @brief Compute Taylor coefficients for vector ODE f(dx, x, t) (in-place).
template < typename T, int N, int D, typename F >
void jetcoeffs( Eigen::Matrix< TruncatedTaylorExpansionT< T, N, 1 >, D, 1 >& x,
                const TruncatedTaylorExpansionT< T, N, 1 >& t, F&& f )
{
    using TTE = TruncatedTaylorExpansionT< T, N, 1 >;
    const Eigen::Index dim = x.size();
    Eigen::Matrix< TTE, D, 1 > dx( dim );

    for ( int k = 0; k < N; ++k )
    {
        f( dx, x, t );
        for ( Eigen::Index i = 0; i < dim; ++i ) x( i )[k + 1] = dx( i )[k] / T( k + 1 );
    }
}

// =============================================================================
// Single Taylor step
// =============================================================================

/// @brief One Taylor step for a scalar ODE. Returns the step-size magnitude.
template < typename T, int N, typename F >
[[nodiscard]] T taylorstep( TruncatedTaylorExpansionT< T, N, 1 >& x,
                            const TruncatedTaylorExpansionT< T, N, 1 >& t, F&& f, T abstol )
{
    jetcoeffs( x, t, std::forward< F >( f ) );
    return stepsize( x, abstol );
}

/// @brief One Taylor step for a vector ODE. Returns the step-size magnitude.
template < typename T, int N, int D, typename F >
[[nodiscard]] T taylorstep( Eigen::Matrix< TruncatedTaylorExpansionT< T, N, 1 >, D, 1 >& x,
                            const TruncatedTaylorExpansionT< T, N, 1 >& t, F&& f, T abstol )
{
    jetcoeffs( x, t, std::forward< F >( f ) );
    return stepsize( x, abstol );
}

// =============================================================================
// Full integration – adaptive stepping
// =============================================================================

/// @brief Integrate scalar ODE dx/dt = f(x, t) with adaptive step size.
/// @tparam N Taylor expansion order.
/// @param f  Right-hand side: callable `f(x, t)` returning the derivative.
/// @param x0 Initial state.
/// @param t0 Initial time.
/// @param tmax Final time.
/// @param abstol Absolute tolerance for step-size control.
/// @param maxsteps Maximum number of integration steps.
/// @return TaylorSolution with time and state at each adaptive step.
template < int N, typename F, typename T = double >
TaylorSolution< T, T > taylorinteg( F&& f, T x0, T t0, T tmax, T abstol, int maxsteps = 500 )
{
    using TTE = TruncatedTaylorExpansionT< T, N, 1 >;

    TaylorSolution< T, T > sol;
    sol.t.reserve( std::size_t( maxsteps + 1 ) );
    sol.x.reserve( std::size_t( maxsteps + 1 ) );
    sol.t.push_back( t0 );
    sol.x.push_back( x0 );

    const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };
    T tc = t0;
    T xc = x0;

    for ( int step = 0; step < maxsteps; ++step )
    {
        if ( sign * ( tmax - tc ) <= T{} ) break;

        TTE t_da{};
        t_da[0] = tc;
        if constexpr ( N >= 1 ) t_da[1] = T{ 1 };

        TTE x_da{};
        x_da[0] = xc;

        T h = taylorstep( x_da, t_da, f, abstol );
        if ( h <= T{} ) break;

        const T dt = sign * std::min( h, std::abs( tmax - tc ) );

        xc = x_da.eval( dt );
        tc += dt;

        sol.t.push_back( tc );
        sol.x.push_back( xc );
    }

    return sol;
}

/// @brief Integrate scalar ODE at specific output times.
/// @param trange Monotonic sequence of output times (first element = t0).
template < int N, typename F, typename T = double >
TaylorSolution< T, T > taylorinteg( F&& f, T x0, const std::vector< T >& trange, T abstol,
                                     int maxsteps = 500 )
{
    using TTE = TruncatedTaylorExpansionT< T, N, 1 >;

    const T t0 = trange.front();
    const T tmax = trange.back();
    const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };

    TaylorSolution< T, T > sol;
    sol.t = trange;
    sol.x.resize( trange.size() );
    sol.x[0] = x0;

    T tc = t0;
    T xc = x0;
    std::size_t nout = 1;
    int nsteps = 0;

    while ( nout < trange.size() && nsteps < maxsteps )
    {
        TTE t_da{};
        t_da[0] = tc;
        if constexpr ( N >= 1 ) t_da[1] = T{ 1 };

        TTE x_da{};
        x_da[0] = xc;

        T h = taylorstep( x_da, t_da, f, abstol );
        if ( h <= T{} ) break;

        const T dt = sign * std::min( h, std::abs( tmax - tc ) );
        const T tc_new = tc + dt;

        // Emit at all output times within this step
        while ( nout < trange.size() && sign * ( trange[nout] - tc_new ) <= T{} )
        {
            sol.x[nout] = x_da.eval( trange[nout] - tc );
            ++nout;
        }

        xc = x_da.eval( dt );
        tc = tc_new;
        ++nsteps;
    }

    return sol;
}

// =============================================================================
// Full integration – vector ODE, adaptive stepping
// =============================================================================

/// @brief Integrate vector ODE f(dx, x, t) with adaptive step size.
/// @tparam N Taylor expansion order.
/// @param f  Right-hand side: callable `f(dx, x, t)` writing derivatives into dx.
/// @param x0 Initial state vector.
/// @param t0 Initial time.
/// @param tmax Final time.
/// @param abstol Absolute tolerance for step-size control.
/// @param maxsteps Maximum number of integration steps.
template < int N, typename F, typename T, int D >
TaylorSolution< Eigen::Matrix< T, D, 1 >, T > taylorinteg(
    F&& f, const Eigen::Matrix< T, D, 1 >& x0, T t0, T tmax, T abstol, int maxsteps = 500 )
{
    using TTE = TruncatedTaylorExpansionT< T, N, 1 >;
    using Vec = Eigen::Matrix< T, D, 1 >;
    using VecDA = Eigen::Matrix< TTE, D, 1 >;

    TaylorSolution< Vec, T > sol;
    sol.t.reserve( std::size_t( maxsteps + 1 ) );
    sol.x.reserve( std::size_t( maxsteps + 1 ) );
    sol.t.push_back( t0 );
    sol.x.push_back( x0 );

    const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };
    const Eigen::Index dim = x0.size();
    T tc = t0;
    Vec xc = x0;

    for ( int step = 0; step < maxsteps; ++step )
    {
        if ( sign * ( tmax - tc ) <= T{} ) break;

        TTE t_da{};
        t_da[0] = tc;
        if constexpr ( N >= 1 ) t_da[1] = T{ 1 };

        VecDA x_da( dim );
        for ( Eigen::Index i = 0; i < dim; ++i )
        {
            x_da( i ) = TTE{};
            x_da( i )[0] = xc( i );
        }

        T h = taylorstep( x_da, t_da, f, abstol );
        if ( h <= T{} ) break;

        const T dt = sign * std::min( h, std::abs( tmax - tc ) );

        xc = eval( x_da, dt );
        tc += dt;

        sol.t.push_back( tc );
        sol.x.push_back( xc );
    }

    return sol;
}

/// @brief Integrate vector ODE at specific output times.
/// @param trange Monotonic sequence of output times (first element = t0).
template < int N, typename F, typename T, int D >
TaylorSolution< Eigen::Matrix< T, D, 1 >, T > taylorinteg(
    F&& f, const Eigen::Matrix< T, D, 1 >& x0, const std::vector< T >& trange, T abstol,
    int maxsteps = 500 )
{
    using TTE = TruncatedTaylorExpansionT< T, N, 1 >;
    using Vec = Eigen::Matrix< T, D, 1 >;
    using VecDA = Eigen::Matrix< TTE, D, 1 >;

    const T t0 = trange.front();
    const T tmax = trange.back();
    const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };
    const Eigen::Index dim = x0.size();

    TaylorSolution< Vec, T > sol;
    sol.t = trange;
    sol.x.resize( trange.size() );
    sol.x[0] = x0;

    T tc = t0;
    Vec xc = x0;
    std::size_t nout = 1;
    int nsteps = 0;

    while ( nout < trange.size() && nsteps < maxsteps )
    {
        TTE t_da{};
        t_da[0] = tc;
        if constexpr ( N >= 1 ) t_da[1] = T{ 1 };

        VecDA x_da( dim );
        for ( Eigen::Index i = 0; i < dim; ++i )
        {
            x_da( i ) = TTE{};
            x_da( i )[0] = xc( i );
        }

        T h = taylorstep( x_da, t_da, f, abstol );
        if ( h <= T{} ) break;

        const T dt = sign * std::min( h, std::abs( tmax - tc ) );
        const T tc_new = tc + dt;

        while ( nout < trange.size() && sign * ( trange[nout] - tc_new ) <= T{} )
        {
            sol.x[nout] = eval( x_da, T( trange[nout] - tc ) );
            ++nout;
        }

        xc = eval( x_da, dt );
        tc = tc_new;
        ++nsteps;
    }

    return sol;
}

}  // namespace tax::ode
