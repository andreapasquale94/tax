#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <tax/tte.hpp>
#include <tax/eigen/eval.hpp>

namespace tax::ode
{

namespace detail
{

/// @brief Map State → Polynomial type used for dense output.
template < int N, typename State, typename T >
struct poly_type
{
    using type = TruncatedExpansionT< T, N, 1 >;
};

template < int N, typename T, int D >
struct poly_type< N, Eigen::Matrix< T, D, 1 >, T >
{
    using type = Eigen::Matrix< TruncatedExpansionT< T, N, 1 >, D, 1 >;
};

template < int N, typename State, typename T >
using poly_t = typename poly_type< N, State, T >::type;

/// @brief Evaluate a polynomial at displacement τ (scalar specialisation).
template < typename T, int N >
[[nodiscard]] T eval_poly( const TruncatedExpansionT< T, N, 1 >& p, T tau ) noexcept
{
    return p.eval( tau );
}

/// @brief Evaluate a polynomial vector at displacement τ (vector specialisation).
template < typename T, int N, int D >
[[nodiscard]] Eigen::Matrix< T, D, 1 > eval_poly(
    const Eigen::Matrix< TruncatedExpansionT< T, N, 1 >, D, 1 >& p, T tau ) noexcept
{
    return eval( p, tau );
}

}  // namespace detail

/**
 * @brief Solution of a Taylor-integrated ODE with optional dense output.
 *
 * @tparam N  Taylor expansion order used during integration.
 * @tparam State  Scalar `T` for scalar ODEs, `Eigen::Matrix<T,D,1>` for vector ODEs.
 * @tparam T  Scalar coefficient type (default `double`).
 *
 * @details Stores the step times, state values, and (optionally) the Taylor
 * polynomials at each step.  When polynomials are present, `operator()` provides
 * dense output: evaluating the solution at any time within the integration range
 * by selecting the appropriate step polynomial and calling `TTE::eval`.
 */
template < int N, typename State, typename T = double >
struct TaylorSolution
{
    using Poly = detail::poly_t< N, State, T >;

    std::vector< T > t;      ///< Step times (monotonic).
    std::vector< State > x;  ///< State at each step time.
    std::vector< Poly > p;   ///< Taylor polynomials centred at each step time.

    /**
     * @brief Dense-output evaluation at arbitrary time.
     * @param time  Query time (must lie within [t.front(), t.back()]).
     * @return  Interpolated state.
     * @throws std::out_of_range if polynomials are empty.
     */
    [[nodiscard]] State operator()( T time ) const
    {
        if ( p.empty() )
            throw std::out_of_range(
                "TaylorSolution::operator(): no polynomials stored for dense output" );

        const T sign = t.back() >= t.front() ? T{ 1 } : T{ -1 };

        // Binary search for the interval containing `time`.
        // p[i] is centred at t[i] and valid over [t[i], t[i+1]).
        std::size_t idx;
        if ( sign > T{} )
        {
            auto it = std::upper_bound( t.begin(), t.end(), time );
            idx = it == t.begin() ? 0 : std::size_t( std::distance( t.begin(), it ) ) - 1;
        }
        else
        {
            auto it = std::lower_bound( t.begin(), t.end(), time, std::greater< T >{} );
            idx = it == t.begin() ? 0 : std::size_t( std::distance( t.begin(), it ) ) - 1;
        }
        if ( idx >= p.size() ) idx = p.size() - 1;

        return detail::eval_poly( p[idx], time - t[idx] );
    }
};

}  // namespace tax::ode
