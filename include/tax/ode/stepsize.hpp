#pragma once

#include <cmath>
#include <limits>

#include <tax/tte.hpp>
#include <tax/eigen/num_traits.hpp>

namespace tax::ode
{

/**
 * @brief Adaptive step size from the last two Taylor coefficients.
 * @details Implements the criterion from Jorba & Zou (2005), Eq. 3-3:
 *   h = min( (ε/|x[N-1]|)^{1/(N-1)}, (ε/|x[N]|)^{1/N} ).
 * @param x Taylor polynomial of the solution.
 * @param abstol Absolute tolerance.
 * @return Recommended step-size magnitude (always positive).
 */
template < typename T, int N >
[[nodiscard]] T stepsize( const TruncatedExpansionT< T, N, 1 >& x, T abstol ) noexcept
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

/**
 * @brief Adaptive step size for a vector ODE (minimum across components).
 */
template < typename T, int N, int D >
[[nodiscard]] T stepsize(
    const Eigen::Matrix< TruncatedExpansionT< T, N, 1 >, D, 1 >& x, T abstol ) noexcept
{
    T h = std::numeric_limits< T >::infinity();
    for ( Eigen::Index i = 0; i < x.size(); ++i ) h = std::min( h, stepsize( x( i ), abstol ) );
    return h;
}

}  // namespace tax::ode
