#pragma once

#include <cam/linalg.hpp>
#include <tax/eigen/eval.hpp>
#include <tax/eigen/invert_map.hpp>
#include <tax/eigen/variables.hpp>
#include <tax/tte.hpp>

namespace cam
{

/// @brief Find time of closest approach as a polynomial in the remaining variables.
/// @details Uses polynomial map inversion: given the relative state `xx` expressed as an
///          `M`-variable polynomial whose last variable is the deviation in time-to-TCA,
///          the routine inverts the map `[dot(r,v), x_1..x_{M-1}]` and returns the last
///          component of the evaluated inverse at `(-dot(r,v)_0, x_1..x_{M-1})`.
/// @tparam N Truncation order.
/// @tparam M Number of variables (must be >= 2; last variable is the time variable).
/// @param xx Relative state (r, v) as an Eigen vector of TTE components.
/// @returns TTE polynomial giving TCA correction as a function of the state perturbations.
template < int N, int M >
[[nodiscard]] tax::TEn< N, M > findTCA(
    const Eigen::Matrix< tax::TEn< N, M >, 6, 1 >& xx ) noexcept
{
    static_assert( M >= 2, "findTCA requires at least 2 variables (state + time)" );

    using DA = tax::TEn< N, M >;

    Vec3< DA > rr, vv;
    rr << xx[0], xx[1], xx[2];
    vv << xx[3], xx[4], xx[5];

    DA rvdot = rr.dot( vv );
    const double rvdot0 = rvdot.value();

    // Build map following the DACE convention:
    //   map[0]     = g(x) - g(0)            (depends on all M variables)
    //   map[1..M-1] = identity on vars 0..M-2 (state perturbations)
    // After inversion, component M-1 gives the time variable as a polynomial in
    // (g-g0, dx_0, ..., dx_{M-2}). Evaluating at g-g0 = -g0 enforces g = 0.
    typename DA::Input x0{};
    Eigen::Matrix< DA, M, 1 > map;
    map( 0 ) = rvdot - rvdot0;
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( map( Eigen::Index( I + 1 ) ) = DA::template variable< int( I ) >( x0 ) ), ... );
    }( std::make_index_sequence< std::size_t( M - 1 ) >{} );

    Eigen::Matrix< DA, M, 1 > mapInv = tax::invert( map );

    Eigen::Matrix< DA, M, 1 > dx;
    dx( 0 ) = DA{ -rvdot0 };
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( dx( Eigen::Index( I + 1 ) ) = DA::template variable< int( I ) >( x0 ) ), ... );
    }( std::make_index_sequence< std::size_t( M - 1 ) >{} );

    return tax::detail::composeOne< DA >( mapInv( M - 1 ), dx );
}

}  // namespace cam
