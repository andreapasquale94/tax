#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <tax/eigen/num_traits.hpp>
#include <tax/eigen/value.hpp>
#include <tax/expr/base.hpp>
#include <tax/tte.hpp>

namespace cam
{

template < typename T >
using Vec3 = Eigen::Matrix< T, 3, 1 >;

template < typename T >
using Vec6 = Eigen::Matrix< T, 6, 1 >;

template < typename T >
using Vec7 = Eigen::Matrix< T, 7, 1 >;

template < typename T >
using Mat3 = Eigen::Matrix< T, 3, 3 >;

/// @brief Euclidean norm of a 3-vector whose scalar supports `+`, `*`, and `sqrt`.
template < typename T >
[[nodiscard]] T vnorm( const Vec3< T >& v ) noexcept
{
    using std::sqrt;
    return sqrt( v[0] * v[0] + v[1] * v[1] + v[2] * v[2] );
}

/// @brief Constant term for a plain scalar is the scalar itself.
template < typename T >
[[nodiscard]] constexpr T cons( const T& x ) noexcept
    requires( std::is_arithmetic_v< T > )
{
    return x;
}

/// @brief Constant term of any tax expression (TTE or expression template).
template < typename Derived, typename T, int N, int M >
[[nodiscard]] constexpr T cons( const tax::Expr< Derived, T, N, M >& e ) noexcept
{
    return tax::TruncatedExpansionT< T, N, M >( e ).value();
}

}  // namespace cam
