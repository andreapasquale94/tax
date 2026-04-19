#pragma once

#include <cam/linalg.hpp>
#include <cmath>

namespace cam
{

namespace detail
{
constexpr int factorial( int n ) noexcept { return ( n <= 1 ) ? 1 : n * factorial( n - 1 ); }
}  // namespace detail

/// @brief Chan's collision-probability formulation truncated at `nOrder = 2`.
/// @param Cb 2x2 covariance in the B-plane (xi, zeta).
/// @param r12 Combined hard-body radius.
/// @param posRel Relative position in the B-plane (xi, zeta).
template < typename T >
[[nodiscard]] T collProbChan( const Eigen::Matrix< T, 2, 2 >& Cb, double r12,
                              const Eigen::Matrix< T, 2, 1 >& posRel )
{
    using std::exp;
    using std::pow;
    using std::sqrt;

    const T xi_exp = posRel[0];
    const T zeta_exp = posRel[1];

    const T sigma_xi = sqrt( Cb( 0, 0 ) );
    const T sigma_zeta = sqrt( Cb( 1, 1 ) );
    const T rho = Cb( 0, 1 ) / ( sigma_xi * sigma_zeta );

    const T u = ( r12 * r12 ) / ( sigma_xi * sigma_zeta * sqrt( 1.0 - rho * rho ) );
    const T v = ( ( xi_exp / sigma_xi ) * ( xi_exp / sigma_xi ) +
                  ( zeta_exp / sigma_zeta ) * ( zeta_exp / sigma_zeta ) -
                  2.0 * rho * ( xi_exp * zeta_exp / ( sigma_xi * sigma_zeta ) ) ) /
                ( 1.0 - rho * rho );

    constexpr int nOrder = 2;
    T secondLoop{ 0.0 };
    for ( int m = 0; m <= nOrder; ++m )
    {
        T firstLoop{ 0.0 };
        for ( int k = 0; k <= m; ++k )
            firstLoop = firstLoop + pow( u, k ) / ( pow( 2.0, k ) * detail::factorial( k ) );
        secondLoop = secondLoop + pow( v, m ) / ( pow( 2.0, m ) * detail::factorial( m ) ) *
                                      ( 1.0 - exp( -u / 2.0 ) * firstLoop );
    }
    return exp( -v / 2.0 ) * secondLoop;
}

/// @brief Alfano's collision-probability formulation.
/// @param Cb 2x2 covariance in the B-plane.
/// @param r12 Combined hard-body radius.
/// @param posRel Relative position in the B-plane.
/// @param n Number of subdivisions (default 30).
template < typename T >
[[nodiscard]] T collProbAlfano( const Eigen::Matrix< T, 2, 2 >& Cb, double r12,
                                const Eigen::Matrix< T, 2, 1 >& posRel, int n = 30 )
{
    using std::atan;
    using std::cos;
    using std::erf;
    using std::exp;
    using std::sin;
    using std::sqrt;

    const T x = posRel[0];
    const T z = posRel[1];

    const T sigma_x = sqrt( Cb( 0, 0 ) );
    const T sigma_z = sqrt( Cb( 1, 1 ) );
    const T rho = Cb( 0, 1 ) / ( sigma_x * sigma_z );
    T theta = 0.5 * atan( 2.0 * rho * sigma_x * sigma_z /
                          ( sigma_x * sigma_x - sigma_z * sigma_z ) );
    if ( cons( sigma_z ) > cons( sigma_x ) ) theta = theta + atan( T{ 1.0 } ) * 2.0;

    Eigen::Matrix< T, 2, 2 > R;
    R( 0, 0 ) = cos( theta );
    R( 0, 1 ) = sin( theta );
    R( 1, 0 ) = -sin( theta );
    R( 1, 1 ) = cos( theta );

    Eigen::Matrix< T, 2, 2 > C = R * Cb * R.transpose();
    Eigen::Matrix< T, 2, 1 > xx = R * posRel;

    const T xm = xx[0];
    const T zm = xx[1];
    const T sigmax = sqrt( C( 0, 0 ) );
    const T sigmaz = sqrt( C( 1, 1 ) );

    T accum{ 0.0 };
    for ( int i = 1; i <= n; ++i )
    {
        const T aux1 = ( zm + 2.0 * r12 / n * sqrt( T( double( ( n - i ) * i ) ) ) ) /
                       ( sigmaz * sqrt( T{ 2.0 } ) );
        const T aux2 = ( -zm + 2.0 * r12 / n * sqrt( T( double( ( n - i ) * i ) ) ) ) /
                       ( sigmaz * sqrt( T{ 2.0 } ) );
        const T aux3 = -( ( r12 * ( 2.0 * i - n ) / n + xm ) *
                          ( r12 * ( 2.0 * i - n ) / n + xm ) ) /
                       ( 2.0 * sigmax * sigmax );
        accum = accum + ( erf( aux1 ) + erf( aux2 ) ) * exp( aux3 );
    }
    return r12 * 2.0 / ( sqrt( 8.0 * atan( T{ 1.0 } ) * 4.0 ) * sigmax * n ) * accum;
}

}  // namespace cam
