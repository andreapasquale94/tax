#pragma once

#include <cmath>

namespace cam
{

/// @brief True anomaly -> eccentric anomaly.
template < typename T >
[[nodiscard]] T true2eccAnomaly( const T& theta, const T& e )
{
    using std::atan2;
    using std::cos;
    using std::sin;
    using std::sqrt;
    return 2.0 * atan2( sqrt( 1.0 - e ) * sin( theta / 2.0 ),
                        sqrt( 1.0 + e ) * cos( theta / 2.0 ) );
}

/// @brief Eccentric anomaly -> true anomaly.
template < typename T >
[[nodiscard]] T ecc2trueAnomaly( const T& E, const T& e )
{
    using std::atan2;
    using std::cos;
    using std::sin;
    using std::sqrt;
    return 2.0 *
           atan2( sqrt( 1.0 + e ) * sin( E / 2.0 ), sqrt( 1.0 - e ) * cos( E / 2.0 ) );
}

/// @brief Mean anomaly -> eccentric anomaly via fixed-point iteration on Kepler's eqn.
template < typename T >
[[nodiscard]] T mean2eccAnomaly( const T& M, const T& e )
{
    using std::sin;
    T E = M;
    for ( int i = 0; i < 20; ++i ) E = M + e * sin( E );
    return E;
}

/// @brief Mean anomaly -> true anomaly.
template < typename T >
[[nodiscard]] T mean2trueAnomaly( const T& M, const T& e )
{
    return ecc2trueAnomaly( mean2eccAnomaly( M, e ), e );
}

/// @brief True anomaly -> mean anomaly.
template < typename T >
[[nodiscard]] T true2meanAnomaly( const T& theta, const T& e )
{
    using std::sin;
    T E = true2eccAnomaly( theta, e );
    return E - e * sin( E );
}

}  // namespace cam
