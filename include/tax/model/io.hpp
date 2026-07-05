#pragma once

// Human-readable output and vector conveniences for Taylor models.
//
//   std::cout << tm;                 // "P(dx) + I  on  [a,b]"
//   auto v  = tax::model::value(state);     // constant parts
//   auto b  = tax::model::bound(state);     // per-component enclosures
//   auto J  = tax::model::jacobian(state);  // state-transition matrix

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <tax/io/series.hpp>
#include <tax/model/taylor_model.hpp>

namespace tax::model
{

/// Stream a Taylor model as `polynomial + remainder  on  domain-box`.
template < std::floating_point T, int N, int M >
std::ostream& operator<<( std::ostream& os, const TaylorModel< T, N, M >& tm )
{
    os << tm.polynomial() << "  +  " << tm.remainder() << "  on  [";
    for ( int i = 0; i < M; ++i )
    {
        if ( i ) os << " x ";
        os << tm.domain()[std::size_t( i )];
    }
    return os << "]";
}

/// Human-readable string form of a Taylor model.
template < std::floating_point T, int N, int M >
[[nodiscard]] std::string to_string( const TaylorModel< T, N, M >& tm )
{
    std::ostringstream os;
    os << tm;
    return os.str();
}

// ---------------------------------------------------------------------------
// Vector conveniences over a state of D Taylor models
// ---------------------------------------------------------------------------

/// Constant parts of each component (the state value at the expansion point).
template < std::floating_point T, int N, int M, std::size_t D >
[[nodiscard]] std::array< T, D > value( const std::array< TaylorModel< T, N, M >, D >& state )
{
    std::array< T, D > v{};
    for ( std::size_t i = 0; i < D; ++i ) v[i] = state[i].value();
    return v;
}

/// Rigorous enclosure of each component over the domain.
template < std::floating_point T, int N, int M, std::size_t D >
[[nodiscard]] std::array< Interval< T >, D > bound(
    const std::array< TaylorModel< T, N, M >, D >& state, Bounder which = Bounder::Quadratic )
{
    std::array< Interval< T >, D > b{};
    for ( std::size_t i = 0; i < D; ++i ) b[i] = state[i].bound( which );
    return b;
}

/// State-transition matrix: J(i, j) = d(state_i) / d(x_j) at the expansion
/// point, read from the polynomial parts. For a flow map this is the
/// sensitivity of the propagated state to the initial conditions.
template < std::floating_point T, int N, int M, std::size_t D >
[[nodiscard]] Eigen::Matrix< T, int( D ), M > jacobian(
    const std::array< TaylorModel< T, N, M >, D >& state )
{
    Eigen::Matrix< T, int( D ), M > J;
    for ( std::size_t i = 0; i < D; ++i )
    {
        MultiIndex< M > alpha{};
        for ( int j = 0; j < M; ++j )
        {
            alpha[std::size_t( j )] = 1;
            J( int( i ), j ) = state[i].polynomial().derivative( alpha );
            alpha[std::size_t( j )] = 0;
        }
    }
    return J;
}

}  // namespace tax::model
