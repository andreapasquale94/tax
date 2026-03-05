#pragma once

#include <cassert>
#include <concepts>
#include <tax/eigen/num_traits.hpp>
#include <tuple>
#include <utility>

namespace tax
{

namespace detail
{

/// @brief Traits for extracting scalar type, order, and number of variables from a
/// TruncatedTaylorExpansionT type.
template < typename >
struct expansion_traits;

template < typename T, int N, int M >
struct expansion_traits< TruncatedTaylorExpansionT< T, N, M > >
{
    using scalar_type = T;
    static constexpr int order = N;
    static constexpr int vars = M;
};

/// @brief True if `TTE` is a `TruncatedTaylorExpansionT<T, N, M>` specialization.
template < typename >
inline constexpr bool is_tte_v = false;

template < typename T, int N, int M >
inline constexpr bool is_tte_v< TruncatedTaylorExpansionT< T, N, M > > = true;

/// @brief Rebind an Eigen matrix type to use a different scalar.
template < typename Derived, typename Scalar >
using rebind_matrix_t =
    Eigen::Matrix< Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime, Derived::Options,
                   Derived::MaxRowsAtCompileTime, Derived::MaxColsAtCompileTime >;

/// @brief Concept for Eigen dense expression types.
template < typename T >
concept EigenDenseExpr = requires( const T& t ) {
    typename T::Scalar;
    t.derived();
};

}  // namespace detail

/**
 * @brief Create a TTE Eigen container from a compile-time-sized expansion point.
 * @tparam TTE The TruncatedTaylorExpansionT type (e.g., `TEn<2, 4>`). `M` must equal the input size.
 * @param x0 Eigen matrix/vector with `M` entries.
 * @return Eigen matrix of same shape with TTE variable entries.
 */
template < typename TTE, typename Derived >
[[nodiscard]] auto vector( const Eigen::DenseBase< Derived >& x0 ) noexcept
    requires( detail::is_tte_v< TTE > &&
              std::convertible_to< typename Derived::Scalar,
                                   typename detail::expansion_traits< TTE >::scalar_type > )
{
    using T = typename detail::expansion_traits< TTE >::scalar_type;
    constexpr int M = detail::expansion_traits< TTE >::vars;
    constexpr int Rows = Derived::RowsAtCompileTime;
    constexpr int Cols = Derived::ColsAtCompileTime;
    static_assert( Rows != Eigen::Dynamic && Cols != Eigen::Dynamic,
                   "vector(x0) requires compile-time-sized Eigen inputs" );
    static_assert( Rows >= 1 && Cols >= 1, "vector(x0) requires non-empty Eigen inputs" );
    static_assert( Rows * Cols == M, "vector(x0) size must match number of variables M" );
    using Out = detail::rebind_matrix_t< Derived, TTE >;
    typename TTE::Input p{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        (
            [&] {
                if constexpr ( Rows == 1 )
                    p[I] = static_cast< T >( x0( Eigen::Index( 0 ), Eigen::Index( I ) ) );
                else if constexpr ( Cols == 1 )
                    p[I] = static_cast< T >( x0( Eigen::Index( I ), Eigen::Index( 0 ) ) );
                else
                    p[I] = static_cast< T >(
                        x0( Eigen::Index( int( I ) / Cols ), Eigen::Index( int( I ) % Cols ) ) );
            }(),
            ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );

    Out out{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out( Eigen::Index( int( I ) / Cols ), Eigen::Index( int( I ) % Cols ) ) =
                TTE::template variable< int( I ) >( p ) ),
          ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );
    return out;
}

/**
 * @brief Create all coordinate variables from an Eigen vector expansion point.
 * @tparam TTE The TruncatedTaylorExpansionT type (e.g., `TEn<2, 3>`). `M` must match the vector
 * size.
 * @param x0 Eigen vector with `M` entries.
 * @return Tuple `(x_0, ..., x_{M-1})` of TTE variables.
 */
template < typename TTE, typename Derived >
[[nodiscard]] auto variables( const Eigen::DenseBase< Derived >& x0 ) noexcept
    requires( detail::is_tte_v< TTE > &&
              std::convertible_to< typename Derived::Scalar,
                                   typename detail::expansion_traits< TTE >::scalar_type > )
{
    using T = typename detail::expansion_traits< TTE >::scalar_type;
    constexpr int M = detail::expansion_traits< TTE >::vars;

    static_assert( Derived::RowsAtCompileTime == 1 || Derived::ColsAtCompileTime == 1 ||
                       Derived::RowsAtCompileTime == Eigen::Dynamic ||
                       Derived::ColsAtCompileTime == Eigen::Dynamic,
                   "Eigen input must be a vector expression" );
    static_assert( Derived::SizeAtCompileTime == Eigen::Dynamic || Derived::SizeAtCompileTime == M,
                   "Eigen vector size must match number of variables" );
    assert( x0.rows() == 1 || x0.cols() == 1 );
    assert( x0.size() == Eigen::Index( M ) );

    typename TTE::Input p{};
    if ( x0.cols() == 1 )
    {
        for ( int i = 0; i < M; ++i )
            p[std::size_t( i )] = static_cast< T >( x0( Eigen::Index( i ), Eigen::Index( 0 ) ) );
    } else
    {
        for ( int i = 0; i < M; ++i )
            p[std::size_t( i )] = static_cast< T >( x0( Eigen::Index( 0 ), Eigen::Index( i ) ) );
    }
    return TTE::variables( p );
}

}  // namespace tax
