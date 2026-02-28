#pragma once

#include <tax/da.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tax
{

/**
 * @brief Extract the scalar value (constant term) from each DA element.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int M, int Rank >
[[nodiscard]] auto value( const Eigen::Tensor< TDA< T, N, M >, Rank >& t )
{
    Eigen::Tensor< T, Rank > out( t.dimensions() );
    for ( Eigen::Index i = 0; i < t.size(); ++i ) out.data()[i] = t.data()[i].value();
    return out;
}

/**
 * @brief Extract a partial derivative from each DA element (runtime multi-index).
 * @param alpha Multi-index specifying the derivative order per variable.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int M, int Rank >
[[nodiscard]] auto derivative( const Eigen::Tensor< TDA< T, N, M >, Rank >& t,
                               const std::array< int, std::size_t( M ) >& alpha )
{
    Eigen::Tensor< T, Rank > out( t.dimensions() );
    for ( Eigen::Index i = 0; i < t.size(); ++i ) out.data()[i] = t.data()[i].derivative( alpha );
    return out;
}

/**
 * @brief Extract the k-th time derivative from each univariate DA element.
 * @param k Derivative order (0 = value, 1 = first derivative, ...).
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int Rank >
[[nodiscard]] auto derivative( const Eigen::Tensor< TDA< T, N, 1 >, Rank >& t, int k )
{
    return derivative( t, MultiIndex< 1 >{ k } );
}

/**
 * @brief Extract a partial derivative from each DA element (compile-time multi-index).
 * @tparam Alpha Derivative orders for each variable.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < int... Alpha, typename T, int N, int M, int Rank >
[[nodiscard]] auto derivative( const Eigen::Tensor< TDA< T, N, M >, Rank >& t )
{
    static_assert( sizeof...( Alpha ) == M,
                   "Derivative multi-index arity must match number of variables" );
    static_assert( ( ( Alpha >= 0 ) && ... ), "Derivative orders must be non-negative" );
    constexpr int total_order = ( Alpha + ... + 0 );
    static_assert( total_order <= N, "Derivative total order exceeds DA truncation order" );

    Eigen::Tensor< T, Rank > out( t.dimensions() );
    for ( Eigen::Index i = 0; i < t.size(); ++i )
        out.data()[i] = t.data()[i].template derivative< Alpha... >();
    return out;
}

}  // namespace tax
