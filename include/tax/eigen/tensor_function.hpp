#pragma once

#include <tax/eigen/adapters.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tax
{

// =============================================================================
// DenseBase (Matrix / Vector) overloads
// =============================================================================

/**
 * @brief Extract the scalar value (constant term) from each DA matrix/vector element.
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < typename Derived >
[[nodiscard]] auto value( const Eigen::DenseBase< Derived >& t )
    requires( detail::eigen::is_da_v< typename Derived::Scalar > )
{
    using DA = typename Derived::Scalar;
    using T = typename detail::eigen::da_traits< DA >::scalar_type;
    using Out = detail::eigen::rebind_matrix_t< Derived, T >;
    Out out( t.rows(), t.cols() );
    for ( Eigen::Index i = 0; i < t.size(); ++i ) out.coeffRef( i ) = t.derived().coeff( i ).value();
    return out;
}

/**
 * @brief Extract a partial derivative from each DA matrix/vector element (runtime multi-index).
 * @param alpha Multi-index specifying the derivative order per variable.
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < typename Derived, std::size_t M >
[[nodiscard]] auto derivative( const Eigen::DenseBase< Derived >& t,
                               const std::array< int, M >& alpha )
    requires( detail::eigen::is_da_v< typename Derived::Scalar > )
{
    using DA = typename Derived::Scalar;
    using T = typename detail::eigen::da_traits< DA >::scalar_type;
    static_assert( M == std::size_t( detail::eigen::da_traits< DA >::vars ),
                   "Derivative multi-index arity must match number of variables" );
    using Out = detail::eigen::rebind_matrix_t< Derived, T >;
    Out out( t.rows(), t.cols() );
    for ( Eigen::Index i = 0; i < t.size(); ++i )
        out.coeffRef( i ) = t.derived().coeff( i ).derivative( alpha );
    return out;
}

/**
 * @brief Extract the k-th time derivative from each univariate DA matrix/vector element.
 * @param k Derivative order (0 = value, 1 = first derivative, ...).
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < typename Derived >
[[nodiscard]] auto derivative( const Eigen::DenseBase< Derived >& t, int k )
    requires( detail::eigen::is_da_v< typename Derived::Scalar > &&
              detail::eigen::da_traits< typename Derived::Scalar >::vars == 1 )
{
    return derivative( t, MultiIndex< 1 >{ k } );
}

/**
 * @brief Extract a partial derivative from each DA matrix/vector element (compile-time multi-index).
 * @tparam Alpha Derivative orders for each variable.
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < int... Alpha, typename Derived >
[[nodiscard]] auto derivative( const Eigen::DenseBase< Derived >& t )
    requires( detail::eigen::is_da_v< typename Derived::Scalar > )
{
    using DA = typename Derived::Scalar;
    using T = typename detail::eigen::da_traits< DA >::scalar_type;
    constexpr int N = detail::eigen::da_traits< DA >::order;
    constexpr int M = detail::eigen::da_traits< DA >::vars;
    static_assert( sizeof...( Alpha ) == std::size_t( M ),
                   "Derivative multi-index arity must match number of variables" );
    static_assert( ( ( Alpha >= 0 ) && ... ), "Derivative orders must be non-negative" );
    constexpr int total_order = ( Alpha + ... + 0 );
    static_assert( total_order <= N, "Derivative total order exceeds DA truncation order" );

    using Out = detail::eigen::rebind_matrix_t< Derived, T >;
    Out out( t.rows(), t.cols() );
    for ( Eigen::Index i = 0; i < t.size(); ++i )
        out.coeffRef( i ) = t.derived().coeff( i ).template derivative< Alpha... >();
    return out;
}

/**
 * @brief Compute the gradient of a scalar DA at its expansion point.
 * @returns Eigen column vector `[df/dx_0, ..., df/dx_{M-1}]`.
 */
template < typename T, int N, int M >
[[nodiscard]] auto gradient( const TruncatedTaylorExpansionT< T, N, M >& f )
{
    static_assert( N >= 1, "gradient requires DA order >= 1" );
    Eigen::Matrix< T, M, 1 > g;
    for ( int i = 0; i < M; ++i )
    {
        MultiIndex< M > alpha{};
        alpha[i] = 1;
        g( i ) = f.derivative( alpha );
    }
    return g;
}

/**
 * @brief Compute the Jacobian matrix of a vector-valued DA function at its expansion point.
 * @param vec Eigen vector/matrix of DA elements (treated as a flat list of `K` components).
 * @returns Eigen matrix of shape `(K, M)` where `J(i,j) = df_i / dx_j`.
 */
template < typename Derived >
[[nodiscard]] auto jacobian( const Eigen::DenseBase< Derived >& vec )
    requires( detail::eigen::is_da_v< typename Derived::Scalar > )
{
    using DA = typename Derived::Scalar;
    using T = typename detail::eigen::da_traits< DA >::scalar_type;
    constexpr int M = detail::eigen::da_traits< DA >::vars;
    constexpr int K = Derived::SizeAtCompileTime;

    Eigen::Matrix< T, K, M > out( vec.size(), M );
    for ( Eigen::Index r = 0; r < vec.size(); ++r )
    {
        for ( int j = 0; j < M; ++j )
        {
            MultiIndex< M > alpha{};
            alpha[j] = 1;
            out( r, j ) = vec.derived().coeff( r ).derivative( alpha );
        }
    }
    return out;
}

/**
 * @brief Evaluate each DA element of an Eigen matrix/vector at displacement `dx`.
 * @param container Eigen matrix/vector of DA elements.
 * @param dx Displacement: scalar `T` (univariate), `point_type` (multivariate),
 *           or Eigen vector (multivariate, converted internally).
 * @returns Eigen matrix/vector of same shape with scalar type `T`.
 */
template < typename Derived, typename Dx >
[[nodiscard]] auto eval( const Eigen::DenseBase< Derived >& container, const Dx& dx )
    requires( detail::eigen::is_da_v< typename Derived::Scalar > )
{
    using DA = typename Derived::Scalar;
    using T = typename detail::eigen::da_traits< DA >::scalar_type;
    using Out = detail::eigen::rebind_matrix_t< Derived, T >;
    Out out( container.rows(), container.cols() );

    if constexpr ( detail::eigen::EigenDenseExpr< Dx > )
    {
        constexpr int M = detail::eigen::da_traits< DA >::vars;
        typename DA::point_type p{};
        for ( int i = 0; i < M; ++i )
            p[std::size_t( i )] = static_cast< T >( dx( Eigen::Index( i ) ) );
        for ( Eigen::Index i = 0; i < container.size(); ++i )
            out.coeffRef( i ) = container.derived().coeff( i ).eval( p );
    } else
    {
        for ( Eigen::Index i = 0; i < container.size(); ++i )
            out.coeffRef( i ) = container.derived().coeff( i ).eval( dx );
    }
    return out;
}

// =============================================================================
// Eigen::Tensor overloads (rank >= 1)
// =============================================================================

/**
 * @brief Extract the scalar value (constant term) from each DA element.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int M, int Rank >
[[nodiscard]] auto value( const Eigen::Tensor< TruncatedTaylorExpansionT< T, N, M >, Rank >& t )
    requires( Rank >= 1 )
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
[[nodiscard]] auto derivative( const Eigen::Tensor< TruncatedTaylorExpansionT< T, N, M >, Rank >& t,
                               const std::array< int, std::size_t( M ) >& alpha )
    requires( Rank >= 1 )
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
[[nodiscard]] auto derivative( const Eigen::Tensor< TruncatedTaylorExpansionT< T, N, 1 >, Rank >& t, int k )
    requires( Rank >= 1 )
{
    return derivative( t, MultiIndex< 1 >{ k } );
}

/**
 * @brief Extract a partial derivative from each DA element (compile-time multi-index).
 * @tparam Alpha Derivative orders for each variable.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < int... Alpha, typename T, int N, int M, int Rank >
[[nodiscard]] auto derivative( const Eigen::Tensor< TruncatedTaylorExpansionT< T, N, M >, Rank >& t )
    requires( Rank >= 1 )
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

/**
 * @brief Evaluate each DA element of an Eigen::Tensor at displacement `dx`.
 * @param t Eigen::Tensor of DA elements.
 * @param dx Displacement: scalar `T` (univariate), `point_type` (multivariate),
 *           or Eigen vector (multivariate, converted internally).
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int M, int Rank, typename Dx >
[[nodiscard]] auto eval( const Eigen::Tensor< TruncatedTaylorExpansionT< T, N, M >, Rank >& t, const Dx& dx )
    requires( Rank >= 1 )
{
    Eigen::Tensor< T, Rank > out( t.dimensions() );

    if constexpr ( detail::eigen::EigenDenseExpr< Dx > )
    {
        typename TruncatedTaylorExpansionT< T, N, M >::point_type p{};
        for ( int i = 0; i < M; ++i )
            p[std::size_t( i )] = static_cast< T >( dx( Eigen::Index( i ) ) );
        for ( Eigen::Index i = 0; i < t.size(); ++i ) out.data()[i] = t.data()[i].eval( p );
    } else
    {
        for ( Eigen::Index i = 0; i < t.size(); ++i ) out.data()[i] = t.data()[i].eval( dx );
    }
    return out;
}

}  // namespace tax
