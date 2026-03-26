#pragma once

#include <tax/eigen/variables.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tax
{

namespace detail::tensor
{

template < int K >
[[nodiscard]] constexpr Eigen::array< Eigen::Index, K > tensorDims( Eigen::Index extent ) noexcept
{
    Eigen::array< Eigen::Index, K > dims{};
    for ( int i = 0; i < K; ++i ) dims[i] = extent;
    return dims;
}

template < int Pos, int K, int M, typename F, int... I >
constexpr void forEachIndexTuple( F& f )
{
    if constexpr ( Pos == K )
    {
        f( std::integer_sequence< int, I... >{} );
    } else
    {
        [&]< std::size_t... J >( std::index_sequence< J... > ) {
            ( forEachIndexTuple< Pos + 1, K, M, F, I..., int( J ) >( f ), ... );
        }( std::make_index_sequence< std::size_t( M ) >{} );
    }
}

template < int Var, int... I >
[[nodiscard]] consteval int countInTuple() noexcept
{
    return ( ( I == Var ? 1 : 0 ) + ... + 0 );
}

template < int M, int... I, std::size_t... V >
[[nodiscard]] consteval auto alphaSequenceImpl( std::index_sequence< V... > ) noexcept
{
    return std::integer_sequence< int, countInTuple< int( V ), I... >()... >{};
}

template < int M, int... I >
using alpha_sequence_t =
    decltype( alphaSequenceImpl< M, I... >( std::make_index_sequence< std::size_t( M ) >{} ) );

template < typename TTE, int... Alpha >
[[nodiscard]] constexpr auto derivativeFromAlphaSeq(
    const TTE& f, std::integer_sequence< int, Alpha... > ) noexcept
{
    return f.template derivative< Alpha... >();
}

}  // namespace detail::tensor

// =============================================================================
// DenseBase (Matrix / Vector) overloads
// =============================================================================

/**
 * @brief Extract a partial derivative from each TTE matrix/vector element (runtime multi-index).
 * @param alpha Multi-index specifying the derivative order per variable.
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < typename Derived, std::size_t M >
[[nodiscard]] auto derivative( const Eigen::DenseBase< Derived >& t,
                               const std::array< int, M >& alpha )
    requires( detail::is_tte_v< typename Derived::Scalar > )
{
    using TTE = typename Derived::Scalar;
    using T = typename detail::expansion_traits< TTE >::scalar_type;
    static_assert( M == std::size_t( detail::expansion_traits< TTE >::vars ),
                   "Derivative multi-index arity must match number of variables" );
    using Out = detail::rebind_matrix_t< Derived, T >;
    Out out( t.rows(), t.cols() );
    for ( Eigen::Index i = 0; i < t.size(); ++i )
        out.coeffRef( i ) = t.derived().coeff( i ).derivative( alpha );
    return out;
}

/**
 * @brief Extract the k-th time derivative from each univariate TTE matrix/vector element.
 * @param k Derivative order (0 = value, 1 = first derivative, ...).
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < typename Derived >
[[nodiscard]] auto derivative( const Eigen::DenseBase< Derived >& t, int k )
    requires( detail::is_tte_v< typename Derived::Scalar > &&
              detail::expansion_traits< typename Derived::Scalar >::vars == 1 )
{
    return derivative( t, MultiIndex< 1 >{ k } );
}

/**
 * @brief Extract a partial derivative from each TTE matrix/vector element (compile-time
 * multi-index).
 * @tparam Alpha Derivative orders for each variable.
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < int... Alpha, typename Derived >
[[nodiscard]] auto derivative( const Eigen::DenseBase< Derived >& t )
    requires( detail::is_tte_v< typename Derived::Scalar > )
{
    using TTE = typename Derived::Scalar;
    using T = typename detail::expansion_traits< TTE >::scalar_type;
    constexpr int N = detail::expansion_traits< TTE >::order;
    constexpr int M = detail::expansion_traits< TTE >::vars;
    static_assert( sizeof...( Alpha ) == std::size_t( M ),
                   "Derivative multi-index arity must match number of variables" );
    static_assert( ( ( Alpha >= 0 ) && ... ), "Derivative orders must be non-negative" );
    constexpr int total_order = ( Alpha + ... + 0 );
    static_assert( total_order <= N, "Derivative total order exceeds TTE truncation order" );

    using Out = detail::rebind_matrix_t< Derived, T >;
    Out out( t.rows(), t.cols() );
    for ( Eigen::Index i = 0; i < t.size(); ++i )
        out.coeffRef( i ) = t.derived().coeff( i ).template derivative< Alpha... >();
    return out;
}

/**
 * @brief Compute the gradient of a scalar TTE at its expansion point.
 * @returns Eigen column vector `[df/dx_0, ..., df/dx_{M-1}]`.
 */
template < typename T, int N, int M >
[[nodiscard]] auto gradient( const TruncatedExpansionT< T, N, M >& f )
{
    static_assert( N >= 1, "gradient requires TTE order >= 1" );
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
 * @brief Compute the Jacobian matrix of a vector-valued TTE function at its expansion point.
 * @param vec Eigen vector/matrix of TTE elements (treated as a flat list of `K` components).
 * @returns Eigen matrix of shape `(K, M)` where `J(i,j) = df_i / dx_j`.
 */
template < typename Derived >
[[nodiscard]] auto jacobian( const Eigen::DenseBase< Derived >& vec )
    requires( detail::is_tte_v< typename Derived::Scalar > )
{
    using TTE = typename Derived::Scalar;
    using T = typename detail::expansion_traits< TTE >::scalar_type;
    constexpr int M = detail::expansion_traits< TTE >::vars;
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

// =============================================================================
// Eigen::Tensor overloads (rank >= 1)
// =============================================================================

/**
 * @brief Extract a partial derivative from each TTE element (runtime multi-index).
 * @param alpha Multi-index specifying the derivative order per variable.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int M, int Rank >
[[nodiscard]] auto derivative( const Eigen::Tensor< TruncatedExpansionT< T, N, M >, Rank >& t,
                               const std::array< int, std::size_t( M ) >& alpha )
    requires( Rank >= 1 )
{
    Eigen::Tensor< T, Rank > out( t.dimensions() );
    for ( Eigen::Index i = 0; i < t.size(); ++i ) out.data()[i] = t.data()[i].derivative( alpha );
    return out;
}

/**
 * @brief Extract the k-th time derivative from each univariate TTE element.
 * @param k Derivative order (0 = value, 1 = first derivative, ...).
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int Rank >
[[nodiscard]] auto derivative( const Eigen::Tensor< TruncatedExpansionT< T, N, 1 >, Rank >& t,
                               int k )
    requires( Rank >= 1 )
{
    return derivative( t, MultiIndex< 1 >{ k } );
}

/**
 * @brief Extract a partial derivative from each TTE element (compile-time multi-index).
 * @tparam Alpha Derivative orders for each variable.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < int... Alpha, typename T, int N, int M, int Rank >
[[nodiscard]] auto derivative(
    const Eigen::Tensor< TruncatedExpansionT< T, N, M >, Rank >& t )
    requires( Rank >= 1 )
{
    static_assert( sizeof...( Alpha ) == M,
                   "Derivative multi-index arity must match number of variables" );
    static_assert( ( ( Alpha >= 0 ) && ... ), "Derivative orders must be non-negative" );
    constexpr int total_order = ( Alpha + ... + 0 );
    static_assert( total_order <= N, "Derivative total order exceeds TTE truncation order" );

    Eigen::Tensor< T, Rank > out( t.dimensions() );
    for ( Eigen::Index i = 0; i < t.size(); ++i )
        out.data()[i] = t.data()[i].template derivative< Alpha... >();
    return out;
}

/**
 * @brief Build the order-`K` derivative object at the expansion point.
 * @details
 * - `K == 0`: returns scalar value `f(x0)`.
 * - `K == 1`: returns gradient as `Eigen::Matrix<T, M, 1>`.
 * - `K == 2`: returns Hessian as `Eigen::Matrix<T, M, M>`.
 * - `K >= 3`: returns `Eigen::Tensor<T, K, Eigen::RowMajor>`.
 */
template < int K, typename T, int N, int M >
[[nodiscard]] auto derivative( const TruncatedExpansionT< T, N, M >& f )
{
    static_assert( M > 1, "Eigen derivative objects are only provided for multivariate TTE (M > 1)" );
    static_assert( K >= 0, "Derivative order K must be non-negative" );
    static_assert( K <= N, "Tensor order K exceeds TTE truncation order N" );

    if constexpr ( K == 0 )
    {
        return f.value();
    } else if constexpr ( K == 1 )
    {
        Eigen::Matrix< T, M, 1 > out;
        for ( int i = 0; i < M; ++i )
        {
            MultiIndex< M > alpha{};
            alpha[i] = 1;
            out( i ) = f.derivative( alpha );
        }
        return out;
    } else if constexpr ( K == 2 )
    {
        Eigen::Matrix< T, M, M > out;
        for ( int i = 0; i < M; ++i )
        {
            for ( int j = 0; j < M; ++j )
            {
                MultiIndex< M > alpha{};
                alpha[i] += 1;
                alpha[j] += 1;
                out( i, j ) = f.derivative( alpha );
            }
        }
        return out;
    } else
    {
        Eigen::Tensor< T, K, Eigen::RowMajor > out{
            detail::tensor::tensorDims< K >( Eigen::Index( M ) ) };

        auto fill = [&]< int... I >( std::integer_sequence< int, I... > ) {
            using alpha_seq = detail::tensor::alpha_sequence_t< M, I... >;
            out( Eigen::Index( I )... ) = detail::tensor::derivativeFromAlphaSeq( f, alpha_seq{} );
        };

        detail::tensor::forEachIndexTuple< 0, K, M >( fill );
        return out;
    }
}

}  // namespace tax
