#pragma once

#include <tax/da.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>

namespace tax
{

namespace detail::tensor
{

[[nodiscard]] constexpr std::size_t factorial( int n ) noexcept
{
    std::size_t out = 1;
    for ( int i = 2; i <= n; ++i ) out *= std::size_t( i );
    return out;
}

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

template < typename DA, int... Alpha >
[[nodiscard]] constexpr auto derivativeFromAlphaSeq(
    const DA& f, std::integer_sequence< int, Alpha... > ) noexcept
{
    return f.template derivative< Alpha... >();
}

template < typename DA, int... Alpha >
[[nodiscard]] constexpr auto coeffFromAlphaSeq( const DA& f,
                                                std::integer_sequence< int, Alpha... > ) noexcept
{
    return f.template coeff< Alpha... >();
}

template < int... Alpha >
[[nodiscard]] consteval std::size_t multinomialFromAlpha() noexcept
{
    constexpr int total_order = ( Alpha + ... + 0 );
    std::size_t denom = 1;
    ( ( denom *= factorial( Alpha ) ), ... );
    return factorial( total_order ) / denom;
}

template < int... Alpha >
[[nodiscard]] consteval std::size_t multinomialFromAlphaSeq(
    std::integer_sequence< int, Alpha... > ) noexcept
{
    return multinomialFromAlpha< Alpha... >();
}

}  // namespace detail::tensor

/**
 * @brief Build the order-`K` derivative tensor at the expansion point.
 * @details Tensor entry `(i_1, ..., i_K)` equals
 * `d^K f / (dx_{i_1} ... dx_{i_K})` evaluated at the expansion point.
 */
template < int K, typename T, int N, int M >
[[nodiscard]] auto derivativeTensor( const TDA< T, N, M >& f )
{
    static_assert( M > 1, "Eigen tensors are only provided for multivariate DA (M > 1)" );
    static_assert( K >= 1, "Tensor order K must be at least 1" );
    static_assert( K <= N, "Tensor order K exceeds DA truncation order N" );

    Eigen::Tensor< T, K, Eigen::RowMajor > out{
        detail::tensor::tensorDims< K >( Eigen::Index( M ) ) };

    auto fill = [&]< int... I >( std::integer_sequence< int, I... > ) {
        using alpha_seq = detail::tensor::alpha_sequence_t< M, I... >;
        out( Eigen::Index( I )... ) = detail::tensor::derivativeFromAlphaSeq( f, alpha_seq{} );
    };

    detail::tensor::forEachIndexTuple< 0, K, M >( fill );
    return out;
}

/**
 * @brief Build the order-`K` coefficient tensor.
 * @details Tensor entry `(i_1, ..., i_K)` is normalized so that summing
 * `C(i_1,...,i_K) * dx_{i_1}...dx_{i_K}` reproduces the order-`K` polynomial part.
 */
template < int K, typename T, int N, int M >
[[nodiscard]] auto coeffTensor( const TDA< T, N, M >& f )
{
    static_assert( M > 1, "Eigen tensors are only provided for multivariate DA (M > 1)" );
    static_assert( K >= 1, "Tensor order K must be at least 1" );
    static_assert( K <= N, "Tensor order K exceeds DA truncation order N" );

    Eigen::Tensor< T, K, Eigen::RowMajor > out{
        detail::tensor::tensorDims< K >( Eigen::Index( M ) ) };

    auto fill = [&]< int... I >( std::integer_sequence< int, I... > ) {
        using alpha_seq = detail::tensor::alpha_sequence_t< M, I... >;
        constexpr auto perms = detail::tensor::multinomialFromAlphaSeq( alpha_seq{} );
        out( Eigen::Index( I )... ) =
            detail::tensor::coeffFromAlphaSeq( f, alpha_seq{} ) / T( perms );
    };

    detail::tensor::forEachIndexTuple< 0, K, M >( fill );
    return out;
}

}  // namespace tax
