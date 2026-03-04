#pragma once

#include <tax/eigen/variables.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tax
{

/**
 * @brief Evaluate scalar TTE at a displacement type accepted by `TruncatedTaylorExpansionT::eval`.
 */
template < typename T, int N, int M, typename Dx >
[[nodiscard]] constexpr T eval( const TruncatedTaylorExpansionT< T, N, M >& f,
                                const Dx& dx ) noexcept
    requires( requires { f.eval( dx ); } )
{
    return f.eval( dx );
}

/**
 * @brief Evaluate a TTE polynomial at displacement given as an Eigen vector.
 * @param f The TTE polynomial.
 * @param dx Eigen vector with `M` entries holding the displacement.
 * @return `f(x0 + dx)` truncated to order `N`.
 */
template < typename T, int N, int M, typename Derived >
[[nodiscard]] T eval( const TruncatedTaylorExpansionT< T, N, M >& f,
                      const Eigen::DenseBase< Derived >& dx ) noexcept
    requires( M > 1 && std::convertible_to< typename Derived::Scalar, T > )
{
    typename TruncatedTaylorExpansionT< T, N, M >::Input p{};
    for ( int i = 0; i < M; ++i ) p[std::size_t( i )] = static_cast< T >( dx( Eigen::Index( i ) ) );
    return f.eval( p );
}

/**
 * @brief Evaluate each TTE element of an Eigen matrix/vector at displacement `dx`.
 * @param f Eigen matrix/vector of TTE elements.
 * @param dx Displacement: scalar `T` (univariate), `Input` (multivariate),
 *           or Eigen vector (multivariate, converted internally).
 * @returns Eigen matrix/vector of same shape with scalar type `T`.
 */
template < typename Derived, typename Dx >
[[nodiscard]] auto eval( const Eigen::DenseBase< Derived >& f, const Dx& dx )
    requires( detail::is_tte_v< typename Derived::Scalar > )
{
    using TTE = typename Derived::Scalar;
    using T = typename detail::expansion_traits< TTE >::scalar_type;
    using Out = detail::rebind_matrix_t< Derived, T >;
    Out out( f.rows(), f.cols() );

    if constexpr ( detail::EigenDenseExpr< Dx > )
    {
        constexpr int M = detail::expansion_traits< TTE >::vars;
        typename TTE::Input p{};
        for ( int i = 0; i < M; ++i ) p[i] = static_cast< T >( dx( i ) );
        for ( Eigen::Index i = 0; i < f.size(); ++i )
            out.coeffRef( i ) = f.derived().coeff( i ).eval( p );
    } else
    {
        for ( Eigen::Index i = 0; i < f.size(); ++i )
            out.coeffRef( i ) = f.derived().coeff( i ).eval( dx );
    }
    return out;
}

/**
 * @brief Evaluate each TTE element of an Eigen::Tensor at displacement `dx`.
 * @param t Eigen::Tensor of TTE elements.
 * @param dx Displacement: scalar `T` (univariate), `Input` (multivariate),
 *           or Eigen vector (multivariate, converted internally).
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int M, int Rank, typename Dx >
[[nodiscard]] auto eval( const Eigen::Tensor< TruncatedTaylorExpansionT< T, N, M >, Rank >& t,
                         const Dx& dx )
    requires( Rank >= 1 )
{
    Eigen::Tensor< T, Rank > out( t.dimensions() );

    if constexpr ( detail::EigenDenseExpr< Dx > )
    {
        typename TruncatedTaylorExpansionT< T, N, M >::Input p{};
        for ( int i = 0; i < M; ++i )
            p[std::size_t( i )] = static_cast< T >( dx( Eigen::Index( i ) ) );
        for ( Eigen::Index i = 0; i < t.size(); ++i ) out.data()[i] = t.data()[i].eval( p );
    } else
    {
        for ( Eigen::Index i = 0; i < t.size(); ++i ) out.data()[i] = t.data()[i].eval( dx );
    }
    return out;
}

/**
 * @brief Vectorised evaluation of a univariate TTE column-vector at scalar @p dx.
 *
 * Uses a single matrix-vector product `C * [1, h, h², …, hᴺ]ᵀ` where `C(i,k)=y_da(i)[k]`.
 */
template < typename T, int N, int Dim >
[[nodiscard]] Eigen::Matrix< T, Dim, 1 > eval(
    const Eigen::Matrix< TruncatedTaylorExpansionT< T, N, 1 >, Dim, 1 >& f, T dx ) noexcept
{
    constexpr int N1 = N + 1;
    const Eigen::Index dim = f.size();

    Eigen::Matrix< T, Dim, N1 > C( dim, N1 );
    for ( Eigen::Index i = 0; i < dim; ++i )
    {
        const auto& c = f( i ).coeffs();
        for ( int k = 0; k < N1; ++k ) C( i, k ) = c[k];
    }

    Eigen::Matrix< T, N1, 1 > out;
    out( 0 ) = T( 1 );
    for ( int k = 1; k < N1; ++k ) out( k ) = out( k - 1 ) * dx;

    return C * out;
}

}  // namespace tax
