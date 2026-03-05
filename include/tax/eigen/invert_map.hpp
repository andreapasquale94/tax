#pragma once

#include <Eigen/LU>
#include <stdexcept>
#include <tax/eigen/derivative.hpp>

namespace tax
{

namespace detail
{

template < typename DA >
[[nodiscard]] auto identityMap()
{
    constexpr int M = ::tax::detail::expansion_traits< DA >::vars;
    using Map = Eigen::Matrix< DA, M, 1 >;

    Map out{};
    typename DA::Input x0{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out( Eigen::Index( I ) ) = DA::template variable< int( I ) >( x0 ) ), ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );
    return out;
}

template < typename DA, typename Mat >
[[nodiscard]] auto linearMap( const Mat& a, const Eigen::Matrix< DA, Mat::ColsAtCompileTime, 1 >& vars )
{
    using Map = Eigen::Matrix< DA, Mat::RowsAtCompileTime, 1 >;
    Map out{};

    for ( Eigen::Index i = 0; i < a.rows(); ++i )
    {
        out( i ) = DA::zero();
        for ( Eigen::Index j = 0; j < a.cols(); ++j ) out( i ) += a( i, j ) * vars( j );
    }
    return out;
}

template < typename DA, typename Map >
[[nodiscard]] auto composeOne( const DA& f, const Map& g )
{
    using T = typename ::tax::detail::expansion_traits< DA >::scalar_type;
    constexpr int N = ::tax::detail::expansion_traits< DA >::order;
    constexpr int M = ::tax::detail::expansion_traits< DA >::vars;

    DA out = DA::zero();
    for ( int d = 0; d <= N; ++d )
    {
        ::tax::detail::forEachMonomial< M >( d, [&]( const MultiIndex< M >& alpha, std::size_t ) {
            const T coeff = f.coeff( alpha );
            if ( coeff == T{} ) return;

            DA term = DA::constant( coeff );
            for ( int j = 0; j < M; ++j )
                for ( int k = 0; k < alpha[j]; ++k ) term *= g( j );
            out += term;
        } );
    }
    return out;
}

template < typename DA, typename Map >
[[nodiscard]] auto composeMap( const Map& a, const Map& b )
{
    Map out{};
    for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = composeOne< DA >( a( i ), b );
    return out;
}

}  // namespace detail

/**
 * @brief Invert a square Taylor map represented as an Eigen vector of TTE components.
 * @details The constant terms of the input components are ignored. The inverse is computed by
 * inverting the linear part at the expansion point and applying Picard iterations up to order `N`.
 *
 * @param map_in Eigen vector containing `M` TTE components with the same type.
 * @return Inverse map with the same Eigen shape as `map_in`.
 *
 * @throws std::invalid_argument If the input size is not `M` or the linear part is not invertible.
 */
template < typename Derived >
[[nodiscard]] auto invert( const Eigen::DenseBase< Derived >& map_in )
    requires( detail::is_tte_v< typename Derived::Scalar > )
{
    using DA = typename Derived::Scalar;
    using T = typename detail::expansion_traits< DA >::scalar_type;
    constexpr int N = detail::expansion_traits< DA >::order;
    constexpr int M = detail::expansion_traits< DA >::vars;

    static_assert( Derived::RowsAtCompileTime == 1 || Derived::ColsAtCompileTime == 1 ||
                       Derived::RowsAtCompileTime == Eigen::Dynamic ||
                       Derived::ColsAtCompileTime == Eigen::Dynamic,
                   "invert expects an Eigen vector expression" );
    static_assert( Derived::SizeAtCompileTime == Eigen::Dynamic || Derived::SizeAtCompileTime == M,
                   "invert expects map size equal to number of variables M" );

    if ( map_in.size() != Eigen::Index( M ) )
        throw std::invalid_argument(
            "invert expects a square map: vector size must match number of variables" );

    using Map = Eigen::Matrix< DA, M, 1 >;
    using Mat = Eigen::Matrix< T, M, M >;

    Map map{};
    for ( Eigen::Index i = 0; i < map_in.size(); ++i )
    {
        map( i ) = map_in.derived().coeff( i );
        // The inversion is defined for the non-constant map terms.
        map( i )[0] = T{};
    }

    const Map I = detail::identityMap< DA >();
    const Mat J = jacobian( map );
    const Eigen::FullPivLU< Mat > lu( J );
    if ( !lu.isInvertible() )
        throw std::invalid_argument( "invert failed: linear part is singular" );

    const Mat Jinv = lu.inverse();
    const Map Mlin = detail::linearMap< DA >( J, I );
    const Map Minv = detail::linearMap< DA >( Jinv, I );

    Map nonlinear{};
    for ( int i = 0; i < M; ++i ) nonlinear( i ) = map( i ) - Mlin( i );

    Map out = Minv;
    for ( int k = 1; k < N; ++k )
    {
        const Map composed_nonlin = detail::composeMap< DA >( nonlinear, out );
        Map correction{};
        for ( int i = 0; i < M; ++i ) correction( i ) = I( i ) - composed_nonlin( i );
        out = detail::composeMap< DA >( Minv, correction );
    }

    using Out = detail::rebind_matrix_t< Derived, DA >;
    Out ret( map_in.rows(), map_in.cols() );
    for ( Eigen::Index i = 0; i < map_in.size(); ++i ) ret.coeffRef( i ) = out( i );
    return ret;
}

}  // namespace tax
