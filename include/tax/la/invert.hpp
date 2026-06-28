// include/tax/la/invert.hpp
//
// Formal inversion of a polynomial map represented as an Eigen
// vector of `TaylorExpansion` components, plus the helper
// machinery (identityMap, composeOne, composeMap, linear) that the
// Picard-iteration loop is built out of.

#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <stdexcept>
#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/la/derivatives.hpp>
#include <tax/la/num_traits.hpp>
#include <utility>

namespace tax::la::detail
{

/// Build the identity map as `Eigen::Matrix<TE, M, 1>` around the zero expansion point.
template < typename TE >
[[nodiscard]] auto identityMap()
{
    constexpr int M = te_traits< TE >::vars_v;
    using Map = Eigen::Matrix< TE, M, 1 >;
    Map out{};
    typename TE::Input x0{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out( Eigen::Index( I ) ) = TE::template variable< int( I ) >( x0 ) ), ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );
    return out;
}

/// Compose a single scalar TE `f` with a map `g` (substitution: x_j -> g_j).
template < typename TE, typename Map >
[[nodiscard]] TE composeOne( const TE& f, const Map& g )
{
    using T = typename te_traits< TE >::scalar_type;
    constexpr int N = te_traits< TE >::order_v;
    constexpr int M = te_traits< TE >::vars_v;

    TE out = TE::zero();
    for ( int d = 0; d <= N; ++d )
    {
        forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
            const T coeff = f.coeff( alpha );
            if ( coeff == T{} ) return;
            TE term = TE::constant( coeff );
            for ( int j = 0; j < M; ++j )
                for ( int k = 0; k < alpha[std::size_t( j )]; ++k ) term = term * g( j );
            out = out + term;
        } );
    }
    return out;
}

/// Compose a vector map `a` with map `b` component-wise.
template < typename TE, typename Map >
[[nodiscard]] Map composeMap( const Map& a, const Map& b )
{
    Map out{};
    for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = composeOne< TE >( a( i ), b );
    return out;
}

/// Build the linear map `J * vars` as a TE vector.
template < typename TE, typename Mat >
[[nodiscard]] Eigen::Matrix< TE, Mat::RowsAtCompileTime, 1 > linear(
    const Mat& J, const Eigen::Matrix< TE, Mat::ColsAtCompileTime, 1 >& vars )
{
    Eigen::Matrix< TE, Mat::RowsAtCompileTime, 1 > out{};
    for ( Eigen::Index i = 0; i < J.rows(); ++i )
    {
        out( i ) = TE::zero();
        for ( Eigen::Index j = 0; j < J.cols(); ++j ) out( i ) = out( i ) + J( i, j ) * vars( j );
    }
    return out;
}

}  // namespace tax::la::detail

namespace tax::la
{

/// Formally invert a square polynomial map (perturbation part only; constant terms are
/// dropped). Requires an invertible linear part; throws std::invalid_argument if singular.
template < typename Derived >
    requires( detail::is_taylor_te_v< typename Derived::Scalar > )
[[nodiscard]] auto invert( const Eigen::MatrixBase< Derived >& map_in )
{
    using TE = typename Derived::Scalar;
    using T = typename detail::te_traits< TE >::scalar_type;
    constexpr int N = detail::te_traits< TE >::order_v;
    constexpr int M = detail::te_traits< TE >::vars_v;

    static_assert( Derived::SizeAtCompileTime == M || Derived::SizeAtCompileTime == Eigen::Dynamic,
                   "invert: map size must equal number of variables M" );

    using Map = Eigen::Matrix< TE, M, 1 >;
    using Mat = Eigen::Matrix< T, M, M >;

    // Build the non-constant map (zero out constant terms).
    Map map{};
    for ( Eigen::Index i = 0; i < map_in.size(); ++i )
    {
        map( i ) = map_in.derived().coeff( i );
        map( i )[0] = T{};  // drop constant term
    }

    // Build the identity map and Jacobian of the non-constant map.
    const Map I = detail::identityMap< TE >();
    const Mat J = jacobian( map );
    const Eigen::FullPivLU< Mat > lu( J );
    if ( !lu.isInvertible() )
        throw std::invalid_argument( "invert failed: linear part is singular" );

    const Mat Jinv = lu.inverse();
    const Map Mlin = detail::linear< TE >( J, I );
    const Map Minv = detail::linear< TE >( Jinv, I );

    // nonlinear = map - linear_part
    Map nonlinear{};
    for ( int i = 0; i < M; ++i ) nonlinear( i ) = map( i ) - Mlin( i );

    // Picard iteration: out_{k+1} = Jinv * (I - nonlinear ∘ out_k)
    Map out = Minv;
    for ( int k = 1; k < N; ++k )
    {
        const Map composed_nonlin = detail::composeMap< TE >( nonlinear, out );
        Map correction{};
        for ( int i = 0; i < M; ++i ) correction( i ) = I( i ) - composed_nonlin( i );
        out = detail::composeMap< TE >( Minv, correction );
    }

    // Copy back into an output with the same Eigen shape as map_in.
    detail::rebind_matrix_t< Derived, TE > ret( map_in.rows(), map_in.cols() );
    for ( Eigen::Index i = 0; i < map_in.size(); ++i ) ret.coeffRef( i ) = out( i );
    return ret;
}

}  // namespace tax::la
