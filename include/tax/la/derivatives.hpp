// Differential operators on TaylorExpansion (scalar TEs or Eigen matrices of TEs):
// derivative<Alpha...>, gradient, hessian, jacobian.

#pragma once

#include <Eigen/Core>

#include <tax/core/multi_index.hpp>
#include <tax/la/num_traits.hpp>

namespace tax::la
{

/// Element-wise compile-time partial derivative of an Eigen matrix/vector of TEs.
template < int... Alpha, typename Derived >
    requires( detail::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto derivative( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::te_traits< TE >::scalar_type;
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime >
        out( F.rows(), F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i )
        out( i ) = F.derived().coeff( i ).template derivative< Alpha... >();
    return out;
}

template < typename T, typename Scheme >
[[nodiscard]] Eigen::Matrix< T, Scheme::vars, 1 > gradient(
    const TaylorExpansion< T, Scheme >& f ) noexcept
{
    return f.gradient();
}

template < typename T, typename Scheme >
[[nodiscard]] Eigen::Matrix< T, Scheme::vars, Scheme::vars > hessian(
    const TaylorExpansion< T, Scheme >& f ) noexcept
{
    return f.hessian();
}

/// Jacobian of a vector-valued `TaylorExpansion` function.
template < typename Derived >
    requires( detail::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto jacobian( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using tr = detail::te_traits< TE >;
    using T  = typename tr::scalar_type;
    constexpr int M = tr::vars_v;
    constexpr int K = Derived::SizeAtCompileTime;
    Eigen::Matrix< T, K, M > out( F.size(), M );
    for ( Eigen::Index r = 0; r < F.size(); ++r )
    {
        MultiIndex< M > alpha{};
        for ( int j = 0; j < M; ++j )
        {
            alpha[std::size_t( j )] = 1;
            out( r, j ) = F.derived().coeff( r ).derivative( alpha );
            alpha[std::size_t( j )] = 0;
        }
    }
    return out;
}

}  // namespace tax::la
