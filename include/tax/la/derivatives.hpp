// Differential operators on TaylorExpansion (scalar TEs or Eigen matrices of TEs):
// derivative<Alpha...>, gradient, hessian, jacobian.

#pragma once

#include <Eigen/Core>
#include <tax/core/multi_index.hpp>
#include <tax/la/num_traits.hpp>

namespace tax::la
{

// -----------------------------------------------------------------------------
// derivative — element-wise partial derivative extraction
// -----------------------------------------------------------------------------

/// Extract a compile-time partial derivative from each element of an Eigen matrix/vector of
/// `TaylorExpansion` objects.
template < int... Alpha, typename Derived >
    requires( detail::is_taylor_te_v< typename Derived::Scalar > )
[[nodiscard]] auto derivative( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T = typename detail::te_traits< TE >::scalar_type;
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime > out( F.rows(),
                                                                                    F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i )
        out( i ) = F.derived().coeff( i ).template derivative< Alpha... >();
    return out;
}

// -----------------------------------------------------------------------------
// gradient — free-function wrapper around TaylorExpansion::gradient()
// -----------------------------------------------------------------------------

/// Compute the gradient of a scalar `TaylorExpansion` at its expansion point.
template < typename T, typename Scheme, typename S >
[[nodiscard]] Eigen::Matrix< T, Scheme::vars, 1 > gradient(
    const TaylorExpansion< T, Scheme, S >& f ) noexcept
{
    return f.gradient();
}

// -----------------------------------------------------------------------------
// hessian — free-function wrapper around TaylorExpansion::hessian()
// -----------------------------------------------------------------------------

/// Compute the Hessian matrix of a scalar `TaylorExpansion` at its expansion point.
template < typename T, typename Scheme, typename S >
[[nodiscard]] Eigen::Matrix< T, Scheme::vars, Scheme::vars > hessian(
    const TaylorExpansion< T, Scheme, S >& f ) noexcept
{
    return f.hessian();
}

// -----------------------------------------------------------------------------
// jacobian — Jacobian of a vector-valued TE function
// -----------------------------------------------------------------------------

/// Compute the Jacobian matrix of a vector-valued `TaylorExpansion` function.
template < typename Derived >
    requires( detail::is_taylor_te_v< typename Derived::Scalar > )
[[nodiscard]] auto jacobian( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using tr = detail::te_traits< TE >;
    using T = typename tr::scalar_type;
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

// =============================================================================
// Basis-generic differential operators — evaluated at an explicit point.
//
// For a non-Taylor family the partial derivatives are not coefficients; they are
// obtained by differentiating the polynomial (`deriv`) and evaluating at a point.
// These point-form overloads work for ANY basis (Taylor included, with the point
// being the displacement from x0).
// =============================================================================

namespace detail
{
template < typename E, typename PtDerived >
[[nodiscard]] typename E::Input toPoint( const Eigen::MatrixBase< PtDerived >& at )
{
    using T = typename te_traits< E >::scalar_type;
    constexpr int M = te_traits< E >::vars_v;
    static_assert(
        PtDerived::SizeAtCompileTime == M || PtDerived::SizeAtCompileTime == Eigen::Dynamic,
        "point size must match the expansion's variable count" );
    typename E::Input p{};
    for ( int i = 0; i < M; ++i ) p[std::size_t( i )] = T( at( i ) );
    return p;
}
}  // namespace detail

/// Gradient `[∂f/∂x_0, …]` of a scalar expansion evaluated at point `at`.
template < typename T, typename Basis, typename Scheme, typename S, typename PtDerived >
[[nodiscard]] Eigen::Matrix< T, Scheme::vars, 1 > gradient(
    const Expansion< T, Basis, Scheme, S >& f, const Eigen::MatrixBase< PtDerived >& at )
{
    using E = Expansion< T, Basis, Scheme, S >;
    const typename E::Input p = detail::toPoint< E >( at );
    Eigen::Matrix< T, Scheme::vars, 1 > g;
    for ( int i = 0; i < Scheme::vars; ++i ) g( i ) = f.deriv( i ).eval( p );
    return g;
}

/// Hessian `H(i,j) = ∂²f/∂x_i∂x_j` of a scalar expansion evaluated at point `at`.
template < typename T, typename Basis, typename Scheme, typename S, typename PtDerived >
[[nodiscard]] Eigen::Matrix< T, Scheme::vars, Scheme::vars > hessian(
    const Expansion< T, Basis, Scheme, S >& f, const Eigen::MatrixBase< PtDerived >& at )
{
    using E = Expansion< T, Basis, Scheme, S >;
    const typename E::Input p = detail::toPoint< E >( at );
    Eigen::Matrix< T, Scheme::vars, Scheme::vars > H;
    for ( int i = 0; i < Scheme::vars; ++i )
        for ( int j = 0; j < Scheme::vars; ++j ) H( i, j ) = f.deriv( i ).deriv( j ).eval( p );
    return H;
}

/// Jacobian of a vector-valued expansion function evaluated at point `at`.
template < typename Derived, typename PtDerived >
    requires( detail::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto jacobian( const Eigen::MatrixBase< Derived >& F,
                             const Eigen::MatrixBase< PtDerived >& at )
{
    using E = typename Derived::Scalar;
    using tr = detail::te_traits< E >;
    using T = typename tr::scalar_type;
    constexpr int M = tr::vars_v;
    constexpr int K = Derived::SizeAtCompileTime;
    const typename E::Input p = detail::toPoint< E >( at );
    Eigen::Matrix< T, K, M > out( F.size(), M );
    for ( Eigen::Index r = 0; r < F.size(); ++r )
        for ( int j = 0; j < M; ++j ) out( r, j ) = F.derived().coeff( r ).deriv( j ).eval( p );
    return out;
}

}  // namespace tax::la
