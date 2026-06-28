// Builders/evaluators between scalar and Eigen-matrix forms of TaylorExpansion:
// variables(x0), value(F), eval(f|F, dx).

#pragma once

#include <Eigen/Core>
#include <type_traits>
#include <utility>

#include <tax/eigen/num_traits.hpp>

namespace tax::la
{

// -----------------------------------------------------------------------------
// variables — build an Eigen column vector of TE coordinate variables
// -----------------------------------------------------------------------------

/// Construct an `Eigen::Matrix<TE, M, 1>` of coordinate Taylor variables from an Eigen vector expansion point `x0`.
template < typename TE, typename Derived >
[[nodiscard]] auto variables( const Eigen::MatrixBase< Derived >& x0 )
{
    using tr = detail::te_traits< TE >;
    using T  = typename tr::scalar_type;
    constexpr int M = tr::vars_v;
    static_assert( Derived::SizeAtCompileTime == M || Derived::SizeAtCompileTime == Eigen::Dynamic,
                   "variables(): Eigen input size must match TE::vars_v" );
    typename TE::Input p{};
    for ( int i = 0; i < M; ++i ) p[std::size_t( i )] = T( x0( i ) );
    Eigen::Matrix< TE, M, 1 > out;
    [&]< std::size_t... I >( std::index_sequence< I... > )
    {
        ( ( out( int( I ) ) = TE::template variable< int( I ) >( p ) ), ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );
    return out;
}

// -----------------------------------------------------------------------------
// value — extract constant terms from an Eigen matrix of TE objects
// -----------------------------------------------------------------------------

/// Extract the constant term (value at expansion point) from each element of an Eigen matrix/vector of `TaylorExpansion` objects.
template < typename Derived >
    requires( detail::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto value( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::te_traits< TE >::scalar_type;
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime >
        out( F.rows(), F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i ) out( i ) = F.derived().coeff( i ).value();
    return out;
}

// -----------------------------------------------------------------------------
// eval — evaluate scalar TE or Eigen matrix of TEs at a displacement
// -----------------------------------------------------------------------------

/// Evaluate a scalar `TaylorExpansion` at displacement `dx`.
template < typename T, typename Scheme, typename S, typename DxDerived >
[[nodiscard]] T eval( const TaylorExpansion< T, Scheme, S >& f,
                      const Eigen::MatrixBase< DxDerived >& dx ) noexcept
{
    return f.eval( dx );
}

/// Evaluate each element of an Eigen matrix/vector of `TaylorExpansion` objects at a shared displacement vector `dx`.
template < typename Derived, typename DxDerived >
    requires( detail::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto eval( const Eigen::MatrixBase< Derived >& F,
                         const Eigen::MatrixBase< DxDerived >& dx )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::te_traits< TE >::scalar_type;
    constexpr int M = detail::te_traits< TE >::vars_v;
    static_assert( DxDerived::SizeAtCompileTime == M || DxDerived::SizeAtCompileTime == Eigen::Dynamic,
                   "eval(): dx size must match TE::vars_v" );
    typename TE::Input p{};
    for ( int i = 0; i < M; ++i ) p[std::size_t( i )] = T( dx( i ) );
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime >
        out( F.rows(), F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i ) out( i ) = F.derived().coeff( i ).eval( p );
    return out;
}

/// Evaluate a scalar univariate `TaylorExpansion` at a scalar displacement.
template < typename T, int N, typename S >
[[nodiscard]] T eval( const TaylorExpansion< T, IsotropicScheme< N, 1 >, S >& f, T dx ) noexcept
{
    typename TaylorExpansion< T, IsotropicScheme< N, 1 >, S >::Input p{ dx };
    return f.eval( p );
}

/// Evaluate each element of an Eigen matrix/vector of univariate `TaylorExpansion` objects at a shared scalar displacement.
template < typename Derived >
    requires( detail::is_te_v< typename Derived::Scalar >
              && detail::te_traits< typename Derived::Scalar >::vars_v == 1 )
[[nodiscard]] auto eval(
    const Eigen::MatrixBase< Derived >&                                 F,
    typename detail::te_traits< typename Derived::Scalar >::scalar_type dx )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::te_traits< TE >::scalar_type;
    typename TE::Input p{ dx };
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime >
        out( F.rows(), F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i )
        out( i ) = F.derived().coeff( i ).eval( p );
    return out;
}

}  // namespace tax::la

// -----------------------------------------------------------------------------
// value — surfaced under tax:: : the scalar accessors below plus the matrix
// overload (via the using-declaration), so callers write tax::value uniformly.
// tax::la::value remains valid for the matrix form.
// -----------------------------------------------------------------------------

namespace tax
{

/// Constant term (value at the expansion point) of a scalar `TaylorExpansion`.
template < typename TE >
    requires( la::detail::is_te_v< TE > )
[[nodiscard]] auto value( const TE& f ) noexcept
{
    return f.value();
}

/// Value of a plain arithmetic scalar — identity. Lets generic code call
/// `value(x)` uniformly whether `x` is a `double` or a `TaylorExpansion`.
template < typename T >
    requires std::is_arithmetic_v< T >
[[nodiscard]] constexpr T value( T x ) noexcept
{
    return x;
}

}  // namespace tax
