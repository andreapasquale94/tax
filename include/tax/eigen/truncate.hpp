// Free-function order-reducing truncation: tax::truncate< N2 >( x ).
//
// Sugar over the member `x.truncate< N2 >()`, plus an Eigen overload that
// truncates each element of a vector/matrix of expansions (returning the
// lower-order element type). Mirrors the free tax::la::value / eval helpers.

#pragma once

#include <Eigen/Core>
#include <utility>

namespace tax
{

/// Order-reducing truncation of a scalar expansion (dense / named / mixed):
/// `tax::truncate< N2 >( f )` == `f.template truncate< N2 >()`.
template < int N2, typename E >
    requires requires( const E& e ) { e.template truncate< N2 >(); }
[[nodiscard]] constexpr auto truncate( const E& e ) noexcept
{
    return e.template truncate< N2 >();
}

/// Truncate every element of an Eigen vector/matrix of expansions to order `N2`,
/// yielding a same-shape matrix of the lower-order element type.
template < int N2, typename Derived >
[[nodiscard]] auto truncate( const Eigen::MatrixBase< Derived >& m )
{
    using In = typename Derived::Scalar;
    using Out = decltype( std::declval< const In& >().template truncate< N2 >() );
    Eigen::Matrix< Out, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime > out( m.rows(),
                                                                                      m.cols() );
    for ( Eigen::Index i = 0; i < m.size(); ++i )
        out( i ) = m.derived().coeff( i ).template truncate< N2 >();
    return out;
}

}  // namespace tax
