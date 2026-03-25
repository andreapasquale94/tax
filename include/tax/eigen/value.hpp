#pragma once

#include <tax/eigen/variables.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tax
{

/// @brief Value at expansion point for scalar TTE.
template < typename T, int N, int M >
[[nodiscard]] constexpr T value( const TruncatedExpansionT< T, N, M >& f ) noexcept
{
    return f.value();
}

// =============================================================================
// DenseBase (Matrix / Vector) overloads
// =============================================================================

/**
 * @brief Extract the scalar value (constant term) from each TTE matrix/vector element.
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < typename Derived >
[[nodiscard]] auto value( const Eigen::DenseBase< Derived >& t )
    requires( detail::is_tte_v< typename Derived::Scalar > )
{
    using TTE = typename Derived::Scalar;
    using T = typename detail::expansion_traits< TTE >::scalar_type;
    using Out = detail::rebind_matrix_t< Derived, T >;
    Out out( t.rows(), t.cols() );
    for ( Eigen::Index i = 0; i < t.size(); ++i )
        out.coeffRef( i ) = t.derived().coeff( i ).value();
    return out;
}

// =============================================================================
// Eigen::Tensor overloads (rank >= 1)
// =============================================================================

/**
 * @brief Extract the scalar value (constant term) from each TTE element.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int M, int Rank >
[[nodiscard]] auto value( const Eigen::Tensor< TruncatedExpansionT< T, N, M >, Rank >& t )
    requires( Rank >= 1 )
{
    Eigen::Tensor< T, Rank > out( t.dimensions() );
    for ( Eigen::Index i = 0; i < t.size(); ++i ) out.data()[i] = t.data()[i].value();
    return out;
}

}  // namespace tax
