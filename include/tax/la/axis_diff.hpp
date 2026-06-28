#pragma once

// Shared coefficient-form differential kernels for the named layers.
//
// The per-axis gradient/Hessian/Jacobian of a Taylor named expansion (single
// order, tax::named::NamedTaylorExpansion) and of a mixed-order named expansion
// (tax::named::MixedTaylorExpansion) are the same computation: read Taylor
// coefficients off the inner dense expansion at the multi-indices of one axis
// block. The two public surfaces (la/named.hpp, la/mixed_named.hpp) differ only
// in the wrapper type they match and how they locate the axis block (Axis vs
// OrderedAxis); both compute `(dim, off)` their own way and delegate the actual
// coefficient walk to the helpers below, so the loop bodies live in one place.

#include <Eigen/Core>
#include <cstddef>
#include <tax/expansion/multi_index.hpp>

namespace tax::named::detail
{

/// Gradient ∂f/∂x_{off+i}, i ∈ [0, Dim), of an inner dense expansion `inner`
/// over `Vars` variables (first-order Taylor coefficients along the axis block).
template < typename T, int Dim, int Vars, typename Inner >
[[nodiscard]] Eigen::Matrix< T, Dim, 1 > axisGradient( const Inner& inner, int off ) noexcept
{
    Eigen::Matrix< T, Dim, 1 > g;
    MultiIndex< Vars > alpha{};
    for ( int i = 0; i < Dim; ++i )
    {
        alpha[std::size_t( off + i )] = 1;
        g( i ) = inner.derivative( alpha );
        alpha[std::size_t( off + i )] = 0;
    }
    return g;
}

/// Hessian ∂²f/∂x_{off+i}∂x_{off+j} over one axis block (second-order Taylor
/// coefficients).
template < typename T, int Dim, int Vars, typename Inner >
[[nodiscard]] Eigen::Matrix< T, Dim, Dim > axisHessian( const Inner& inner, int off ) noexcept
{
    Eigen::Matrix< T, Dim, Dim > H;
    for ( int i = 0; i < Dim; ++i )
        for ( int j = 0; j < Dim; ++j )
        {
            MultiIndex< Vars > alpha{};
            alpha[std::size_t( off + i )] += 1;
            alpha[std::size_t( off + j )] += 1;
            H( i, j ) = inner.derivative( alpha );
        }
    return H;
}

/// Jacobian of an Eigen vector `F` of named expansions w.r.t. one axis block:
/// row r, column j is ∂F_r/∂x_{off+j}. Each element exposes `.inner()`.
template < typename T, int Dim, int Vars, typename Derived >
[[nodiscard]] auto axisJacobian( const Eigen::MatrixBase< Derived >& F, int off )
{
    constexpr int K = Derived::SizeAtCompileTime;
    Eigen::Matrix< T, K, Dim > out( F.size(), Dim );
    for ( Eigen::Index r = 0; r < F.size(); ++r )
    {
        MultiIndex< Vars > alpha{};
        for ( int j = 0; j < Dim; ++j )
        {
            alpha[std::size_t( off + j )] = 1;
            out( r, j ) = F.derived().coeff( r ).inner().derivative( alpha );
            alpha[std::size_t( off + j )] = 0;
        }
    }
    return out;
}

}  // namespace tax::named::detail
