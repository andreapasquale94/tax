#pragma once

#include <concepts>
#include <cstddef>

namespace tax
{

/// Scalar constraint for coefficients and function values.
template < typename T >
concept Scalar = std::floating_point< T >;

/// Any truncated Taylor polynomial type.
template < typename P >
concept TaylorPolynomial = requires( const P& p, std::size_t k )
{
    typename P::scalar_type;
    typename P::container_t;
    { P::order_v } -> std::convertible_to< int >;
    { P::vars_v } -> std::convertible_to< int >;
    { P::nCoefficients } -> std::convertible_to< std::size_t >;
    { p.value() } -> std::convertible_to< typename P::scalar_type >;
};

/// Refinement of `TaylorPolynomial` exposing flat-index coefficient access.
template < typename P >
concept DensePolynomial = TaylorPolynomial< P > && requires( const P& p, std::size_t k )
{
    { p[k] } -> std::convertible_to< typename P::scalar_type >;
};

}  // namespace tax
