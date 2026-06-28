#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace tax
{

/// @brief Opt-in trait marking a type usable as a TaylorExpansion coefficient.
///
/// Every floating-point type qualifies by default. A non-floating coefficient
/// type (e.g. a SIMD batch wrapper with an element-wise scalar-like math API)
/// opts in by specialising this to `std::true_type`, keeping `Scalar` a strict
/// superset of `std::floating_point`.
template < typename T >
struct is_tax_scalar : std::bool_constant< std::floating_point< T > >
{
};

/// Scalar constraint for coefficients and function values.
template < typename T >
concept Scalar = is_tax_scalar< T >::value;

/// @brief Underlying real (floating-point) scalar of a coefficient type.
///
/// Identity for ordinary floating-point coefficients; a batch coefficient type
/// specialises this to its lane scalar so generic kernels can name real
/// constants (e.g. `std::numbers::inv_sqrtpi_v< real_scalar_t< T > >`).
template < typename T >
struct real_scalar
{
    using type = T;
};

template < typename T >
using real_scalar_t = typename real_scalar< T >::type;

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

/// Refinement of `TaylorPolynomial` for dense (flat-index) storage.
template < typename P >
concept DensePolynomial = TaylorPolynomial< P > && requires( const P& p, std::size_t k )
{
    { p[k] } -> std::convertible_to< typename P::scalar_type >;
};

/// Refinement of `TaylorPolynomial` for sparse storage.
template < typename P >
concept SparsePolynomial = TaylorPolynomial< P > && requires( const P& p )
{
    { p.nnz() } -> std::convertible_to< std::size_t >;
    { p.support() };
    { p.values() };
};

}  // namespace tax
