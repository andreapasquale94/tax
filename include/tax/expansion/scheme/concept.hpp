#pragma once

// IndexScheme: the monomial-set abstraction the dense kernels are generic over.
// A scheme bundles the storage size, a graded recurrence-row walker (ascending
// total degree, so forward substitution is causal), and the flat<->multi maps.

#include <array>
#include <cstddef>

namespace tax
{

/// Concept: a monomial-set index scheme usable by the dense kernels.
template < typename S >
concept IndexScheme = requires( const S& ) {
    { S::nCoeff } -> std::convertible_to< std::size_t >;
    { S::order } -> std::convertible_to< int >;
    { S::isUnivariate } -> std::convertible_to< bool >;
};

/// Scheme-generic Cauchy product: delegates to Scheme::cauchyProduct<T>.
template < typename T, IndexScheme Scheme >
constexpr void cauchyProduct( std::array< T, Scheme::nCoeff >& out,
                              const std::array< T, Scheme::nCoeff >& a,
                              const std::array< T, Scheme::nCoeff >& b ) noexcept
{
    Scheme::template cauchyProduct< T >( out, a, b );
}

/// Scheme-generic self-product: delegates to Scheme::cauchySelfProduct<T>.
template < typename T, IndexScheme Scheme >
constexpr void cauchySelfProduct( std::array< T, Scheme::nCoeff >& out,
                                  const std::array< T, Scheme::nCoeff >& f ) noexcept
{
    Scheme::template cauchySelfProduct< T >( out, f );
}

}  // namespace tax
