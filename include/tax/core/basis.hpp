#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <string_view>
#include <tax/core/scheme/isotropic.hpp>

namespace tax
{

// ===========================================================================
// BasisPolicy concept
// ===========================================================================
//
// A *basis policy* describes one family of univariate polynomials {P_0, P_1,
// ...} in which a truncated expansion  f = sum_{k=0}^{N} c_k P_k(x)  is stored.
// Everything that distinguishes one polynomial family from another lives in
// the policy; the carrier type `Series< Basis, N, M, T >` only knows how to store
// the coefficient array and how to do basis-independent linear-space work
// (addition, scalar multiply, the `variable`/`constant` factories).
//
// A conforming policy exposes the following *static* surface (all templated on
// the coefficient type `T` and the truncation order `N`, so a single policy
// type serves every order):
//
//   static constexpr std::string_view name();           // e.g. "taylor"
//   static std::string                term( int k );    // pretty basis label, e.g. "x^2" / "T_2"
//
//   template< typename T, int N >
//   static constexpr void product( std::array< T, N + 1 >& out,
//                                  const std::array< T, N + 1 >& a,
//                                  const std::array< T, N + 1 >& b ) noexcept;
//
//   template< typename T, int N >
//   static constexpr T eval( const std::array< T, N + 1 >& c, T x ) noexcept;
//
//   template< typename T, typename Scheme >
//   static constexpr void derivative( std::array< T, Scheme::nCoeff >& out,
//                                     const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept;
//
//   template< typename T, typename Scheme >
//   static constexpr void integral( std::array< T, Scheme::nCoeff >& out,
//                                   const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept;
//
// Both `P_0 == 1` and `P_1 == x` are *required* of every basis (true for the
// monomial and every classical orthogonal family normalised the usual way).
// This is what makes the `constant(v)`  (c_0 = v)  and `variable()`  (c_1 = 1)
// factories basis-independent: a constant is `v * P_0` and the identity map is
// `1 * P_1` in any conforming basis.
//
// Transcendental functions (exp, sin, ...) and series division are *optional*
// per basis and are supplied by free functions constrained on the policy.
// ===========================================================================

/// Conforming basis policy: opts in with `is_tax_basis`, names itself, and
/// supplies the four scheme-templated operations the carrier delegates to
/// (checked here against a representative scheme so a malformed policy fails at
/// the concept boundary, not deep inside instantiation).
template < typename B >
concept BasisPolicy =
    requires {
        { B::is_tax_basis } -> std::convertible_to< bool >;
        { B::name() } -> std::convertible_to< std::string_view >;
    } && B::is_tax_basis &&
    requires( std::array< double, IsotropicScheme< 2, 1 >::nCoeff > c, std::array< double, 1 > x ) {
        B::template product< double, IsotropicScheme< 2, 1 > >( c, c, c );
        {
            B::template eval< double, IsotropicScheme< 2, 1 > >( c, x )
        } -> std::convertible_to< double >;
        B::template derivative< double, IsotropicScheme< 2, 1 > >( c, c, 0 );
        B::template integral< double, IsotropicScheme< 2, 1 > >( c, c, 0 );
    };

}  // namespace tax
