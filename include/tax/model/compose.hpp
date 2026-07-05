#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tax/model/arithmetic.hpp>
#include <tax/model/taylor_model.hpp>

namespace tax::model
{

// ===========================================================================
// Composition of Taylor models
//
// Given an "outer" model G over M variables and an "inner" vector
// H = (H_0, ..., H_{M-1}) of models over M2 variables, compose them into a
// model of G(H(u)) over the M2 variables u:
//
//   G(y) in P_G(y - x0_G) + I_G   for all y in domain(G),
//   H_j(u) in P_{H_j}(u - x0_H) + I_{H_j}   for all u in domain(H),
//
// so provided every inner range B(H_j) lies inside G's domain box, the
// composition
//
//   G(H(u)) in P_G(H(u) - x0_G) + I_G
//
// is enclosed by evaluating the outer polynomial at the shifted inner models
// (in Taylor-model arithmetic over u) and adding I_G. All the inner models,
// and every intermediate, share H's expansion point and domain, so the TM
// multiplications are always compatible.
// ===========================================================================

/// Compose the outer model `g` with the inner model vector `h`. Requires all
/// `h[j]` to share an expansion point / domain and every `h[j].bound()` to
/// lie inside `g.domain()[j]`; throws std::domain_error / std::invalid_argument
/// otherwise.
template < std::floating_point T, int N, int M, int M2 >
[[nodiscard]] TaylorModel< T, N, M2 > compose(
    const TaylorModel< T, N, M >& g,
    const std::array< TaylorModel< T, N, M2 >, std::size_t( M ) >& h )
{
    using Inner = TaylorModel< T, N, M2 >;
    static_assert( M >= 1 && M2 >= 1 );

    // All inner models must live over the same expansion parameter.
    for ( int j = 1; j < M; ++j )
    {
        if ( !h[std::size_t( j )].compatibleWith( h[0] ) )
            throw std::invalid_argument(
                "tax::model::compose: inner models differ in expansion point or domain" );
    }
    // Every inner range must stay inside the outer domain, else G's model is
    // not valid there.
    for ( int j = 0; j < M; ++j )
    {
        if ( !g.domain()[std::size_t( j )].contains( h[std::size_t( j )].bound() ) )
            throw std::domain_error(
                "tax::model::compose: inner range leaves the outer model's domain" );
    }

    const auto& x0_inner = h[0].expansionPoint();
    const auto& dom_inner = h[0].domain();

    // Shifted arguments A_j = H_j - x0_G[j] (still models over u).
    std::array< Inner, std::size_t( M ) > arg{};
    for ( int j = 0; j < M; ++j )
        arg[std::size_t( j )] = h[std::size_t( j )] - g.expansionPoint()[std::size_t( j )];

    // Power table pw[j][e] = A_j^e (e = 0..N), built by repeated multiplication.
    std::array< std::array< Inner, std::size_t( N ) + 1 >, std::size_t( M ) > pw{};
    for ( int j = 0; j < M; ++j )
    {
        pw[std::size_t( j )][0] = Inner::constant( T{ 1 }, x0_inner, dom_inner );
        for ( int e = 1; e <= N; ++e )
            pw[std::size_t( j )][std::size_t( e )] =
                pw[std::size_t( j )][std::size_t( e - 1 )] * arg[std::size_t( j )];
    }

    // Accumulate the outer polynomial evaluated at the inner models.
    Inner acc = Inner::constant( T{ 0 }, x0_inner, dom_inner );
    using OuterScheme = IsotropicScheme< N, M >;
    for ( std::size_t k = 0; k < TaylorModel< T, N, M >::nCoefficients; ++k )
    {
        const T c = g.polynomial()[k];
        if ( c == T{ 0 } ) continue;
        const auto alpha = OuterScheme::multiOf( k );
        Inner term = Inner::constant( c, x0_inner, dom_inner );
        for ( int j = 0; j < M; ++j )
        {
            const int e = alpha[std::size_t( j )];
            if ( e != 0 ) term = term * pw[std::size_t( j )][std::size_t( e )];
        }
        acc = acc + term;
    }

    // Add the outer remainder: G(H(u)) in P_G(H(u)-x0_G) + I_G.
    acc.remainder() += g.remainder();
    return acc;
}

/// Compose a vector of outer models with a shared inner vector, component-wise.
template < std::floating_point T, int N, int M, int M2, std::size_t D >
[[nodiscard]] std::array< TaylorModel< T, N, M2 >, D > compose(
    const std::array< TaylorModel< T, N, M >, D >& g,
    const std::array< TaylorModel< T, N, M2 >, std::size_t( M ) >& h )
{
    std::array< TaylorModel< T, N, M2 >, D > out{};
    for ( std::size_t i = 0; i < D; ++i ) out[i] = compose( g[i], h );
    return out;
}

}  // namespace tax::model
