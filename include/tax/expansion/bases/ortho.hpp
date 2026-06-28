#pragma once

#include <array>
#include <cstddef>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/scheme/concept.hpp>

namespace tax::detail
{

// ===========================================================================
// Generic orthogonal-polynomial engine.
//
// Every classical orthogonal family is fixed by its three-term recurrence
//   x · P_n = α_n P_{n+1} + β_n P_n + γ_n P_{n-1},
// supplied by a `Rec` policy as `Rec::xmul<T>(n, α, β, γ)`. From it alone we can
// build, for any index scheme (uni- or multivariate):
//   * multiplication by a coordinate (the Jacobi operator),
//   * the truncated product (multiply-by-P_n ladder, tensored over axes),
//   * evaluation (per-axis recurrence table).
// Derivative / integral are family-specific (closed forms) and supplied by each
// basis directly, walking fibres with `forEachFiber`.
// ===========================================================================

/// Walk every fibre along `axis` (lines of constant other-axis indices), hand
/// the contiguous 1-D coefficient run to `op(a, L, b)`, and scatter back.
template < typename T, typename Scheme, typename Op >
constexpr void forEachFiber( const std::array< T, Scheme::nCoeff >& c, int axis, Op&& op,
                             std::array< T, Scheme::nCoeff >& out ) noexcept
{
    constexpr int M = Scheme::vars;
    constexpr std::size_t MAXL = std::size_t( Scheme::order ) + 1;
    for ( std::size_t k = 0; k < Scheme::nCoeff; ++k )
    {
        MultiIndex< M > base = Scheme::multiOf( k );
        if ( base[std::size_t( axis )] != 0 ) continue;  // only fibre origins
        std::array< T, MAXL > a{};
        std::array< std::size_t, MAXL > slot{};
        int L = 0;
        for ( int m = 0; m < int( MAXL ); ++m )
        {
            MultiIndex< M > idx = base;
            idx[std::size_t( axis )] = m;
            const std::size_t kk = Scheme::flatOf( idx );
            if ( kk == Scheme::kNotInBox ) break;
            a[std::size_t( m )] = c[kk];
            slot[std::size_t( m )] = kk;
            ++L;
        }
        std::array< T, MAXL > b{};
        op( a, L, b );
        for ( int m = 0; m < L; ++m ) out[slot[std::size_t( m )]] = b[std::size_t( m )];
    }
}

/// Multiply `u` by the coordinate `x_axis` in coefficient space (Jacobi op).
template < typename Rec, typename T, typename Scheme >
constexpr void orthoMulByX( std::array< T, Scheme::nCoeff >& out,
                            const std::array< T, Scheme::nCoeff >& u, int axis ) noexcept
{
    constexpr int M = Scheme::vars;
    out = {};
    for ( std::size_t k = 0; k < Scheme::nCoeff; ++k )
    {
        if ( u[k] == T{ 0 } ) continue;
        const MultiIndex< M > alpha = Scheme::multiOf( k );
        const int m = alpha[std::size_t( axis )];
        T a{}, b{}, g{};
        Rec::template xmul< T >( m, a, b, g );
        if ( b != T{ 0 } ) out[k] += b * u[k];  // β_m P_m
        {                                       // α_m P_{m+1}
            MultiIndex< M > idx = alpha;
            idx[std::size_t( axis )] = m + 1;
            const std::size_t kk = Scheme::flatOf( idx );
            if ( kk != Scheme::kNotInBox ) out[kk] += a * u[k];
        }
        if ( m >= 1 )  // γ_m P_{m-1}
        {
            MultiIndex< M > idx = alpha;
            idx[std::size_t( axis )] = m - 1;
            out[Scheme::flatOf( idx )] += g * u[k];
        }
    }
}

/// Multiply `u` by the basis element P_n(x_axis) via the recurrence ladder.
template < typename Rec, typename T, typename Scheme >
constexpr void orthoMulByElem( std::array< T, Scheme::nCoeff >& out,
                               const std::array< T, Scheme::nCoeff >& u, int axis, int n ) noexcept
{
    if ( n == 0 )
    {
        out = u;
        return;
    }
    std::array< T, Scheme::nCoeff > prev = u;  // u · P_0
    std::array< T, Scheme::nCoeff > cur{};     // u · P_1 = x_axis · u
    orthoMulByX< Rec, T, Scheme >( cur, u, axis );
    for ( int j = 1; j < n; ++j )
    {
        T a{}, b{}, g{};
        Rec::template xmul< T >( j, a, b, g );
        std::array< T, Scheme::nCoeff > Xc{};
        orthoMulByX< Rec, T, Scheme >( Xc, cur, axis );
        std::array< T, Scheme::nCoeff > next{};
        // u·P_{j+1} = (X(u·P_j) − β_j u·P_j − γ_j u·P_{j-1}) / α_j
        for ( std::size_t i = 0; i < Scheme::nCoeff; ++i )
            next[i] = ( Xc[i] - b * cur[i] - g * prev[i] ) / a;
        prev = cur;
        cur = next;
    }
    out = cur;
}

/// Truncated product in the orthogonal basis (uni- or multivariate).
template < typename Rec, typename T, typename Scheme >
constexpr void orthoProduct( std::array< T, Scheme::nCoeff >& out,
                             const std::array< T, Scheme::nCoeff >& a,
                             const std::array< T, Scheme::nCoeff >& b ) noexcept
{
    constexpr int M = Scheme::vars;
    out = {};
    for ( std::size_t j = 0; j < Scheme::nCoeff; ++j )
    {
        if ( b[j] == T{ 0 } ) continue;
        const MultiIndex< M > beta = Scheme::multiOf( j );
        std::array< T, Scheme::nCoeff > term = a;  // a · P_β, axis by axis
        for ( int d = 0; d < M; ++d )
        {
            std::array< T, Scheme::nCoeff > t2{};
            orthoMulByElem< Rec, T, Scheme >( t2, term, d, beta[std::size_t( d )] );
            term = t2;
        }
        for ( std::size_t i = 0; i < Scheme::nCoeff; ++i ) out[i] += b[j] * term[i];
    }
}

/// Evaluate Σ_α c_α Π_i P_{α_i}(x_i) via the per-axis recurrence table.
template < typename Rec, typename T, typename Scheme >
[[nodiscard]] constexpr T orthoEval(
    const std::array< T, Scheme::nCoeff >& c,
    const std::array< T, std::size_t( Scheme::vars ) >& x ) noexcept
{
    constexpr int N = Scheme::order;
    constexpr int M = Scheme::vars;
    std::array< std::array< T, std::size_t( N ) + 1 >, std::size_t( M ) > P{};
    for ( int i = 0; i < M; ++i )
    {
        P[std::size_t( i )][0] = T{ 1 };
        if constexpr ( N >= 1 )
        {
            T a{}, b{}, g{};
            Rec::template xmul< T >( 0, a, b, g );
            P[std::size_t( i )][1] = ( x[std::size_t( i )] - b ) / a;  // P_1 = (x − β_0)/α_0
        }
        for ( int m = 1; m < N; ++m )
        {
            T a{}, b{}, g{};
            Rec::template xmul< T >( m, a, b, g );
            P[std::size_t( i )][std::size_t( m + 1 )] =
                ( ( x[std::size_t( i )] - b ) * P[std::size_t( i )][std::size_t( m )] -
                  g * P[std::size_t( i )][std::size_t( m - 1 )] ) /
                a;
        }
    }
    T r{};
    for ( std::size_t k = 0; k < Scheme::nCoeff; ++k )
    {
        if ( c[k] == T{ 0 } ) continue;
        const MultiIndex< M > alpha = Scheme::multiOf( k );
        T term = c[k];
        for ( int i = 0; i < M; ++i )
            term *= P[std::size_t( i )][std::size_t( alpha[std::size_t( i )] )];
        r += term;
    }
    return r;
}

}  // namespace tax::detail

namespace tax
{

// ===========================================================================
// OrthogonalBasis< Derived > — CRTP scaffolding for the classical families.
//
// Every classical orthogonal family is fixed by its three-term recurrence
// (supplied as Derived::xmul) plus its closed-form derivative / integral. The
// truncated product, evaluation, and the is_tax_basis opt-in are then identical
// across families and come from the generic engine above, so they live here once
// and a concrete family supplies only `name`, `term`, `xmul`, `derivative`, and
// `integral`.
// ===========================================================================

template < typename Derived >
struct OrthogonalBasis
{
    static constexpr bool is_tax_basis = true;

    /// Truncated product in the family basis (Jacobi-operator ladder).
    template < typename T, typename Scheme >
    static constexpr void product( std::array< T, Scheme::nCoeff >& out,
                                   const std::array< T, Scheme::nCoeff >& a,
                                   const std::array< T, Scheme::nCoeff >& b ) noexcept
    {
        detail::orthoProduct< Derived, T, Scheme >( out, a, b );
    }

    /// Evaluate Σ_k c_k P_k(x) via the per-axis three-term recurrence.
    template < typename T, typename Scheme >
    [[nodiscard]] static constexpr T eval(
        const std::array< T, Scheme::nCoeff >& c,
        const std::array< T, std::size_t( Scheme::vars ) >& x ) noexcept
    {
        return detail::orthoEval< Derived, T, Scheme >( c, x );
    }
};

}  // namespace tax
