#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <tax/core/multi_index.hpp>
#include <tax/core/scheme/concept.hpp>
#include <tax/series/basis.hpp>
#include <tax/series/ortho.hpp>

namespace tax
{

// ===========================================================================
// ChebyshevBasisOn< Lo, Hi > — Chebyshev polynomials of the first kind T_k,
// mapped onto the interval [Lo, Hi] (tensored over variables).
// ===========================================================================
//
// Stores  f(x) = Σ_α c_α Π_i T_{α_i}( u(x_i) )  in the plain (un-normalised)
// convention, where  u(x) = (2x − (Hi+Lo)) / (Hi−Lo)  maps [Lo,Hi] → [−1,1].
// The domain lives in the *type* (floating-point NTTP), so two models on
// different intervals are different types and cannot be silently mixed; the
// canonical default `ChebyshevBasis` is [−1, 1].
//
// 1-D identities (in the canonical variable u):
//   T_0 = 1, T_1 = u, T_{k+1} = 2u T_k − T_{k−1}
//   T_i T_j = (T_{i+j} + T_{|i−j|}) / 2
// The multivariate product is the tensor of the 1-D product:
//   T_α T_β = 2^{−M} Σ_{s∈{+,−}^M} T_{γ(s)},  γ_i(+) = α_i+β_i,  γ_i(−) = |α_i−β_i|.
// Derivative / integral act one axis at a time by the 1-D recurrences, with the
// affine chain-rule factor du/dx = 2/(Hi−Lo).
// ===========================================================================

template < double Lo, double Hi >
struct ChebyshevBasisOn
{
    static constexpr bool is_tax_basis = true;
    static constexpr double domainLo = Lo;
    static constexpr double domainHi = Hi;

    [[nodiscard]] static constexpr std::string_view name() noexcept { return "chebyshev"; }

    [[nodiscard]] static std::string term( int k )
    {
        if ( k == 0 ) return "1";
        return "T_" + std::to_string( k );
    }

    /// Map a physical coordinate to the canonical variable u ∈ [−1, 1].
    template < typename T >
    [[nodiscard]] static constexpr T toCanonical( T x ) noexcept
    {
        return ( T( 2 ) * x - T( Hi + Lo ) ) / T( Hi - Lo );
    }

    /// du/dx — the chain-rule factor applied per differentiation.
    template < typename T >
    [[nodiscard]] static constexpr T canonicalSlope() noexcept
    {
        return T( 2 ) / T( Hi - Lo );
    }

    // ------------------------------------------------------------------
    // Tensor Chebyshev product
    // ------------------------------------------------------------------
    template < typename T, typename Scheme >
    static constexpr void product( std::array< T, Scheme::nCoeff >& out,
                                   const std::array< T, Scheme::nCoeff >& a,
                                   const std::array< T, Scheme::nCoeff >& b ) noexcept
    {
        constexpr int M = Scheme::vars;
        out = {};
        T scale = T{ 1 };
        for ( int d = 0; d < M; ++d ) scale *= T( 0.5 );

        for ( std::size_t i = 0; i < Scheme::nCoeff; ++i )
        {
            if ( a[i] == T{ 0 } ) continue;
            const MultiIndex< M > alpha = Scheme::multiOf( i );
            for ( std::size_t j = 0; j < Scheme::nCoeff; ++j )
            {
                if ( b[j] == T{ 0 } ) continue;
                const MultiIndex< M > beta = Scheme::multiOf( j );
                const T base = scale * a[i] * b[j];
                // Enumerate the 2^M sign combinations of the per-axis fold.
                const unsigned combos = 1u << unsigned( M );
                for ( unsigned mask = 0; mask < combos; ++mask )
                {
                    MultiIndex< M > gamma{};
                    for ( int d = 0; d < M; ++d )
                    {
                        const int ad = alpha[std::size_t( d )];
                        const int bd = beta[std::size_t( d )];
                        gamma[std::size_t( d )] = ( mask >> unsigned( d ) ) & 1u
                                                      ? ad + bd
                                                      : ( ad < bd ? bd - ad : ad - bd );
                    }
                    const std::size_t kk = Scheme::flatOf( gamma );
                    if ( kk != Scheme::kNotInBox ) out[kk] += base;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------
    template < typename T, typename Scheme >
    [[nodiscard]] static constexpr T eval(
        const std::array< T, Scheme::nCoeff >& c,
        const std::array< T, std::size_t( Scheme::vars ) >& x ) noexcept
    {
        constexpr int N = Scheme::order;
        constexpr int M = Scheme::vars;
        // Per-axis Chebyshev value table Tt[i][m] = T_m( u(x_i) ).
        std::array< std::array< T, std::size_t( N ) + 1 >, std::size_t( M ) > Tt{};
        for ( int i = 0; i < M; ++i )
        {
            const T u = toCanonical( x[std::size_t( i )] );
            Tt[std::size_t( i )][0] = T{ 1 };
            if constexpr ( N >= 1 ) Tt[std::size_t( i )][1] = u;
            for ( int m = 2; m <= N; ++m )
                Tt[std::size_t( i )][std::size_t( m )] =
                    T( 2 ) * u * Tt[std::size_t( i )][std::size_t( m - 1 )] -
                    Tt[std::size_t( i )][std::size_t( m - 2 )];
        }
        T r{};
        for ( std::size_t k = 0; k < Scheme::nCoeff; ++k )
        {
            if ( c[k] == T{ 0 } ) continue;
            const MultiIndex< M > alpha = Scheme::multiOf( k );
            T term = c[k];
            for ( int i = 0; i < M; ++i )
                term *= Tt[std::size_t( i )][std::size_t( alpha[std::size_t( i )] )];
            r += term;
        }
        return r;
    }

    // ------------------------------------------------------------------
    // Per-axis derivative / integral (fiber-wise 1-D recurrences)
    // ------------------------------------------------------------------
    template < typename T, typename Scheme >
    static constexpr void derivative( std::array< T, Scheme::nCoeff >& out,
                                      const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept
    {
        out = {};
        const T slope = canonicalSlope< T >();
        detail::forEachFiber< T, Scheme >(
            c, axis,
            [&]( const std::array< T, std::size_t( Scheme::order ) + 1 >& a, int L,
                 std::array< T, std::size_t( Scheme::order ) + 1 >& b ) {
                // chder (plain convention): degree-(L-1) input -> derivative.
                b = {};
                if ( L >= 2 )
                {
                    for ( int m = L - 2; m >= 0; --m )
                    {
                        T v = T( 2 * ( m + 1 ) ) * a[std::size_t( m + 1 )];
                        if ( m + 2 <= L - 1 ) v += b[std::size_t( m + 2 )];
                        b[std::size_t( m )] = v;
                    }
                    b[0] *= T( 0.5 );
                }
                for ( int m = 0; m < L; ++m ) b[std::size_t( m )] *= slope;
            },
            out );
    }

    template < typename T, typename Scheme >
    static constexpr void integral( std::array< T, Scheme::nCoeff >& out,
                                    const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept
    {
        out = {};
        const T inv_slope = T{ 1 } / canonicalSlope< T >();  // dx/du = (Hi-Lo)/2
        detail::forEachFiber< T, Scheme >(
            c, axis,
            [&]( const std::array< T, std::size_t( Scheme::order ) + 1 >& a, int L,
                 std::array< T, std::size_t( Scheme::order ) + 1 >& b ) {
                // chint (plain convention), constant of integration 0.
                b = {};
                if ( L >= 2 )
                {
                    b[1] = a[0];
                    if ( L >= 3 ) b[1] -= T( 0.5 ) * a[2];
                    for ( int m = 2; m <= L - 1; ++m )
                    {
                        T v = a[std::size_t( m - 1 )];
                        if ( m + 1 <= L - 1 ) v -= a[std::size_t( m + 1 )];
                        b[std::size_t( m )] = v / T( 2 * m );
                    }
                }
                for ( int m = 0; m < L; ++m ) b[std::size_t( m )] *= inv_slope;
            },
            out );
    }
};

/// Canonical Chebyshev basis on [−1, 1].
using ChebyshevBasis = ChebyshevBasisOn< -1.0, 1.0 >;

static_assert( BasisPolicy< ChebyshevBasis > );

}  // namespace tax
