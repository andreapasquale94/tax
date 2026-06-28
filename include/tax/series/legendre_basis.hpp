#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <tax/series/basis.hpp>
#include <tax/series/ortho.hpp>

namespace tax
{

// ===========================================================================
// LegendreBasis — Legendre polynomials P_n on [-1, 1] (tensored over variables)
// ===========================================================================
//
// Recurrence:  P_0 = 1, P_1 = x,  (n+1) P_{n+1} = (2n+1) x P_n − n P_{n-1},
// i.e.  x P_n = (n+1)/(2n+1) P_{n+1} + n/(2n+1) P_{n-1}.
// Product and evaluation come from the generic orthogonal engine; the
// derivative / integral use Legendre's closed-form coefficient recurrences:
//   f' :  b_{n-1} = b_{n+1} + (2n-1) a_n
//   ∫f :  B_m = a_{m-1}/(2m-1) − a_{m+1}/(2m+3)   (constant of integration 0)
// ===========================================================================

struct LegendreBasis : OrthogonalBasis< LegendreBasis >
{
    [[nodiscard]] static constexpr std::string_view name() noexcept { return "legendre"; }

    [[nodiscard]] static std::string term( int k )
    {
        if ( k == 0 ) return "1";
        return "P_" + std::to_string( k );
    }

    /// x P_n = α_n P_{n+1} + β_n P_n + γ_n P_{n-1}.
    template < typename T >
    static constexpr void xmul( int n, T& alpha, T& beta, T& gamma ) noexcept
    {
        alpha = T( n + 1 ) / T( 2 * n + 1 );
        beta = T{ 0 };
        gamma = T( n ) / T( 2 * n + 1 );
    }

    template < typename T, typename Scheme >
    static constexpr void derivative( std::array< T, Scheme::nCoeff >& out,
                                      const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept
    {
        out = {};
        detail::forEachFiber< T, Scheme >(
            c, axis,
            []( const std::array< T, std::size_t( Scheme::order ) + 1 >& a, int L,
                std::array< T, std::size_t( Scheme::order ) + 1 >& b ) {
                // f' = Σ_m (2m+1) S_m P_m,  S_m = a_{m+1} + a_{m+3} + …
                // Accumulate S_m in b (descending), then scale by (2m+1).
                b = {};
                for ( int m = L - 1; m >= 0; --m )
                {
                    T s = ( m + 1 <= L - 1 ) ? a[std::size_t( m + 1 )] : T{ 0 };
                    if ( m + 2 <= L - 1 ) s += b[std::size_t( m + 2 )];
                    b[std::size_t( m )] = s;
                }
                for ( int m = 0; m < L; ++m ) b[std::size_t( m )] *= T( 2 * m + 1 );
            },
            out );
    }

    template < typename T, typename Scheme >
    static constexpr void integral( std::array< T, Scheme::nCoeff >& out,
                                    const std::array< T, Scheme::nCoeff >& c, int axis ) noexcept
    {
        out = {};
        detail::forEachFiber< T, Scheme >(
            c, axis,
            []( const std::array< T, std::size_t( Scheme::order ) + 1 >& a, int L,
                std::array< T, std::size_t( Scheme::order ) + 1 >& b ) {
                b = {};
                for ( int m = 0; m < L; ++m )
                {
                    T v{};
                    if ( m >= 1 ) v += a[std::size_t( m - 1 )] / T( 2 * m - 1 );
                    if ( m + 1 <= L - 1 ) v -= a[std::size_t( m + 1 )] / T( 2 * m + 3 );
                    b[std::size_t( m )] = v;
                }
            },
            out );
    }
};

static_assert( Basis< LegendreBasis > );

}  // namespace tax
