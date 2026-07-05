#pragma once
#include <gtest/gtest.h>

#include <array>
#include <tax/tax.hpp>

namespace tax::test
{

/**
 * @brief Assert the fundamental Taylor-model containment property on a grid.
 *
 * Samples a uniform grid of the displacement domain and checks that the true
 * value `f_true(x)` (called with the absolute coordinates x = x0 + dx) lies
 * inside `tm.eval(dx)` up to `slack`, which only covers the libm rounding of
 * the reference values themselves.
 */
template < typename TMT, typename F >
inline void ExpectEncloses( const TMT& tm, F&& f_true, int n = 11, double slack = 1e-12 )
{
    constexpr int m = TMT::vars_v;
    const auto disp = tm.displacementDomain();
    const auto& x0 = tm.expansionPoint();

    std::array< int, std::size_t( m ) > idx{};
    while ( true )
    {
        typename TMT::Input dx{};
        typename TMT::Input x{};
        for ( std::size_t i = 0; i < std::size_t( m ); ++i )
        {
            const double lo = disp[i].lower();
            const double hi = disp[i].upper();
            double v = lo + ( hi - lo ) * double( idx[i] ) / double( n - 1 );
            v = std::min( std::max( v, lo ), hi );
            dx[i] = v;
            x[i] = x0[i] + v;
        }
        const double fv = f_true( x );
        const auto enc = tm.eval( dx );
        EXPECT_GE( fv, enc.lower() - slack ) << "containment violated (below) at dx[0]=" << dx[0];
        EXPECT_LE( fv, enc.upper() + slack ) << "containment violated (above) at dx[0]=" << dx[0];

        // Odometer increment over the grid.
        std::size_t k = 0;
        while ( k < std::size_t( m ) && ++idx[k] == n )
        {
            idx[k] = 0;
            ++k;
        }
        if ( k == std::size_t( m ) ) break;
    }
}

}  // namespace tax::test
