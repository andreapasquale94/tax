#pragma once

#include <array>
#include <cmath>
#include <span>
#include <tax/expansion/detail/cauchy.hpp>
#include <tax/expansion/detail/recurrence_stencil.hpp>
#include <tax/expansion/scheme/isotropic.hpp>

namespace tax::detail::kernels
{

/// Self-product `out = f * f` (M == 1 exploits pair symmetry; M >= 2 forwards to cauchyProduct).
template < typename T, int N, int M >
constexpr void cauchySelfProduct( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& f ) noexcept
{
    tax::cauchySelfProduct< T, tax::IsotropicScheme< N, M > >( out, f );
}

/// Square series `out = a^2` via the symmetric self-product (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesSquare( std::array< T, Scheme::nCoeff >& out,
                             const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    tax::cauchySelfProduct< T, Scheme >( out, a );
}

/// Cube series `out = a^3` via two Cauchy products (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesCube( std::array< T, Scheme::nCoeff >& out,
                           const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;
    std::array< T, S > tmp{};
    tax::cauchySelfProduct< T, Scheme >( tmp, a );
    tax::cauchyProduct< T, Scheme >( out, tmp, a );
}

/// Reciprocal series `a * out = 1` by forward substitution (scheme-generic). Requires `a[0] != 0`.
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesReciprocal( std::array< T, Scheme::nCoeff >& out,
                                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    out = {};
    const T inv_a0 = T{ 1 } / a[0];
    out[0] = inv_a0;

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k <= d; ++k ) rhs -= a[std::size_t( k )] * out[std::size_t( d - k )];
            out[std::size_t( d )] = rhs * inv_a0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row ) rhs -= a[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv_a0;
            } );
    }
}

/// Reciprocal series `a * out = 1` by forward substitution. Requires `a[0] != 0`.
template < typename T, int N, int M >
constexpr void seriesReciprocal( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    seriesReciprocal< T, tax::IsotropicScheme< N, M > >( out, a );
}

/// Quotient series `out = a / b` by forward substitution (scheme-generic). Requires `b[0] != 0`;
/// `out` must not alias `a` or `b`.
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesDivide( std::array< T, Scheme::nCoeff >& out,
                             const std::array< T, Scheme::nCoeff >& a,
                             const std::array< T, Scheme::nCoeff >& b ) noexcept
{
    out = {};
    const T inv_b0 = T{ 1 } / b[0];
    out[0] = a[0] * inv_b0;

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = a[std::size_t( d )];
            for ( int k = 1; k <= d; ++k ) rhs -= b[std::size_t( k )] * out[std::size_t( d - k )];
            out[std::size_t( d )] = rhs * inv_b0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = a[ai];
                for ( const RecurrenceEntry& e : row ) rhs -= b[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv_b0;
            } );
    }
}

/// Square-root series `out * out = a` (principal branch, scheme-generic). Requires `a[0] > 0`.
template < typename T, tax::IndexScheme Scheme >
void seriesSqrt( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::sqrt;
    out = {};
    out[0] = sqrt( a[0] );
    const T inv2g0 = T{ 1 } / ( T{ 2 } * out[0] );

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = a[std::size_t( d )];
            for ( int k = 1; k + k < d; ++k )
                rhs -= T{ 2 } * out[std::size_t( k )] * out[std::size_t( d - k )];
            if ( d % 2 == 0 ) rhs -= out[std::size_t( d / 2 )] * out[std::size_t( d / 2 )];
            out[std::size_t( d )] = rhs * inv2g0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = a[ai];
                // |beta| == d entries read out[ai], which is still zero here,
                // so the ordered walk needs no |beta| < d filter.
                for ( const RecurrenceEntry& e : row ) rhs -= out[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv2g0;
            } );
    }
}

/// Cubic-root series `out^3 = a` (real branch, scheme-generic). Requires `a[0] != 0`.
template < typename T, tax::IndexScheme Scheme >
void seriesCbrt( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::cbrt;
    constexpr std::size_t S = Scheme::nCoeff;

    out = {};
    out[0] = cbrt( a[0] );
    const T inv3g0sq = T{ 1 } / ( T{ 3 } * out[0] * out[0] );

    if constexpr ( Scheme::isUnivariate )
    {
        std::array< T, S > sq{};
        sq[0] = out[0] * out[0];
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T sq_d_partial = T{ 0 };
            for ( int k = 1; k + k < d; ++k )
                sq_d_partial += T{ 2 } * out[std::size_t( k )] * out[std::size_t( d - k )];
            if ( d % 2 == 0 ) sq_d_partial += out[std::size_t( d / 2 )] * out[std::size_t( d / 2 )];

            T rhs = out[0] * sq_d_partial;
            for ( int j = 1; j < d; ++j ) rhs += out[std::size_t( j )] * sq[std::size_t( d - j )];

            out[std::size_t( d )] = ( a[std::size_t( d )] - rhs ) * inv3g0sq;
            sq[std::size_t( d )] = T{ 2 } * out[0] * out[std::size_t( d )] + sq_d_partial;
        }
    } else
    {
        std::array< T, S > sq{};
        sq[0] = out[0] * out[0];
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = a[ai];
                // |beta| == d entries read out[ai], which is still zero here,
                // so the ordered walk needs no |beta| < d filter; sq is only
                // read at |gamma| < d, already final from earlier rows.
                for ( const RecurrenceEntry& e : row )
                    rhs -= out[e.b_idx] * ( out[0] * out[e.g_idx] + sq[e.g_idx] );
                out[ai] = rhs * inv3g0sq;

                // Maintain sq = out^2 at alpha: the beta = 0 term plus all
                // |beta| >= 1 decompositions (out[ai] is final now, so the
                // beta = alpha entry contributes out[ai]*out[0] correctly).
                T val = out[0] * out[ai];
                for ( const RecurrenceEntry& e : row ) val += out[e.b_idx] * out[e.g_idx];
                sq[ai] = val;
            } );
    }
}

/// Real-exponent power series `out = a^c` (scheme-generic). Requires `a[0] != 0`; not constexpr.
template < typename T, tax::IndexScheme Scheme >
void seriesPow( std::array< T, Scheme::nCoeff >& out, const std::array< T, Scheme::nCoeff >& a,
                T c ) noexcept
{
    using std::pow;
    out = {};
    out[0] = pow( a[0], c );
    const T inv_a0 = T{ 1 } / a[0];

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k )
                rhs +=
                    ( c * T( d - k ) - T( k ) ) * a[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = rhs * inv_a0 / T( d );
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row )
                    rhs += ( c * T( e.db ) - T( d - int( e.db ) ) ) * a[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv_a0 / T( d );
            } );
    }
}

/// Integer-exponent power series `out = a^n` via binary exponentiation (scheme-generic; negative n
/// via reciprocal).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesPowInt( std::array< T, Scheme::nCoeff >& out,
                             const std::array< T, Scheme::nCoeff >& a, int n ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;

    if ( n == 0 )
    {
        out = {};
        out[0] = T{ 1 };
        return;
    }
    if ( n == 1 )
    {
        out = a;
        return;
    }
    if ( n == -1 )
    {
        seriesReciprocal< T, Scheme >( out, a );
        return;
    }
    if ( n < 0 )
    {
        std::array< T, S > rec{};
        seriesReciprocal< T, Scheme >( rec, a );
        seriesPowInt< T, Scheme >( out, rec, n );
        return;
    }
    // n >= 2: binary exponentiation (square-and-multiply). Squarings go
    // through the symmetric self-product kernel; `out` is seeded with the
    // base power of the lowest set bit, skipping the wasted 1 * base
    // multiply of the textbook formulation.
    std::array< T, S > base = a;
    int e = n;
    while ( !( e & 1 ) )
    {
        std::array< T, S > tmp{};
        tax::cauchySelfProduct< T, Scheme >( tmp, base );
        base = tmp;
        e >>= 1;
    }
    out = base;
    e >>= 1;
    while ( e > 0 )
    {
        std::array< T, S > sq{};
        tax::cauchySelfProduct< T, Scheme >( sq, base );
        base = sq;
        if ( e & 1 )
        {
            std::array< T, S > tmp{};
            tax::cauchyProduct< T, Scheme >( tmp, out, base );
            out = tmp;
        }
        e >>= 1;
    }
}

}  // namespace tax::detail::kernels
