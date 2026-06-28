#pragma once

#include <cmath>
#include <span>
#include <tax/expansion/scheme/isotropic.hpp>
#include <tax/expansion/detail/algebra.hpp>

namespace tax::detail::kernels
{

/// Coupled trigonometric series: jointly compute `sin(a)` and `cos(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesSinCos( std::array< T, Scheme::nCoeff >& s, std::array< T, Scheme::nCoeff >& c,
                   const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::cos;
    using std::sin;
    s = {};
    c = {};
    s[0] = sin( a[0] );
    c[0] = cos( a[0] );

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T sr = T{ 0 }, cr = T{ 0 };
            for ( int k = 0; k < d; ++k )
            {
                const T w = T( d - k ) * a[std::size_t( d - k )];
                sr += w * c[std::size_t( k )];
                cr += w * s[std::size_t( k )];
            }
            const T inv_d = T{ 1 } / T( d );
            s[std::size_t( d )] = sr * inv_d;
            c[std::size_t( d )] = -cr * inv_d;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T sin_rhs = T{ 0 };
                T cos_rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row )
                {
                    const T fg = T( e.db ) * a[e.b_idx];
                    sin_rhs += fg * c[e.g_idx];
                    cos_rhs += fg * s[e.g_idx];
                }
                const T inv_d = T{ 1 } / T( d );
                s[ai] = sin_rhs * inv_d;
                c[ai] = -cos_rhs * inv_d;
            } );
    }
}

/// Sine series `out = sin(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesSin( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    std::array< T, Scheme::nCoeff > c{};
    seriesSinCos< T, Scheme >( out, c, a );
}

/// Cosine series `out = cos(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesCos( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    std::array< T, Scheme::nCoeff > s{};
    seriesSinCos< T, Scheme >( s, out, a );
}

/// Tangent series `out = tan(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesTan( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;
    std::array< T, S > s{}, c{};
    seriesSinCos< T, Scheme >( s, c, a );

    out = {};
    out[0] = s[0] / c[0];
    const T inv_c0 = T{ 1 } / c[0];

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = s[std::size_t( d )];
            for ( int k = 1; k <= d; ++k ) rhs -= c[std::size_t( k )] * out[std::size_t( d - k )];
            out[std::size_t( d )] = rhs * inv_c0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = s[ai];
                for ( const RecurrenceEntry& e : row ) rhs -= c[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv_c0;
            } );
    }
}

/// Inverse sine series `out = asin(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesAsin( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::asin;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = sqrt(1 - a^2)
    std::array< T, S > asq{}, omf{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    omf = {};
    omf[0] = T{ 1 };
    for ( std::size_t i = 0; i < S; ++i ) omf[i] -= asq[i];
    seriesSqrt< T, Scheme >( h, omf );

    out = {};
    out[0] = asin( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * h[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( a[std::size_t( d )] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * h[e.b_idx] * out[e.g_idx];
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
    }
}

/// Inverse cosine series `out = acos(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesAcos( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::acos;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = sqrt(1 - a^2)
    std::array< T, S > asq{}, omf{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    omf = {};
    omf[0] = T{ 1 };
    for ( std::size_t i = 0; i < S; ++i ) omf[i] -= asq[i];
    seriesSqrt< T, Scheme >( h, omf );

    out = {};
    out[0] = acos( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * h[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( -a[std::size_t( d )] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * h[e.b_idx] * out[e.g_idx];
                out[ai] = ( -a[ai] - rhs / T( d ) ) * inv_h0;
            } );
    }
}

/// Inverse tangent series `out = atan(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesAtan( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::atan;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = 1 + a^2
    std::array< T, S > asq{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    h = asq;
    h[0] += T{ 1 };

    out = {};
    out[0] = atan( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * h[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( a[std::size_t( d )] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * h[e.b_idx] * out[e.g_idx];
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
    }
}

/// Two-argument arctangent series `out = atan2(y, x)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesAtan2( std::array< T, Scheme::nCoeff >& out, const std::array< T, Scheme::nCoeff >& y,
                  const std::array< T, Scheme::nCoeff >& x ) noexcept
{
    using std::atan2;
    constexpr std::size_t S = Scheme::nCoeff;

    // Compute r = y / x in a single forward-substitution pass.
    std::array< T, S > r{};
    seriesDivide< T, Scheme >( r, y, x );

    // h = 1 + r^2
    std::array< T, S > rsq{}, h{};
    tax::cauchySelfProduct< T, Scheme >( rsq, r );
    h = rsq;
    h[0] += T{ 1 };

    out = {};
    out[0] = atan2( y[0], x[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * h[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( r[std::size_t( d )] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * h[e.b_idx] * out[e.g_idx];
                out[ai] = ( r[ai] - rhs / T( d ) ) * inv_h0;
            } );
    }
}

}  // namespace tax::detail::kernels
