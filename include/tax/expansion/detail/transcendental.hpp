#pragma once

#include <cmath>
#include <numbers>
#include <span>
#include <tax/expansion/concepts.hpp>
#include <tax/expansion/scheme/isotropic.hpp>
#include <tax/expansion/detail/algebra.hpp>

namespace tax::detail::kernels
{

/// Natural exponential series `out = exp(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesExp( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::exp;
    out = {};
    out[0] = exp( a[0] );

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k )
                rhs += T( d - k ) * a[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = rhs / T( d );
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row ) rhs += T( e.db ) * a[e.b_idx] * out[e.g_idx];
                out[ai] = rhs / T( d );
            } );
    }
}

/// Natural logarithm series `out = log(a)` (scheme-generic). Requires `a[0] > 0`.
template < typename T, tax::IndexScheme Scheme >
void seriesLog( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::log;
    out = {};
    out[0] = log( a[0] );
    const T inv_a0 = T{ 1 } / a[0];

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * a[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( a[std::size_t( d )] - rhs / T( d ) ) * inv_a0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * a[e.b_idx] * out[e.g_idx];
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_a0;
            } );
    }
}

/// Hyperbolic sine series `out = sinh(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesSinh( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::sinh;
    constexpr std::size_t S = Scheme::nCoeff;

    // compute exp(a) and exp(-a)
    std::array< T, S > ep{}, em{}, neg_a{};
    neg_a = a;
    for ( std::size_t i = 0; i < S; ++i ) neg_a[i] = -neg_a[i];
    seriesExp< T, Scheme >( ep, a );
    seriesExp< T, Scheme >( em, neg_a );

    out = {};
    out[0] = sinh( a[0] );
    for ( std::size_t i = 1; i < S; ++i ) out[i] = ( ep[i] - em[i] ) * T{ 0.5 };
}

/// Hyperbolic cosine series `out = cosh(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesCosh( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::cosh;
    constexpr std::size_t S = Scheme::nCoeff;

    // compute exp(a) and exp(-a)
    std::array< T, S > ep{}, em{}, neg_a{};
    neg_a = a;
    for ( std::size_t i = 0; i < S; ++i ) neg_a[i] = -neg_a[i];
    seriesExp< T, Scheme >( ep, a );
    seriesExp< T, Scheme >( em, neg_a );

    out = {};
    out[0] = cosh( a[0] );
    for ( std::size_t i = 1; i < S; ++i ) out[i] = ( ep[i] + em[i] ) * T{ 0.5 };
}

/// Hyperbolic tangent series `out = tanh(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesTanh( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::cosh;
    using std::sinh;
    using std::tanh;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = cosh(a), s = sinh(a). Share a single exp(a)/exp(-a) pair rather than
    // calling seriesCosh + seriesSinh, each of which would recompute both exps
    // (4 seriesExp calls -> 2). The derived coefficients are bit-identical to the
    // dedicated kernels.
    std::array< T, S > ep{}, em{}, neg_a{};
    neg_a = a;
    for ( std::size_t i = 0; i < S; ++i ) neg_a[i] = -neg_a[i];
    seriesExp< T, Scheme >( ep, a );
    seriesExp< T, Scheme >( em, neg_a );

    std::array< T, S > h{}, s{};
    h[0] = cosh( a[0] );
    s[0] = sinh( a[0] );
    for ( std::size_t i = 1; i < S; ++i )
    {
        h[i] = ( ep[i] + em[i] ) * T{ 0.5 };
        s[i] = ( ep[i] - em[i] ) * T{ 0.5 };
    }

    out = {};
    out[0] = tanh( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    // Solve  cosh * tanh = sinh  degree-by-degree:
    //   cosh[0]*out[d] = sinh[d] - sum_{k=1}^{d} cosh[k]*out[d-k]
    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = s[std::size_t( d )];
            for ( int k = 1; k <= d; ++k ) rhs -= h[std::size_t( k )] * out[std::size_t( d - k )];
            out[std::size_t( d )] = rhs * inv_h0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = s[ai];
                for ( const RecurrenceEntry& e : row ) rhs -= h[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv_h0;
            } );
    }
}

/// Inverse hyperbolic sine series `out = asinh(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesAsinh( std::array< T, Scheme::nCoeff >& out,
                  const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::asinh;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = sqrt(1 + a^2)
    std::array< T, S > asq{}, opf{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    opf = {};
    opf[0] = T{ 1 };
    for ( std::size_t i = 0; i < S; ++i ) opf[i] += asq[i];
    seriesSqrt< T, Scheme >( h, opf );

    out = {};
    out[0] = asinh( a[0] );
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

/// Inverse hyperbolic cosine series `out = acosh(a)` (scheme-generic). Requires `a[0] > 1`.
template < typename T, tax::IndexScheme Scheme >
void seriesAcosh( std::array< T, Scheme::nCoeff >& out,
                  const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::acosh;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = sqrt(a^2 - 1)
    std::array< T, S > asq{}, amf{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    amf = asq;
    amf[0] -= T{ 1 };
    seriesSqrt< T, Scheme >( h, amf );

    out = {};
    out[0] = acosh( a[0] );
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

/// Inverse hyperbolic tangent series `out = atanh(a)` (scheme-generic). Requires `|a[0]| < 1`.
template < typename T, tax::IndexScheme Scheme >
void seriesAtanh( std::array< T, Scheme::nCoeff >& out,
                  const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::atanh;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = 1 - a^2
    std::array< T, S > h{};
    tax::cauchySelfProduct< T, Scheme >( h, a );
    for ( std::size_t i = 0; i < S; ++i ) h[i] = -h[i];
    h[0] += T{ 1 };

    out = {};
    out[0] = atanh( a[0] );
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

/// Error function series `out = erf(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesErf( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::erf;
    constexpr std::size_t S = Scheme::nCoeff;
    // Name the constant in the underlying real scalar so vectorised coefficient
    // types (whose lanes are floating-point) work too; broadcast into T.
    using R = real_scalar_t< T >;
    const T two_over_sqrtpi = T( R{ 2 } * std::numbers::inv_sqrtpi_v< R > );

    // h = (2/sqrt(pi)) * exp(-a^2)
    std::array< T, S > asq{}, neg_asq{}, e{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    neg_asq = asq;
    for ( std::size_t i = 0; i < S; ++i ) neg_asq[i] = -neg_asq[i];
    seriesExp< T, Scheme >( e, neg_asq );
    h = e;
    for ( std::size_t i = 0; i < S; ++i ) h[i] *= two_over_sqrtpi;

    out = {};
    out[0] = erf( a[0] );

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k )
                rhs += T( d - k ) * a[std::size_t( d - k )] * h[std::size_t( k )];
            out[std::size_t( d )] = rhs / T( d );
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row ) rhs += T( e.db ) * a[e.b_idx] * h[e.g_idx];
                out[ai] = rhs / T( d );
            } );
    }
}

}  // namespace tax::detail::kernels
