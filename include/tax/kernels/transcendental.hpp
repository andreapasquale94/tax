#pragma once

#include <cmath>
#include <numbers>
#include <span>
#include <tax/core/concepts.hpp>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/kernels/algebra.hpp>

namespace tax::detail::kernels
{

/// Natural exponential series `out = exp(a)` (scheme-generic): out' = a' * out.
template < typename T, tax::IndexScheme Scheme >
void seriesExp( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::exp;
    seriesDerivProduct< T, Scheme >( out, exp( a[0] ), a, out );
}

/// Natural logarithm series `out = log(a)` (scheme-generic). Requires `a[0] > 0`.
template < typename T, tax::IndexScheme Scheme >
void seriesLog( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::log;
    seriesDerivQuotient< 1, T, Scheme >( out, log( a[0] ), a, a );
}

/// Joint `exp(a)` / `exp(-a)` pair: one negation, two recurrence passes.
template < typename T, tax::IndexScheme Scheme >
void seriesExpPair( std::array< T, Scheme::nCoeff >& ep, std::array< T, Scheme::nCoeff >& em,
                    const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    std::array< T, Scheme::nCoeff > neg_a = a;
    for ( T& v : neg_a ) v = -v;
    seriesExp< T, Scheme >( ep, a );
    seriesExp< T, Scheme >( em, neg_a );
}

/// Coupled hyperbolic series: jointly compute `sinh(a)` and `cosh(a)` from one
/// exp(a)/exp(-a) pair (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesSinhCosh( std::array< T, Scheme::nCoeff >& s, std::array< T, Scheme::nCoeff >& c,
                     const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::cosh;
    using std::sinh;
    constexpr std::size_t S = Scheme::nCoeff;
    std::array< T, S > ep{}, em{};
    seriesExpPair< T, Scheme >( ep, em, a );

    s = {};
    c = {};
    s[0] = sinh( a[0] );
    c[0] = cosh( a[0] );
    for ( std::size_t i = 1; i < S; ++i )
    {
        s[i] = ( ep[i] - em[i] ) * T{ 0.5 };
        c[i] = ( ep[i] + em[i] ) * T{ 0.5 };
    }
}

/// Hyperbolic sine series `out = sinh(a)` (scheme-generic). Single-output on
/// purpose — writing the discarded cosh companion measurably costs at small N;
/// callers that want both should use seriesSinhCosh.
template < typename T, tax::IndexScheme Scheme >
void seriesSinh( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::sinh;
    constexpr std::size_t S = Scheme::nCoeff;
    std::array< T, S > ep{}, em{};
    seriesExpPair< T, Scheme >( ep, em, a );

    out = {};
    out[0] = sinh( a[0] );
    for ( std::size_t i = 1; i < S; ++i ) out[i] = ( ep[i] - em[i] ) * T{ 0.5 };
}

/// Hyperbolic cosine series `out = cosh(a)` (scheme-generic; single-output,
/// see seriesSinh).
template < typename T, tax::IndexScheme Scheme >
void seriesCosh( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::cosh;
    constexpr std::size_t S = Scheme::nCoeff;
    std::array< T, S > ep{}, em{};
    seriesExpPair< T, Scheme >( ep, em, a );

    out = {};
    out[0] = cosh( a[0] );
    for ( std::size_t i = 1; i < S; ++i ) out[i] = ( ep[i] + em[i] ) * T{ 0.5 };
}

/// Hyperbolic tangent series `out = tanh(a)` (scheme-generic): sinh/cosh in one
/// substitution pass over a single shared exp(a)/exp(-a) pair.
template < typename T, tax::IndexScheme Scheme >
void seriesTanh( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;
    std::array< T, S > s{}, c{};
    seriesSinhCosh< T, Scheme >( s, c, a );
    seriesDivide< T, Scheme >( out, s, c );
}

/// Inverse hyperbolic sine series `out = asinh(a)` (scheme-generic):
/// out' = a' / sqrt(1 + a^2).
template < typename T, tax::IndexScheme Scheme >
void seriesAsinh( std::array< T, Scheme::nCoeff >& out,
                  const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::asinh;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = sqrt(1 + a^2)
    std::array< T, S > asq{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    asq[0] += T{ 1 };
    seriesSqrt< T, Scheme >( h, asq );

    seriesDerivQuotient< 1, T, Scheme >( out, asinh( a[0] ), a, h );
}

/// Inverse hyperbolic cosine series `out = acosh(a)` (scheme-generic). Requires `a[0] > 1`:
/// out' = a' / sqrt(a^2 - 1).
template < typename T, tax::IndexScheme Scheme >
void seriesAcosh( std::array< T, Scheme::nCoeff >& out,
                  const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::acosh;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = sqrt(a^2 - 1)
    std::array< T, S > asq{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    asq[0] -= T{ 1 };
    seriesSqrt< T, Scheme >( h, asq );

    seriesDerivQuotient< 1, T, Scheme >( out, acosh( a[0] ), a, h );
}

/// Inverse hyperbolic tangent series `out = atanh(a)` (scheme-generic). Requires `|a[0]| < 1`:
/// out' = a' / (1 - a^2).
template < typename T, tax::IndexScheme Scheme >
void seriesAtanh( std::array< T, Scheme::nCoeff >& out,
                  const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::atanh;
    // h = 1 - a^2
    std::array< T, Scheme::nCoeff > h{};
    tax::cauchySelfProduct< T, Scheme >( h, a );
    for ( T& v : h ) v = -v;
    h[0] += T{ 1 };

    seriesDerivQuotient< 1, T, Scheme >( out, atanh( a[0] ), a, h );
}

/// Error function series `out = erf(a)` (scheme-generic):
/// out' = a' * (2/sqrt(pi)) exp(-a^2).
template < typename T, tax::IndexScheme Scheme >
void seriesErf( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::erf;
    constexpr std::size_t S = Scheme::nCoeff;
    const T two_over_sqrtpi = T{ 2 } * std::numbers::inv_sqrtpi_v< T >;

    // h = (2/sqrt(pi)) * exp(-a^2)
    std::array< T, S > asq{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    for ( T& v : asq ) v = -v;
    seriesExp< T, Scheme >( h, asq );
    for ( T& v : h ) v *= two_over_sqrtpi;

    seriesDerivProduct< T, Scheme >( out, erf( a[0] ), a, h );
}

}  // namespace tax::detail::kernels
