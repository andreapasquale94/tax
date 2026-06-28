#pragma once

// Human-readable series / stream representation for Taylor expansions.
//
//   std::cout << f;                              // polynomial series (default)
//   std::cout << tax::series( f, { ...opts... } );  // tunable / tabular / vectors
//
// Polynomial output uses Unicode subscripts for variable indices and
// superscripts for powers, with implicit multiplication. The tabular style is
// a plain-ASCII DACE-like table. Works for dense / sparse / named expansions
// and for Eigen vectors/matrices of expansions.

#include <Eigen/Core>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <tax/expansion/expansion.hpp>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/named.hpp>
#include <tax/expansion/taylor_basis.hpp>
#include <type_traits>
#include <vector>

namespace tax
{

/// Output style for the series representation.
enum class SeriesStyle
{
    Polynomial,
    Tabular
};

/// Formatting options for `series()`.
struct SeriesOptions
{
    SeriesStyle style = SeriesStyle::Polynomial;
    int precision = 6;                   ///< significant digits (polynomial); tabular uses 15
    double threshold = 0.0;              ///< drop terms with |coeff| <= threshold
    std::vector< std::string > names{};  ///< optional variable-name override
};

namespace detail
{

/// UTF-8 subscript rendering of a non-negative integer (e.g. 10 -> "₁₀").
[[nodiscard]] inline std::string subscriptOf( int n ) noexcept
{
    const std::string digits = std::to_string( n );
    std::string out;
    for ( char ch : digits )
    {
        const int d = ch - '0';
        out.push_back( char( 0xE2 ) );
        out.push_back( char( 0x82 ) );
        out.push_back( char( 0x80 + d ) );  // U+2080 + d
    }
    return out;
}

/// UTF-8 superscript rendering of a non-negative integer (e.g. 23 -> "²³").
[[nodiscard]] inline std::string superscriptOf( int n ) noexcept
{
    const std::string digits = std::to_string( n );
    std::string out;
    for ( char ch : digits )
    {
        switch ( ch )
        {
            case '1':
                out += "\xc2\xb9";
                break;  // U+00B9
            case '2':
                out += "\xc2\xb2";
                break;  // U+00B2
            case '3':
                out += "\xc2\xb3";
                break;  // U+00B3
            default:
                out.push_back( char( 0xE2 ) );
                out.push_back( char( 0x81 ) );
                out.push_back( char( 0xB0 + ( ch - '0' ) ) );  // U+2070 + d
        }
    }
    return out;
}

/// Format a scalar magnitude with the requested significant-digit precision.
template < typename T >
[[nodiscard]] inline std::string formatMagnitude( T v, int precision )
{
    std::ostringstream o;
    o << std::setprecision( precision ) << v;
    return o.str();
}

/// Default variable name for an unnamed expansion: `x` + subscript index.
[[nodiscard]] inline std::string defaultVarName( int v, const SeriesOptions& opts )
{
    if ( v < int( opts.names.size() ) ) return opts.names[std::size_t( v )];
    return "x" + subscriptOf( v );
}

/// Convert a FixedString axis name to std::string.
template < std::size_t K >
[[nodiscard]] inline std::string fixedToString( const named::FixedString< K >& s )
{
    return std::string( s.data, s.size() );
}

/// Variable name for a named expansion: axis name (1-D) or name+subscript (multi-D).
template < typename... Axes >
[[nodiscard]] inline std::string namedVarName( int v, const SeriesOptions& opts )
{
    if ( v < int( opts.names.size() ) ) return opts.names[std::size_t( v )];
    std::string result;
    int idx = v;
    bool done = false;
    auto visit = [&]< typename Ax >( Ax ) {
        if ( done ) return;
        if ( idx < Ax::dim )
        {
            const std::string nm = fixedToString( Ax::name );
            result = ( Ax::dim == 1 ) ? nm : nm + subscriptOf( idx );
            done = true;
        } else
            idx -= Ax::dim;
    };
    ( visit( Axes{} ), ... );
    return result;
}

/// Core writer shared by every expansion kind. `coeffAt(k)` reads the coefficient
/// at flat index `k`; `nameOf(v)` names variable `v`. The monomial set and
/// flat<->multi maps come from `Scheme`.
template < typename Scheme, typename T, typename CoeffAt, typename NameOf >
void writeSeries( std::ostream& os, CoeffAt&& coeffAt, NameOf&& nameOf, const SeriesOptions& opts )
{
    constexpr int M = Scheme::vars;
    const std::size_t n = Scheme::nCoeff;
    const T thr = T( opts.threshold );

    if ( opts.style == SeriesStyle::Tabular )
    {
        os << "   I  COEFFICIENT                ORDER  EXPONENTS\n";
        int row = 1;
        for ( std::size_t k = 0; k < n; ++k )
        {
            const T c = coeffAt( k );
            if ( c == T{ 0 } ) continue;
            const T mag = c < T{ 0 } ? -c : c;
            if ( mag <= thr ) continue;
            const auto alpha = Scheme::multiOf( k );
            std::ostringstream num;
            num << std::scientific << std::setprecision( 15 ) << c;
            os << std::setw( 4 ) << row++ << "  " << num.str() << "    " << std::setw( 2 )
               << totalDegree( alpha ) << "     ";
            for ( int v = 0; v < M; ++v ) os << ( v ? " " : "" ) << alpha[std::size_t( v )];
            os << "\n";
        }
        return;
    }

    // Polynomial style.
    bool any = false;
    for ( std::size_t k = 0; k < n; ++k )
    {
        const T c = coeffAt( k );
        if ( c == T{ 0 } ) continue;
        const bool neg = c < T{ 0 };
        const T mag = neg ? -c : c;
        if ( mag <= thr ) continue;
        const auto alpha = Scheme::multiOf( k );
        const bool is_const = totalDegree( alpha ) == 0;

        if ( !any )
        {
            if ( neg ) os << "-";
        } else
            os << ( neg ? " - " : " + " );
        any = true;

        if ( is_const )
            os << formatMagnitude( mag, opts.precision );
        else
        {
            if ( mag != T{ 1 } ) os << formatMagnitude( mag, opts.precision );
            for ( int v = 0; v < M; ++v )
            {
                const int e = alpha[std::size_t( v )];
                if ( e == 0 ) continue;
                os << nameOf( v );
                if ( e > 1 ) os << superscriptOf( e );
            }
        }
    }
    if ( !any ) os << "0";
}

/// Core writer for orthogonal-basis series: renders Σ coeff · Π_v B::term(deg_v, name_v).
/// Mirrors writeSeries' coefficient iteration / sign / threshold handling, but each
/// non-constant monomial is a product of per-variable basis labels, e.g. "T_2(x₀)*T_1(x₁)".
/// The coefficient is always shown (no 1-elision); the constant term prints as just the
/// coefficient.
template < typename Scheme, typename T, typename B, typename CoeffAt, typename NameOf >
void writeOrthoSeries( std::ostream& os, CoeffAt&& coeffAt, NameOf&& nameOf,
                       const SeriesOptions& opts )
{
    constexpr int M = Scheme::vars;
    const std::size_t n = Scheme::nCoeff;
    const T thr = T( opts.threshold );
    bool any = false;
    for ( std::size_t k = 0; k < n; ++k )
    {
        const T c = coeffAt( k );
        if ( c == T{ 0 } ) continue;
        const bool neg = c < T{ 0 };
        const T mag = neg ? -c : c;
        if ( mag <= thr ) continue;
        const auto alpha = Scheme::multiOf( k );
        const bool is_const = totalDegree( alpha ) == 0;

        if ( !any )
        {
            if ( neg ) os << "-";
        } else
            os << ( neg ? " - " : " + " );
        any = true;

        os << formatMagnitude( mag, opts.precision );
        if ( !is_const )
            for ( int v = 0; v < M; ++v )
            {
                const int e = alpha[std::size_t( v )];
                if ( e == 0 ) continue;
                os << "*" << B::term( e, nameOf( v ) );
            }
    }
    if ( !any ) os << "0";
}

// --- Per-kind streaming (dispatches coefficient access + naming) ------------

template < typename T, typename Scheme >
void streamScalar( std::ostream& os, const TaylorExpansion< T, Scheme, storage::Dense >& f,
                   const SeriesOptions& opts )
{
    writeSeries< Scheme, T >(
        os, [&]( std::size_t k ) { return f[k]; },
        [&]( int v ) { return defaultVarName( v, opts ); }, opts );
}

template < typename T, typename Scheme >
void streamScalar( std::ostream& os, const TaylorExpansion< T, Scheme, storage::Sparse >& f,
                   const SeriesOptions& opts )
{
    streamScalar( os, f.dense(), opts );
}

template < typename T, int N, typename... Axes >
void streamScalar( std::ostream& os, const named::NamedTaylorExpansion< T, N, Axes... >& f,
                   const SeriesOptions& opts )
{
    constexpr int M = named::NamedTaylorExpansion< T, N, Axes... >::vars_v;
    writeSeries< IsotropicScheme< N, M >, T >(
        os, [&]( std::size_t k ) { return f.inner()[k]; },
        [&]( int v ) { return namedVarName< Axes... >( v, opts ); }, opts );
}

// Unnamed orthogonal expansion (any non-Taylor basis), dense storage.
template < typename T, typename B, typename Scheme >
    requires( !std::is_same_v< B, TaylorBasis > )
void streamScalar( std::ostream& os, const Expansion< T, B, Scheme, storage::Dense >& f,
                   const SeriesOptions& opts )
{
    writeOrthoSeries< Scheme, T, B >(
        os, [&]( std::size_t k ) { return f[k]; },
        [&]( int v ) { return defaultVarName( v, opts ); }, opts );
}

// Named expansion over a non-Taylor basis (named-over-orthogonal).
template < typename T, typename B, int N, typename... Axes >
    requires( !std::is_same_v< B, TaylorBasis > )
void streamScalar( std::ostream& os, const named::NamedExpansion< T, B, N, Axes... >& f,
                   const SeriesOptions& opts )
{
    constexpr int M = named::NamedExpansion< T, B, N, Axes... >::vars_v;
    writeOrthoSeries< IsotropicScheme< N, M >, T, B >(
        os, [&]( std::size_t k ) { return f.inner()[k]; },
        [&]( int v ) { return namedVarName< Axes... >( v, opts ); }, opts );
}

// --- Streamable proxies returned by series() --------------------------------

template < typename F >
struct ScalarSeriesProxy
{
    const F& f;
    SeriesOptions opts;
};

template < typename F >
std::ostream& operator<<( std::ostream& os, const ScalarSeriesProxy< F >& p )
{
    streamScalar( os, p.f, p.opts );
    return os;
}

template < typename Derived >
struct MatrixSeriesProxy
{
    const Derived& m;
    SeriesOptions opts;
};

template < typename Derived >
std::ostream& operator<<( std::ostream& os, const MatrixSeriesProxy< Derived >& p )
{
    const auto& m = p.m;
    if ( m.rows() == 1 || m.cols() == 1 )
        for ( Eigen::Index i = 0; i < m.size(); ++i )
        {
            os << "[" << i << "]: ";
            streamScalar( os, m( i ), p.opts );
            os << "\n";
        }
    else
        for ( Eigen::Index r = 0; r < m.rows(); ++r )
            for ( Eigen::Index c = 0; c < m.cols(); ++c )
            {
                os << "[" << r << "," << c << "]: ";
                streamScalar( os, m( r, c ), p.opts );
                os << "\n";
            }
    return os;
}

}  // namespace detail

// --- Public operator<< (zero-config polynomial series) ----------------------

template < typename T, typename Scheme, typename S >
std::ostream& operator<<( std::ostream& os, const TaylorExpansion< T, Scheme, S >& f )
{
    detail::streamScalar( os, f, SeriesOptions{} );
    return os;
}

// --- Orthogonal-basis streaming (univariate + multivariate) -----------------

template < typename T, typename B, typename Scheme >
    requires( !std::is_same_v< B, TaylorBasis > )
std::ostream& operator<<( std::ostream& os, const Expansion< T, B, Scheme, storage::Dense >& f )
{
    detail::streamScalar( os, f, SeriesOptions{} );
    return os;
}

// --- series() manipulator ---------------------------------------------------

/// Stream a scalar / named expansion with explicit options (style, precision, ...).
template < typename F >
    requires( !std::is_base_of_v< Eigen::EigenBase< F >, F > )
[[nodiscard]] detail::ScalarSeriesProxy< F > series( const F& f, SeriesOptions opts = {} )
{
    return { f, opts };
}

/// Stream an Eigen vector/matrix of expansions, one labeled element per line.
template < typename Derived >
[[nodiscard]] detail::MatrixSeriesProxy< Derived > series( const Eigen::MatrixBase< Derived >& m,
                                                           SeriesOptions opts = {} )
{
    return { m.derived(), opts };
}

/// Render any streamable expansion (scalar, named, or Eigen vector/matrix) to a string.
template < typename F >
[[nodiscard]] std::string to_string( const F& f, SeriesOptions opts = {} )
{
    std::ostringstream os;
    os << series( f, opts );
    return os.str();
}

namespace named
{

/// ADL hook so `os << namedExpansion` resolves for named expansions.
template < typename T, int N, typename... Axes >
std::ostream& operator<<( std::ostream& os, const NamedTaylorExpansion< T, N, Axes... >& f )
{
    tax::detail::streamScalar( os, f, tax::SeriesOptions{} );
    return os;
}

/// ADL hook so `os << namedExpansion` resolves for named-over-orthogonal expansions.
template < typename T, typename B, int N, typename... Axes >
    requires( !std::is_same_v< B, tax::TaylorBasis > )
std::ostream& operator<<( std::ostream& os, const NamedExpansion< T, B, N, Axes... >& f )
{
    tax::detail::streamScalar( os, f, tax::SeriesOptions{} );
    return os;
}

}  // namespace named

}  // namespace tax
