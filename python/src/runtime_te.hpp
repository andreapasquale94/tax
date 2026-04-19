#pragma once

// ---------------------------------------------------------------------------
// runtime_te.hpp — runtime-dimensioned wrapper around tax::TruncatedExpansionT.
//
// The public API is a single `PyTE` struct storing (N, M) and a flat vector of
// Taylor-basis coefficients. All operations dispatch at runtime to precompiled
// `tax::TEn<N, M>` instantiations through the `dispatchNM` template below.
//
// Supported (N, M) pairs are those with:
//     1 <= N <= TAX_PY_MAX_N
//     1 <= M <= TAX_PY_MAX_M
//     C(N + M, M) <= TAX_PY_MAX_COEFFS
// ---------------------------------------------------------------------------

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tax/tax.hpp>
#include <tax/utils/combinatorics.hpp>
#include <utility>
#include <vector>

#ifndef TAX_PY_MAX_N
#define TAX_PY_MAX_N 20
#endif
#ifndef TAX_PY_MAX_M
#define TAX_PY_MAX_M 10
#endif
#ifndef TAX_PY_MAX_COEFFS
#define TAX_PY_MAX_COEFFS 10000
#endif

namespace tax::py
{

// ---------------------------------------------------------------------------
// Feasibility predicate: which (N, M) pairs are actually compiled in.
// ---------------------------------------------------------------------------
template < int N, int M >
constexpr bool isFeasible()
{
    return N >= 1 && N <= TAX_PY_MAX_N && M >= 1 && M <= TAX_PY_MAX_M &&
           tax::detail::numMonomials( N, M ) <= static_cast< std::size_t >( TAX_PY_MAX_COEFFS );
}

// ---------------------------------------------------------------------------
// PyTE — runtime TE. Coefficients are stored in graded-lex (monomial/Taylor)
// order, matching `tax::TEn<N, M>::coeffs()`.
// ---------------------------------------------------------------------------
struct PyTE
{
    int order = 0;
    int nvars = 0;
    std::vector< double > coeffs;

    PyTE() = default;
    PyTE( int n, int m, std::vector< double > c ) : order( n ), nvars( m ), coeffs( std::move( c ) )
    {
    }
    PyTE( int n, int m ) : order( n ), nvars( m ), coeffs( tax::detail::numMonomials( n, m ), 0.0 )
    {
    }

    [[nodiscard]] std::size_t size() const { return coeffs.size(); }
    [[nodiscard]] double value() const { return coeffs.empty() ? 0.0 : coeffs[0]; }
};

inline void checkCompatible( const PyTE& a, const PyTE& b, const char* op )
{
    if ( a.order != b.order || a.nvars != b.nvars )
        throw std::invalid_argument( std::string( "tax: incompatible operands for " ) + op +
                                     ": (N=" + std::to_string( a.order ) +
                                     ",M=" + std::to_string( a.nvars ) + ") vs (N=" +
                                     std::to_string( b.order ) + ",M=" +
                                     std::to_string( b.nvars ) + ")" );
}

// ---------------------------------------------------------------------------
// Conversions between PyTE coefficient vector and tax::TEn<N, M>.
// ---------------------------------------------------------------------------
template < int N, int M >
[[nodiscard]] inline tax::TEn< N, M > toTE( const PyTE& p )
{
    tax::TEn< N, M > out;
    auto& data = out.coeffs();
    const std::size_t n = data.size();
    if ( p.coeffs.size() != n )
        throw std::runtime_error( "tax: coefficient vector size mismatch in toTE" );
    for ( std::size_t i = 0; i < n; ++i ) data[i] = p.coeffs[i];
    return out;
}

template < int N, int M >
[[nodiscard]] inline PyTE fromTE( const tax::TEn< N, M >& t )
{
    PyTE out( N, M );
    const auto& data = t.coeffs();
    for ( std::size_t i = 0; i < data.size(); ++i ) out.coeffs[i] = data[i];
    return out;
}

// ---------------------------------------------------------------------------
// Compile-time (N, M) dispatch.
//
// Calls `f.template operator()<N, M>()` with the first (N, M) pair that is
// both feasible and matches the runtime (n, m). Throws on miss.
// ---------------------------------------------------------------------------
namespace detail
{

template < int N, int M, typename F >
inline auto tryCall( F& f, int m, bool& dispatched )
{
    using R = decltype( f.template operator()< N, 1 >() );
    R result{};
    if constexpr ( isFeasible< N, M >() )
    {
        if ( m == M )
        {
            result = f.template operator()< N, M >();
            dispatched = true;
            return result;
        }
    }
    if constexpr ( M < TAX_PY_MAX_M )
    {
        return tryCall< N, M + 1, F >( f, m, dispatched );
    }
    return result;
}

template < int N, typename F >
inline auto dispatchForN( F& f, int m, bool& dispatched )
{
    return tryCall< N, 1, F >( f, m, dispatched );
}

template < int N, typename F >
inline auto dispatchChain( int n, int m, F& f, bool& dispatched )
{
    using R = decltype( f.template operator()< 1, 1 >() );
    if ( n == N )
    {
        return dispatchForN< N, F >( f, m, dispatched );
    }
    if constexpr ( N < TAX_PY_MAX_N )
    {
        return dispatchChain< N + 1, F >( n, m, f, dispatched );
    }
    return R{};
}

}  // namespace detail

template < typename F >
inline auto dispatchNM( int n, int m, F&& f )
{
    using R = decltype( f.template operator()< 1, 1 >() );
    if ( n < 1 || n > TAX_PY_MAX_N || m < 1 || m > TAX_PY_MAX_M )
        throw std::invalid_argument( "tax: (N=" + std::to_string( n ) +
                                     ", M=" + std::to_string( m ) +
                                     ") out of range for this build" );
    bool dispatched = false;
    R result = detail::dispatchChain< 1, F >( n, m, f, dispatched );
    if ( !dispatched )
        throw std::invalid_argument( "tax: (N=" + std::to_string( n ) +
                                     ", M=" + std::to_string( m ) +
                                     ") is not a supported pair (exceeds coefficient cap)" );
    return result;
}

// ---------------------------------------------------------------------------
// Variant: dispatch restricted to a single value of M (used by per-M TUs).
// ---------------------------------------------------------------------------
namespace detail
{
template < int M, int N, typename F >
inline auto dispatchNForM( int n, F& f, bool& dispatched )
{
    using R = decltype( f.template operator()< 1, M >() );
    if constexpr ( isFeasible< N, M >() )
    {
        if ( n == N )
        {
            R r = f.template operator()< N, M >();
            dispatched = true;
            return r;
        }
    }
    if constexpr ( N < TAX_PY_MAX_N )
    {
        return dispatchNForM< M, N + 1, F >( n, f, dispatched );
    }
    return R{};
}
}  // namespace detail

template < int M, typename F >
inline auto dispatchN( int n, F&& f )
{
    using R = decltype( f.template operator()< 1, M >() );
    bool dispatched = false;
    R r = detail::dispatchNForM< M, 1, F >( n, f, dispatched );
    if ( !dispatched )
        throw std::invalid_argument( "tax: N=" + std::to_string( n ) + " not supported for M=" +
                                     std::to_string( M ) );
    return r;
}

}  // namespace tax::py
