#pragma once

#include <tax/kernels.hpp>

namespace tax::detail
{

template < int N, int M >
struct OpAbs
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesAbs< T, numMonomials( N, M ) >( out, a );
    }
};

/**
 * @brief Tag for `square(expr)`.
 * @details Delegates to `seriesSquare`.
 */
template < int N, int M >
struct OpSquare
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesSquare< T, N, M >( out, a );
    }
};

/**
 * @brief Tag for `cube(expr)`.
 * @details Delegates to `seriesCube`.
 */
template < int N, int M >
struct OpCube
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesCube< T, N, M >( out, a );
    }
};

/**
 * @brief Tag for `sqrt(expr)`.
 * @details Delegates to `seriesSqrt`.
 */
template < int N, int M >
struct OpSqrt
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesSqrt< T, N, M >( out, a );
    }
};

/**
 * @brief Tag for reciprocal series `1/expr`.
 * @details Delegates to `seriesReciprocal`.
 */
template < int N, int M >
struct OpReciprocal
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesReciprocal< T, N, M >( out, a );
    }
};

/**
 * @brief Tag for `sin(expr)`.
 * @details Delegates to `seriesSin`.
 */
template < int N, int M >
struct OpSin
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesSin< T, N, M >( out, a );
    }
};

/**
 * @brief Tag for `cos(expr)`.
 * @details Delegates to `seriesCos`.
 */
template < int N, int M >
struct OpCos
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesCos< T, N, M >( out, a );
    }
};

/**
 * @brief Tag for `tan(expr)`.
 * @details Delegates to `seriesTan`.
 */
template < int N, int M >
struct OpTan
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesTan< T, N, M >( out, a );
    }
};

template < int N, int M >
struct OpErf
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesErf< T, N, M >( out, a );
    }
};

template < int N, int M >
struct OpAsin
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesAsin< T, N, M >( out, a );
    }
};

template < int N, int M >
struct OpAcos
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesAcos< T, N, M >( out, a );
    }
};

template < int N, int M >
struct OpAtan
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesAtan< T, N, M >( out, a );
    }
};

template < int N, int M >
struct OpAtan2
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& y,
                                 const std::array< T, numMonomials( N, M ) >& x ) noexcept
    {
        seriesAtan2< T, N, M >( out, y, x );
    }
};

template < int N, int M >
struct OpSinh
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesSinh< T, N, M >( out, a );
    }
};

template < int N, int M >
struct OpCosh
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesCosh< T, N, M >( out, a );
    }
};

template < int N, int M >
struct OpTanh
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesTanh< T, N, M >( out, a );
    }
};

/**
 * @brief Tag for natural logarithm `log(expr)`.
 * @details Delegates to `seriesLog`.
 */
template < int N, int M >
struct OpLog
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesLog< T, N, M >( out, a );
    }
};

/**
 * @brief Tag for base-10 logarithm `log10(expr)`.
 * @details Uses `seriesLog` then scales by `1/log(10)`.
 */
template < int N, int M >
struct OpLog10
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesLog< T, N, M >( out, a );
        const T scale = T{ 1 } / std::log( T{ 10 } );
        for ( auto& v : out ) v *= scale;
    }
};

/**
 * @brief Tag for `exp(expr)`.
 * @details Delegates to `seriesExp`.
 */
template < int N, int M >
struct OpExp
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a ) noexcept
    {
        seriesExp< T, N, M >( out, a );
    }
};

/**
 * @brief Tag for `ipow(expr, n)` (integer exponent).
 * @details Delegates to `seriesIntPow`.
 */
template < int N, int M >
struct OpIPow
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a, int n ) noexcept
    {
        seriesIntPow< T, N, M >( out, a, n );
    }
};

/**
 * @brief Tag for `dpow(expr, c)` (real exponent).
 * @details Delegates to `seriesPow`.
 */
template < int N, int M >
struct OpDPow
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& a, T c ) noexcept
    {
        seriesPow< T, N, M >( out, a, c );
    }
};

/**
 * @brief Tag for `tpow(base, exponent)` — DA^DA via exp(g * log(f)).
 * @details Binary op: takes two coefficient arrays (base, exponent).
 */
template < int N, int M >
struct OpTPow
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& base,
                                 const std::array< T, numMonomials( N, M ) >& exponent ) noexcept
    {
        std::array< T, numMonomials( N, M ) > lg{}, prod{};
        seriesLog< T, N, M >( lg, base );
        cauchyProduct< T, N, M >( prod, exponent, lg );
        seriesExp< T, N, M >( out, prod );
    }
};

template < int N, int M >
struct OpHypot
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& x,
                                 const std::array< T, numMonomials( N, M ) >& y ) noexcept
    {
        constexpr auto S = numMonomials( N, M );
        std::array< T, S > xsq{}, ysq{}, sum{};
        cauchyProduct< T, N, M >( xsq, x, x );
        cauchyProduct< T, N, M >( ysq, y, y );
        sum = xsq;
        addInPlace< T, S >( sum, ysq );
        seriesSqrt< T, N, M >( out, sum );
    }
};

template < int N, int M >
struct OpHypot3
{
    template < typename T >
    static constexpr void apply( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& x,
                                 const std::array< T, numMonomials( N, M ) >& y,
                                 const std::array< T, numMonomials( N, M ) >& z ) noexcept
    {
        constexpr auto S = numMonomials( N, M );
        std::array< T, S > xsq{}, ysq{}, zsq{}, sum{};
        cauchyProduct< T, N, M >( xsq, x, x );
        cauchyProduct< T, N, M >( ysq, y, y );
        cauchyProduct< T, N, M >( zsq, z, z );
        sum = xsq;
        addInPlace< T, S >( sum, ysq );
        addInPlace< T, S >( sum, zsq );
        seriesSqrt< T, N, M >( out, sum );
    }
};

}  // namespace tax::detail
