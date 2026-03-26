#pragma once

#include <tax/basis/taylor_traits.hpp>
#include <tax/kernels.hpp>
#include <tax/utils/aliases.hpp>
#include <tax/utils/combinatorics.hpp>
#include <tax/utils/enumeration.hpp>
#include <tax/utils/fwd.hpp>

namespace tax::detail
{

// =============================================================================
// Shared convert→operate→convert helpers for non-Taylor bases
// =============================================================================

/**
 * @brief Multiply in a non-Taylor basis via convert→Cauchy→convert back.
 * @param toMono   Function (out, in) converting basis→monomial.
 * @param fromMono Function (out, in) converting monomial→basis.
 */
template < typename T, int N, int M, typename ToMono, typename FromMono >
static void convertMultiply( CoeffArray< T, N, M >& out,
                             const CoeffArray< T, N, M >& a,
                             const CoeffArray< T, N, M >& b,
                             ToMono toMono, FromMono fromMono ) noexcept
{
    constexpr auto nC = numMonomials( N, M );
    std::array< T, nC > ma{}, mb{}, mo{};
    toMono( ma, a );
    toMono( mb, b );
    cauchyProduct< T, N, M >( mo, ma, mb );
    fromMono( out, mo );
}

/// @brief Multiply-accumulate via convertMultiply + addInPlace.
template < typename T, int N, int M, typename ToMono, typename FromMono >
static void convertMultiplyAccumulate( CoeffArray< T, N, M >& out,
                                       const CoeffArray< T, N, M >& a,
                                       const CoeffArray< T, N, M >& b,
                                       ToMono toMono, FromMono fromMono ) noexcept
{
    constexpr auto nC = numMonomials( N, M );
    std::array< T, nC > tmp{};
    convertMultiply< T, N, M >( tmp, a, b, toMono, fromMono );
    addInPlace< T, nC >( out, tmp );
}

/// @brief Reciprocal via convert→seriesReciprocal→convert back.
template < typename T, int N, int M, typename ToMono, typename FromMono >
static void convertReciprocal( CoeffArray< T, N, M >& out,
                               const CoeffArray< T, N, M >& a,
                               ToMono toMono, FromMono fromMono ) noexcept
{
    constexpr auto nC = numMonomials( N, M );
    std::array< T, nC > ma{}, mo{};
    toMono( ma, a );
    seriesReciprocal< T, N, M >( mo, ma );
    fromMono( out, mo );
}

/// @brief Differentiate via convert→Taylor differentiate→convert back.
template < typename T, int N, int M, typename ToMono, typename FromMono >
static void convertDifferentiate( CoeffArray< T, N, M >& out,
                                  const CoeffArray< T, N, M >& in, int var,
                                  ToMono toMono, FromMono fromMono ) noexcept
{
    constexpr auto nC = numMonomials( N, M );
    std::array< T, nC > mono_in{}, mono_out{};
    toMono( mono_in, in );
    BasisTraits< Taylor >::differentiate< T, N, M >( mono_out, mono_in, var );
    fromMono( out, mono_out );
}

/// @brief Integrate via convert→Taylor integrate→convert back.
template < typename T, int N, int M, typename ToMono, typename FromMono >
static void convertIntegrate( CoeffArray< T, N, M >& out,
                              const CoeffArray< T, N, M >& in, int var,
                              ToMono toMono, FromMono fromMono ) noexcept
{
    constexpr auto nC = numMonomials( N, M );
    std::array< T, nC > mono_in{}, mono_out{};
    toMono( mono_in, in );
    BasisTraits< Taylor >::integrate< T, N, M >( mono_out, mono_in, var );
    fromMono( out, mono_out );
}

// =============================================================================
// Shared multivariate evaluation via basis value precomputation
// =============================================================================

/**
 * @brief Evaluate a multivariate polynomial in any basis whose values can be computed
 *        via a three-term recurrence.
 * @param fillBasisValues Callable(array<T,N+1>&, T x_i) that fills basis_vals[0..N]
 *                        for a single variable evaluated at x_i.
 */
template < typename T, int N, int M, typename FillBasis >
[[nodiscard]] static constexpr T evaluateMultivariate(
    const CoeffArray< T, N, M >& c,
    const std::array< T, M >& x,
    FillBasis fillBasisValues ) noexcept
{
    T result{};
    constexpr auto nC = numMonomials( N, M );

    // Precompute basis values for each variable
    std::array< std::array< T, N + 1 >, M > basis_vals{};
    for ( int i = 0; i < M; ++i ) fillBasisValues( basis_vals[i], x[i] );

    for ( std::size_t idx = 0; idx < nC; ++idx )
    {
        if ( c[idx] == T{} ) continue;
        auto alpha = unflatIndex< M >( idx );
        T basis_val{ 1 };
        for ( int i = 0; i < M; ++i ) basis_val *= basis_vals[i][alpha[i]];
        result += c[idx] * basis_val;
    }
    return result;
}

}  // namespace tax::detail
