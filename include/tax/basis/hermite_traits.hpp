#pragma once

#include <tax/basis/convert_ops.hpp>
#include <tax/basis/transforms.hpp>
#include <tax/utils/aliases.hpp>
#include <tax/utils/combinatorics.hpp>

namespace tax
{

/**
 * @brief BasisTraits specialization for probabilist's Hermite polynomials.
 * @details All non-evaluation operations use the convert→monomial→apply→convert back strategy.
 *          Evaluation uses the Clenshaw algorithm for the Hermite three-term recurrence.
 */
template <>
struct BasisTraits< Hermite >
{
    template < typename T, int N, int M >
    static void multiply(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a,
        const detail::CoeffArray< T, N, M >& b ) noexcept
    {
        detail::convertMultiply< T, N, M >(
            out, a, b, detail::hermiteToMonomial< T, N, M >,
            detail::monomialToHermite< T, N, M > );
    }

    template < typename T, int N, int M >
    static void multiplyAccumulate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a,
        const detail::CoeffArray< T, N, M >& b ) noexcept
    {
        detail::convertMultiplyAccumulate< T, N, M >(
            out, a, b, detail::hermiteToMonomial< T, N, M >,
            detail::monomialToHermite< T, N, M > );
    }

    template < typename T, int N, int M >
    static void reciprocal(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a ) noexcept
    {
        detail::convertReciprocal< T, N, M >(
            out, a, detail::hermiteToMonomial< T, N, M >,
            detail::monomialToHermite< T, N, M > );
    }

    /// @brief Evaluate univariate Hermite polynomial using Clenshaw algorithm.
    /// @details Recurrence: He_{k+1} = x*He_k - k*He_{k-1}
    template < typename T, int N >
    [[nodiscard]] static constexpr T evaluate(
        const detail::CoeffArray< T, N, 1 >& c, T x ) noexcept
    {
        if constexpr ( N == 0 )
        {
            return c[0];
        }
        else
        {
            T b_k1 = T{};
            T b_k2 = T{};

            for ( int k = N; k >= 1; --k )
            {
                T b_k = c[k] + x * b_k1 - T( k + 1 ) * b_k2;
                b_k2 = b_k1;
                b_k1 = b_k;
            }
            return c[0] + x * b_k1 - b_k2;
        }
    }

    template < typename T, int N, int M >
    [[nodiscard]] static constexpr T evaluate(
        const detail::CoeffArray< T, N, M >& c,
        const std::array< T, M >& x ) noexcept
    {
        if constexpr ( M == 1 )
        {
            return evaluate< T, N >( c, x[0] );
        }
        else
        {
            return detail::evaluateMultivariate< T, N, M >(
                c, x, []( std::array< T, N + 1 >& v, T xi ) constexpr noexcept {
                    v[0] = T{ 1 };
                    if constexpr ( N >= 1 ) v[1] = xi;
                    for ( int k = 2; k <= N; ++k )
                        v[k] = xi * v[k - 1] - T( k - 1 ) * v[k - 2];
                } );
        }
    }

    template < typename T, int N, int M >
    static void differentiate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        detail::convertDifferentiate< T, N, M >(
            out, in, var, detail::hermiteToMonomial< T, N, M >,
            detail::monomialToHermite< T, N, M > );
    }

    template < typename T, int N, int M >
    static void integrate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        detail::convertIntegrate< T, N, M >(
            out, in, var, detail::hermiteToMonomial< T, N, M >,
            detail::monomialToHermite< T, N, M > );
    }

    template < typename T, int N, int M >
    static void toMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        detail::hermiteToMonomial< T, N, M >( out, in );
    }

    template < typename T, int N, int M >
    static void fromMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        detail::monomialToHermite< T, N, M >( out, in );
    }
};

}  // namespace tax
