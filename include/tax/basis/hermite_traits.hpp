#pragma once

#include <tax/basis/taylor_traits.hpp>
#include <tax/basis/transforms.hpp>
#include <tax/kernels.hpp>
#include <tax/utils/aliases.hpp>
#include <tax/utils/combinatorics.hpp>
#include <tax/utils/enumeration.hpp>
#include <tax/utils/fwd.hpp>

namespace tax
{

/**
 * @brief BasisTraits specialization for probabilist's Hermite polynomials.
 * @details All operations use the convert→monomial→apply→convert back strategy.
 *          Evaluation uses the Clenshaw algorithm for the Hermite three-term recurrence.
 */
template <>
struct BasisTraits< Hermite >
{
    /// @brief Hermite polynomial multiplication via convert→Cauchy→convert back.
    template < typename T, int N, int M >
    static void multiply(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a,
        const detail::CoeffArray< T, N, M >& b ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );
        std::array< T, nC > ma{}, mb{}, mo{};
        detail::hermiteToMonomial< T, N, M >( ma, a );
        detail::hermiteToMonomial< T, N, M >( mb, b );
        detail::cauchyProduct< T, N, M >( mo, ma, mb );
        detail::monomialToHermite< T, N, M >( out, mo );
    }

    /// @brief Hermite polynomial multiply-accumulate.
    template < typename T, int N, int M >
    static void multiplyAccumulate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a,
        const detail::CoeffArray< T, N, M >& b ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );
        std::array< T, nC > tmp{};
        multiply< T, N, M >( tmp, a, b );
        detail::addInPlace< T, nC >( out, tmp );
    }

    /// @brief Multiplicative inverse via convert→seriesReciprocal→convert back.
    template < typename T, int N, int M >
    static void reciprocal(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );
        std::array< T, nC > ma{}, mo{};
        detail::hermiteToMonomial< T, N, M >( ma, a );
        detail::seriesReciprocal< T, N, M >( mo, ma );
        detail::monomialToHermite< T, N, M >( out, mo );
    }

    /// @brief Evaluate univariate Hermite polynomial using Clenshaw algorithm.
    /// @details Uses the three-term recurrence He_{k+1} = x*He_k - k*He_{k-1}
    ///          Clenshaw: b_{N+1} = b_{N+2} = 0
    ///          For k = N, ..., 1: b_k = c_k + x*b_{k+1} - (k+1)*b_{k+2}
    ///          result = c_0 + x*b_1 - b_2
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
            T b_k1 = T{};  // b_{k+1}
            T b_k2 = T{};  // b_{k+2}

            for ( int k = N; k >= 1; --k )
            {
                T b_k = c[k] + x * b_k1 - T( k + 1 ) * b_k2;
                b_k2 = b_k1;
                b_k1 = b_k;
            }
            // k = 0: result = c_0 + x*b_1 - 1*b_2
            return c[0] + x * b_k1 - b_k2;
        }
    }

    /// @brief Evaluate multivariate Hermite polynomial.
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
            T result{};
            constexpr auto nC = detail::numMonomials( N, M );

            // Precompute He_k(x_i) for each variable i and degree k
            std::array< std::array< T, N + 1 >, M > basis_vals{};
            for ( int i = 0; i < M; ++i )
            {
                basis_vals[i][0] = T{ 1 };  // He_0 = 1
                if constexpr ( N >= 1 ) basis_vals[i][1] = x[i];  // He_1 = x
                for ( int k = 2; k <= N; ++k )
                    basis_vals[i][k] =
                        x[i] * basis_vals[i][k - 1] - T( k - 1 ) * basis_vals[i][k - 2];
            }

            for ( std::size_t idx = 0; idx < nC; ++idx )
            {
                if ( c[idx] == T{} ) continue;
                auto alpha = detail::unflatIndex< M >( idx );
                T basis_val{ 1 };
                for ( int i = 0; i < M; ++i ) basis_val *= basis_vals[i][alpha[i]];
                result += c[idx] * basis_val;
            }
            return result;
        }
    }

    /// @brief Partial derivative via convert→Taylor differentiate→convert back.
    template < typename T, int N, int M >
    static void differentiate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );
        std::array< T, nC > mono_in{}, mono_out{};
        detail::hermiteToMonomial< T, N, M >( mono_in, in );
        BasisTraits< Taylor >::differentiate< T, N, M >( mono_out, mono_in, var );
        detail::monomialToHermite< T, N, M >( out, mono_out );
    }

    /// @brief Indefinite integral via convert→Taylor integrate→convert back.
    template < typename T, int N, int M >
    static void integrate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );
        std::array< T, nC > mono_in{}, mono_out{};
        detail::hermiteToMonomial< T, N, M >( mono_in, in );
        BasisTraits< Taylor >::integrate< T, N, M >( mono_out, mono_in, var );
        detail::monomialToHermite< T, N, M >( out, mono_out );
    }

    /// @brief Convert Hermite coefficients to monomial (Taylor) coefficients.
    template < typename T, int N, int M >
    static void toMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        detail::hermiteToMonomial< T, N, M >( out, in );
    }

    /// @brief Convert monomial (Taylor) coefficients to Hermite coefficients.
    template < typename T, int N, int M >
    static void fromMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        detail::monomialToHermite< T, N, M >( out, in );
    }
};

}  // namespace tax
