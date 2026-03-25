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
 * @brief BasisTraits specialization for Legendre polynomials.
 * @details All operations use the convert→monomial→apply→convert back strategy.
 *          Evaluation uses the Clenshaw algorithm for the Legendre three-term recurrence.
 */
template <>
struct BasisTraits< Legendre >
{
    /// @brief Legendre polynomial multiplication via convert→Cauchy→convert back.
    template < typename T, int N, int M >
    static void multiply(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a,
        const detail::CoeffArray< T, N, M >& b ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );
        std::array< T, nC > ma{}, mb{}, mo{};
        detail::legendreToMonomial< T, N, M >( ma, a );
        detail::legendreToMonomial< T, N, M >( mb, b );
        detail::cauchyProduct< T, N, M >( mo, ma, mb );
        detail::monomialToLegendre< T, N, M >( out, mo );
    }

    /// @brief Legendre polynomial multiply-accumulate.
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
        detail::legendreToMonomial< T, N, M >( ma, a );
        detail::seriesReciprocal< T, N, M >( mo, ma );
        detail::monomialToLegendre< T, N, M >( out, mo );
    }

    /// @brief Evaluate univariate Legendre polynomial using Clenshaw algorithm.
    /// @details Uses the three-term recurrence P_{k+1} = ((2k+1)/(k+1))*x*P_k - (k/(k+1))*P_{k-1}
    ///          Clenshaw: b_{N+1} = b_{N+2} = 0
    ///          For k = N, ..., 1: b_k = c_k + ((2k+1)/(k+1))*x*b_{k+1} - ((k+1)/(k+2))*b_{k+2}
    ///          result = c_0 + x*b_1 - (1/2)*b_2
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
                T b_k = c[k] + ( T( 2 * k + 1 ) / T( k + 1 ) ) * x * b_k1
                        - ( T( k + 1 ) / T( k + 2 ) ) * b_k2;
                b_k2 = b_k1;
                b_k1 = b_k;
            }
            // k = 0: result = c_0 + x*b_1 - (1/2)*b_2
            return c[0] + x * b_k1 - T{ 0.5 } * b_k2;
        }
    }

    /// @brief Evaluate multivariate Legendre polynomial.
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

            // Precompute P_k(x_i) for each variable i and degree k
            std::array< std::array< T, N + 1 >, M > basis_vals{};
            for ( int i = 0; i < M; ++i )
            {
                basis_vals[i][0] = T{ 1 };  // P_0 = 1
                if constexpr ( N >= 1 ) basis_vals[i][1] = x[i];  // P_1 = x
                for ( int k = 2; k <= N; ++k )
                    basis_vals[i][k] = ( T( 2 * k - 1 ) * x[i] * basis_vals[i][k - 1]
                                         - T( k - 1 ) * basis_vals[i][k - 2] )
                                       / T( k );
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
        detail::legendreToMonomial< T, N, M >( mono_in, in );
        BasisTraits< Taylor >::differentiate< T, N, M >( mono_out, mono_in, var );
        detail::monomialToLegendre< T, N, M >( out, mono_out );
    }

    /// @brief Indefinite integral via convert→Taylor integrate→convert back.
    template < typename T, int N, int M >
    static void integrate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );
        std::array< T, nC > mono_in{}, mono_out{};
        detail::legendreToMonomial< T, N, M >( mono_in, in );
        BasisTraits< Taylor >::integrate< T, N, M >( mono_out, mono_in, var );
        detail::monomialToLegendre< T, N, M >( out, mono_out );
    }

    /// @brief Convert Legendre coefficients to monomial (Taylor) coefficients.
    template < typename T, int N, int M >
    static void toMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        detail::legendreToMonomial< T, N, M >( out, in );
    }

    /// @brief Convert monomial (Taylor) coefficients to Legendre coefficients.
    template < typename T, int N, int M >
    static void fromMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        detail::monomialToLegendre< T, N, M >( out, in );
    }
};

}  // namespace tax
