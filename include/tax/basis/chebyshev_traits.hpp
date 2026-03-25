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
 * @brief BasisTraits specialization for Chebyshev polynomials of the first kind.
 * @details Elementary function application uses the convert→Taylor→convert back strategy:
 *          inputs are converted from Chebyshev to monomial basis, the Taylor kernel is applied,
 *          and the result is converted back to Chebyshev.
 */
template <>
struct BasisTraits< Chebyshev >
{
    /// @brief Chebyshev polynomial multiplication using linearization.
    /// @details Uses: T_i(x) * T_j(x) = 0.5 * (T_{|i-j|}(x) + T_{i+j}(x)),
    ///          generalized to multivariate via convert→Taylor→convert back.
    template < typename T, int N, int M >
    static void multiply(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a,
        const detail::CoeffArray< T, N, M >& b ) noexcept
    {
        if constexpr ( M == 1 )
        {
            // Univariate Chebyshev product linearization:
            // T_i * T_j = 0.5 * (T_{|i-j|} + T_{i+j})
            constexpr auto nC = detail::numMonomials( N, M );
            for ( std::size_t i = 0; i < nC; ++i ) out[i] = T{};

            for ( int i = 0; i <= N; ++i )
            {
                if ( a[i] == T{} ) continue;
                for ( int j = 0; j <= N; ++j )
                {
                    if ( b[j] == T{} ) continue;
                    const T prod = a[i] * b[j];
                    const int sum_deg = i + j;
                    const int diff_deg = ( i >= j ) ? ( i - j ) : ( j - i );

                    if ( diff_deg <= N ) out[diff_deg] += T{ 0.5 } * prod;
                    if ( sum_deg <= N ) out[sum_deg] += T{ 0.5 } * prod;
                }
            }
        }
        else
        {
            // Multivariate: convert to monomial, multiply, convert back
            constexpr auto nC = detail::numMonomials( N, M );
            std::array< T, nC > ma{}, mb{}, mo{};
            detail::chebyshevToMonomial< T, N, M >( ma, a );
            detail::chebyshevToMonomial< T, N, M >( mb, b );
            detail::cauchyProduct< T, N, M >( mo, ma, mb );
            detail::monomialToChebyshev< T, N, M >( out, mo );
        }
    }

    /// @brief Chebyshev polynomial multiply-accumulate.
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

    /// @brief Multiplicative inverse via convert→Taylor→convert back.
    template < typename T, int N, int M >
    static void reciprocal(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );
        std::array< T, nC > ma{}, mo{};
        detail::chebyshevToMonomial< T, N, M >( ma, a );
        detail::seriesReciprocal< T, N, M >( mo, ma );
        detail::monomialToChebyshev< T, N, M >( out, mo );
    }

    /// @brief Evaluate univariate Chebyshev polynomial using Clenshaw algorithm.
    template < typename T, int N >
    [[nodiscard]] static constexpr T evaluate(
        const detail::CoeffArray< T, N, 1 >& c, T x ) noexcept
    {
        // Clenshaw recurrence for Chebyshev:
        // b_{N+1} = b_{N+2} = 0
        // b_k = c_k + 2*x*b_{k+1} - b_{k+2}
        // result = b_0 - x*b_1  (but we use c_0 + b_1*x - b_2 form for T_0 = 1)
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
                T b_k = c[k] + T{ 2 } * x * b_k1 - b_k2;
                b_k2 = b_k1;
                b_k1 = b_k;
            }
            // f(x) = c_0 * T_0(x) + b_1 * x - b_2
            //       = c_0 + x * b_1 - b_2
            return c[0] + x * b_k1 - b_k2;
        }
    }

    /// @brief Evaluate multivariate Chebyshev polynomial.
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
            // General multivariate: evaluate by computing T_alpha(x) = prod T_{alpha_i}(x_i)
            // for each multi-index alpha and summing c_alpha * T_alpha(x)
            T result{};
            constexpr auto nC = detail::numMonomials( N, M );

            // Precompute T_k(x_i) for each variable i and degree k
            std::array< std::array< T, N + 1 >, M > basis_vals{};
            for ( int i = 0; i < M; ++i )
            {
                basis_vals[i][0] = T{ 1 };  // T_0 = 1
                if constexpr ( N >= 1 ) basis_vals[i][1] = x[i];  // T_1 = x
                for ( int k = 2; k <= N; ++k )
                    basis_vals[i][k] = T{ 2 } * x[i] * basis_vals[i][k - 1] - basis_vals[i][k - 2];
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

    /// @brief Partial derivative of Chebyshev polynomial w.r.t. variable `var`.
    /// @details Uses the recurrence:
    ///   c'_{N-1} = 2*N*c_N
    ///   c'_k = c'_{k+2} + 2*(k+1)*c_{k+1}  for k = N-2, ..., 1
    ///   c'_0 = 0.5*c'_2 + c_1
    template < typename T, int N, int M >
    static void differentiate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );

        if constexpr ( M == 1 )
        {
            (void)var;
            for ( std::size_t i = 0; i < nC; ++i ) out[i] = T{};

            if constexpr ( N == 0 ) return;

            // Chebyshev derivative recurrence (univariate)
            // d/dx [sum c_k T_k(x)] = sum c'_k T_k(x)
            std::array< T, N + 1 > dp{};

            if constexpr ( N >= 1 ) dp[N - 1] = T{ 2 } * T( N ) * in[N];

            for ( int k = N - 2; k >= 1; --k )
                dp[k] = dp[k + 2] + T{ 2 } * T( k + 1 ) * in[k + 1];

            if constexpr ( N >= 2 )
                dp[0] = T{ 0.5 } * dp[2] + in[1];
            else
                dp[0] = in[1];

            for ( int k = 0; k < N; ++k ) out[k] = dp[k];
        }
        else
        {
            // Multivariate: convert→differentiate in monomial→convert back
            std::array< T, nC > mono_in{}, mono_out{};
            detail::chebyshevToMonomial< T, N, M >( mono_in, in );
            BasisTraits< Taylor >::differentiate< T, N, M >( mono_out, mono_in, var );
            detail::monomialToChebyshev< T, N, M >( out, mono_out );
        }
    }

    /// @brief Indefinite integral of Chebyshev polynomial w.r.t. variable `var`.
    /// @details Uses: integral T_n(x) dx = T_{n+1}(x)/(2(n+1)) - T_{n-1}(x)/(2(n-1))
    ///          for n >= 2, with special cases for n=0 and n=1.
    template < typename T, int N, int M >
    static void integrate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );

        if constexpr ( M == 1 )
        {
            (void)var;
            for ( std::size_t i = 0; i < nC; ++i ) out[i] = T{};

            // Chebyshev integration recurrence (univariate)
            // integral c_0 T_0 dx = c_0 * T_1
            // integral c_1 T_1 dx = c_1 * (T_0/4 + T_2/4) ... but standard formula:
            // integral T_0 dx = T_1
            // integral T_1 dx = T_2/4 + T_0/4  ... actually:
            // integral T_n dx = T_{n+1}/(2(n+1)) - T_{n-1}/(2(n-1)) for n >= 2
            // integral T_0 dx = T_1
            // integral T_1 dx = T_2/4

            // The standard Chebyshev integration formula:
            // If f(x) = sum c_k T_k(x), then F(x) = sum C_k T_k(x) where
            // C_k = (c_{k-1} - c_{k+1}) / (2k) for k >= 1
            // C_0 is the constant of integration (set to 0)

            // C_k for k=1..N
            for ( int k = 1; k <= N; ++k )
            {
                T c_km1 = in[k - 1];
                T c_kp1 = ( k + 1 <= N ) ? in[k + 1] : T{};
                out[k] = ( c_km1 - c_kp1 ) / ( T{ 2 } * T( k ) );
            }
            out[0] = T{};  // constant of integration = 0
        }
        else
        {
            // Multivariate: convert→integrate in monomial→convert back
            std::array< T, nC > mono_in{}, mono_out{};
            detail::chebyshevToMonomial< T, N, M >( mono_in, in );
            BasisTraits< Taylor >::integrate< T, N, M >( mono_out, mono_in, var );
            detail::monomialToChebyshev< T, N, M >( out, mono_out );
        }
    }

    /// @brief Convert Chebyshev coefficients to monomial (Taylor) coefficients.
    template < typename T, int N, int M >
    static void toMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        detail::chebyshevToMonomial< T, N, M >( out, in );
    }

    /// @brief Convert monomial (Taylor) coefficients to Chebyshev coefficients.
    template < typename T, int N, int M >
    static void fromMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        detail::monomialToChebyshev< T, N, M >( out, in );
    }
};

}  // namespace tax
