#pragma once

#include <tax/basis/convert_ops.hpp>
#include <tax/basis/transforms.hpp>
#include <tax/utils/aliases.hpp>
#include <tax/utils/combinatorics.hpp>

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
    /// @details Univariate uses T_i * T_j = 0.5*(T_{|i-j|} + T_{i+j}).
    ///          Multivariate uses convert→Cauchy→convert back.
    template < typename T, int N, int M >
    static void multiply(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a,
        const detail::CoeffArray< T, N, M >& b ) noexcept
    {
        if constexpr ( M == 1 )
        {
            // Univariate Chebyshev product linearization
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
            detail::convertMultiply< T, N, M >(
                out, a, b, detail::chebyshevToMonomial< T, N, M >,
                detail::monomialToChebyshev< T, N, M > );
        }
    }

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

    template < typename T, int N, int M >
    static void reciprocal(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a ) noexcept
    {
        detail::convertReciprocal< T, N, M >(
            out, a, detail::chebyshevToMonomial< T, N, M >,
            detail::monomialToChebyshev< T, N, M > );
    }

    /// @brief Evaluate univariate Chebyshev polynomial using Clenshaw algorithm.
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
            // Clenshaw: b_k = c_k + 2*x*b_{k+1} - b_{k+2}
            T b_k1 = T{};
            T b_k2 = T{};

            for ( int k = N; k >= 1; --k )
            {
                T b_k = c[k] + T{ 2 } * x * b_k1 - b_k2;
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
                        v[k] = T{ 2 } * xi * v[k - 1] - v[k - 2];
                } );
        }
    }

    /// @brief Partial derivative using Chebyshev recurrence (univariate) or
    ///        convert→Taylor→convert back (multivariate).
    template < typename T, int N, int M >
    static void differentiate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        if constexpr ( M == 1 )
        {
            (void)var;
            constexpr auto nC = detail::numMonomials( N, M );
            for ( std::size_t i = 0; i < nC; ++i ) out[i] = T{};

            if constexpr ( N == 0 ) return;

            // Chebyshev derivative recurrence
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
            detail::convertDifferentiate< T, N, M >(
                out, in, var, detail::chebyshevToMonomial< T, N, M >,
                detail::monomialToChebyshev< T, N, M > );
        }
    }

    /// @brief Indefinite integral using Chebyshev formula (univariate) or
    ///        convert→Taylor→convert back (multivariate).
    template < typename T, int N, int M >
    static void integrate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        if constexpr ( M == 1 )
        {
            (void)var;
            constexpr auto nC = detail::numMonomials( N, M );
            for ( std::size_t i = 0; i < nC; ++i ) out[i] = T{};

            // C_k = (c_{k-1} - c_{k+1}) / (2k)
            for ( int k = 1; k <= N; ++k )
            {
                T c_km1 = in[k - 1];
                T c_kp1 = ( k + 1 <= N ) ? in[k + 1] : T{};
                out[k] = ( c_km1 - c_kp1 ) / ( T{ 2 } * T( k ) );
            }
            out[0] = T{};
        }
        else
        {
            detail::convertIntegrate< T, N, M >(
                out, in, var, detail::chebyshevToMonomial< T, N, M >,
                detail::monomialToChebyshev< T, N, M > );
        }
    }

    template < typename T, int N, int M >
    static void toMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        detail::chebyshevToMonomial< T, N, M >( out, in );
    }

    template < typename T, int N, int M >
    static void fromMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        detail::monomialToChebyshev< T, N, M >( out, in );
    }
};

}  // namespace tax
