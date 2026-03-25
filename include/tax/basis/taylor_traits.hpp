#pragma once

#include <tax/basis/traits.hpp>
#include <tax/kernels.hpp>
#include <tax/utils/aliases.hpp>
#include <tax/utils/combinatorics.hpp>
#include <tax/utils/enumeration.hpp>
#include <tax/utils/fwd.hpp>

namespace tax
{

/**
 * @brief BasisTraits specialization for the monomial (Taylor) basis.
 * @details Delegates to the existing kernel implementations.
 */
template <>
struct BasisTraits< Taylor >
{
    /// @brief Truncated polynomial multiplication (Cauchy product).
    template < typename T, int N, int M >
    static constexpr void multiply(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a,
        const detail::CoeffArray< T, N, M >& b ) noexcept
    {
        detail::cauchyProduct< T, N, M >( out, a, b );
    }

    /// @brief Truncated polynomial multiply-accumulate.
    template < typename T, int N, int M >
    static constexpr void multiplyAccumulate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a,
        const detail::CoeffArray< T, N, M >& b ) noexcept
    {
        detail::cauchyAccumulate< T, N, M >( out, a, b );
    }

    /// @brief Multiplicative inverse.
    template < typename T, int N, int M >
    static constexpr void reciprocal(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& a ) noexcept
    {
        detail::seriesReciprocal< T, N, M >( out, a );
    }

    /// @brief Evaluate univariate polynomial at displacement dx (Horner's method).
    template < typename T, int N >
    [[nodiscard]] static constexpr T evaluate(
        const detail::CoeffArray< T, N, 1 >& c, T dx ) noexcept
    {
        T result = c[N];
        for ( int i = N - 1; i >= 0; --i ) result = result * dx + c[i];
        return result;
    }

    /// @brief Evaluate multivariate polynomial at displacement dx.
    template < typename T, int N, int M >
    [[nodiscard]] static constexpr T evaluate(
        const detail::CoeffArray< T, N, M >& c,
        const std::array< T, M >& dx ) noexcept
    {
        if constexpr ( M == 1 )
        {
            return evaluate< T, N >( c, dx[0] );
        }
        else
        {
            T result{};
            MultiIndex< M > alpha{};

            auto accumulate = [&]( auto& self, int var, int rem ) constexpr -> void {
                if ( var == M - 1 )
                {
                    alpha[var] = rem;
                    T monomial{ 1 };
                    for ( int i = 0; i < M; ++i )
                        for ( int j = 0; j < alpha[i]; ++j ) monomial *= dx[i];
                    result += c[detail::flatIndex< M >( alpha )] * monomial;
                    return;
                }
                for ( int k = rem; k >= 0; --k )
                {
                    alpha[var] = k;
                    self( self, var + 1, rem - k );
                }
            };

            for ( int d = 0; d <= N; ++d ) accumulate( accumulate, 0, d );
            return result;
        }
    }

    /// @brief Partial derivative polynomial w.r.t. variable `var`.
    template < typename T, int N, int M >
    static constexpr void differentiate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );
        for ( std::size_t i = 0; i < nC; ++i ) out[i] = T{};

        for ( std::size_t i = 0; i < nC; ++i )
        {
            if ( in[i] == T{} ) continue;
            auto alpha = detail::unflatIndex< M >( i );
            const int exp = alpha[var];
            if ( exp == 0 ) continue;
            alpha[var] = exp - 1;
            out[detail::flatIndex< M >( alpha )] += in[i] * T( exp );
        }
    }

    /// @brief Indefinite integral polynomial w.r.t. variable `var`.
    template < typename T, int N, int M >
    static constexpr void integrate(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in, int var ) noexcept
    {
        constexpr auto nC = detail::numMonomials( N, M );
        for ( std::size_t i = 0; i < nC; ++i ) out[i] = T{};

        for ( std::size_t i = 0; i < nC; ++i )
        {
            if ( in[i] == T{} ) continue;
            auto alpha = detail::unflatIndex< M >( i );
            if ( detail::totalDegree< M >( alpha ) >= N ) continue;
            const int exp = alpha[var];
            alpha[var] = exp + 1;
            out[detail::flatIndex< M >( alpha )] = in[i] / T( exp + 1 );
        }
    }

    /// @brief Identity transform (monomial IS the native basis for Taylor).
    template < typename T, int N, int M >
    static constexpr void toMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        out = in;
    }

    /// @brief Identity transform (monomial IS the native basis for Taylor).
    template < typename T, int N, int M >
    static constexpr void fromMonomial(
        detail::CoeffArray< T, N, M >& out,
        const detail::CoeffArray< T, N, M >& in ) noexcept
    {
        out = in;
    }
};

}  // namespace tax
