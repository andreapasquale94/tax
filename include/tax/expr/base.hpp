#pragma once

#include <tax/kernels.hpp>
#include <tax/utils/combinatorics.hpp>
#include <tax/utils/fwd.hpp>
#include <tax/utils/leaf.hpp>

namespace tax
{

// =============================================================================
// CRTP expression base
// =============================================================================

/**
 * @brief CRTP base class for DA expression nodes.
 * @tparam Derived Concrete expression type.
 * @tparam T Scalar coefficient type.
 * @tparam N Maximum total polynomial order.
 * @tparam M Number of variables.
 */
template < typename Derived, typename T, int N, int M >
struct Expr
{
    using scalar_type = T;
    static constexpr int order = N;
    static constexpr int nvars = M;
    static constexpr std::size_t nCoefficients = detail::numMonomials( N, M );
    using Data = std::array< T, nCoefficients >;

    /// @brief Access the concrete expression implementation.
    [[nodiscard]] constexpr const Derived& self() const noexcept
    {
        return static_cast< const Derived& >( *this );
    }

    /**
     * @brief Evaluate this expression into `out`.
     * @param out Destination coefficient buffer.
     */
    constexpr void evalTo( Data& out ) const noexcept { self().evalTo( out ); }

    /**
     * @brief Accumulate this expression into `out`.
     * @param out Destination buffer updated as `out += eval()`.
     */
    constexpr void addTo( Data& out ) const noexcept
    {
        Data tmp{};
        self().evalTo( tmp );
        detail::addInPlace< T, nCoefficients >( out, tmp );
    }
    /**
     * @brief Subtract this expression from `out`.
     * @param out Destination buffer updated as `out -= eval()`.
     */
    constexpr void subTo( Data& out ) const noexcept
    {
        Data tmp{};
        self().evalTo( tmp );
        detail::subInPlace< T, nCoefficients >( out, tmp );
    }

    /// @brief Materialize this expression as a coefficient array.
    [[nodiscard]] constexpr Data eval() const noexcept
    {
        Data out{};
        self().evalTo( out );
        return out;
    }

    /// @brief Value at the expansion point.
    [[nodiscard]] constexpr T value() const noexcept { return eval()[0]; }

    /// @brief Coefficient corresponding to multi-index `alpha`.
    [[nodiscard]] constexpr T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return eval()[detail::flatIndex< M >( alpha )];
    }

    /**
     * @brief Partial derivative selected by `alpha` at the expansion point.
     * @details Returns `coeff(alpha) * prod_i alpha[i]!`.
     */
    [[nodiscard]] constexpr T derivative( const MultiIndex< M >& alpha ) const noexcept
    {
        std::size_t fac = 1;
        for ( int i = 0; i < M; ++i )
            for ( int j = 1; j <= alpha[i]; ++j ) fac *= std::size_t( j );
        return eval()[detail::flatIndex< M >( alpha )] * T( fac );
    }
};

}  // namespace tax
