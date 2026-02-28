#pragma once

#include <tax/expr/arithmetic_ops.hpp>
#include <tax/expr/base.hpp>

namespace tax::detail
{

/**
 * @brief Unary expression node applying in-place operation tag `Op`.
 * @details Supports optimized add/sub behavior for negation.
 */
template < typename E, typename Op >
class UnaryExpr
    : public tax::DAExpr< UnaryExpr< E, Op >, typename E::scalar_type, E::order, E::nvars >
{
   public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array< T, numMonomials( N, M ) >;

    /// @brief Construct from operand.
    explicit constexpr UnaryExpr( const E& e ) noexcept : e_( e ) {}

    /// @brief Evaluate operation result into `out`.
    constexpr void evalTo( coeff_array& out ) const noexcept
    {
        e_.evalTo( out );
        Op::template apply< T, numMonomials( N, M ) >( out );
    }

    /// @brief Accumulate operation result into `out`.
    constexpr void addTo( coeff_array& out ) const noexcept
    {
        if constexpr ( std::is_same_v< Op, OpNeg > )
            e_.subTo( out );
        else
        {
            coeff_array tmp{};
            evalTo( tmp );
            addInPlace< T, numMonomials( N, M ) >( out, tmp );
        }
    }
    /// @brief Subtract operation result from `out`.
    constexpr void subTo( coeff_array& out ) const noexcept
    {
        if constexpr ( std::is_same_v< Op, OpNeg > )
            e_.addTo( out );
        else
        {
            coeff_array tmp{};
            evalTo( tmp );
            subInPlace< T, numMonomials( N, M ) >( out, tmp );
        }
    }

   private:
    stored_t< E > e_;
};

}  // namespace tax::detail
