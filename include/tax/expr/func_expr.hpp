#pragma once

#include <tax/expr/base.hpp>
#include <tax/expr/math_ops.hpp>

namespace tax::detail
{

/**
 * @brief Expression node applying a series function kernel `Op`.
 * @details For leaf operands, coefficients are passed directly to the kernel;
 * otherwise the operand is materialized once into a temporary buffer.
 */
template < typename E, typename Op >
class FuncExpr : public tax::Expr< FuncExpr< E, Op >, typename E::scalar_type, E::order, E::nvars >
{
   public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array< T, numMonomials( N, M ) >;

    /// @brief Construct from operand.
    explicit constexpr FuncExpr( const E& e ) noexcept : e_( e ) {}

    /// @brief Evaluate function result into `out`.
    constexpr void evalTo( coeff_array& out ) const noexcept
    {
        if constexpr ( is_leaf_v< E > )
        {
            Op::template apply< T >( out, e_.coeffs() );
        } else
        {
            coeff_array a{};
            e_.evalTo( a );
            Op::template apply< T >( out, a );
        }
    }

   private:
    stored_t< E > e_;
};

/**
 * @brief Expression node applying a parameterized series kernel `Op`.
 * @details Like FuncExpr but passes an extra parameter `P` to the kernel.
 */
template < typename E, typename Op, typename P >
class ParamFuncExpr
    : public tax::Expr< ParamFuncExpr< E, Op, P >, typename E::scalar_type, E::order, E::nvars >
{
   public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array< T, numMonomials( N, M ) >;

    /// @brief Construct from operand and parameter.
    constexpr ParamFuncExpr( const E& e, P p ) noexcept : e_( e ), p_( p ) {}

    /// @brief Evaluate parameterized function result into `out`.
    constexpr void evalTo( coeff_array& out ) const noexcept
    {
        if constexpr ( is_leaf_v< E > )
        {
            Op::template apply< T >( out, e_.coeffs(), p_ );
        } else
        {
            coeff_array a{};
            e_.evalTo( a );
            Op::template apply< T >( out, a, p_ );
        }
    }

   private:
    stored_t< E > e_;
    P p_;
};

/**
 * @brief Expression node applying a binary function kernel `Op` to two operands.
 * @details Materializes both operands then calls `Op::apply(out, la, ra)`.
 */
template < typename L, typename R, typename Op >
class BinFuncExpr
    : public tax::Expr< BinFuncExpr< L, R, Op >, typename L::scalar_type, L::order, L::nvars >
{
    static_assert( L::order == R::order && L::nvars == R::nvars &&
                   std::is_same_v< typename L::scalar_type, typename R::scalar_type > );

   public:
    using T = typename L::scalar_type;
    static constexpr int N = L::order, M = L::nvars;
    using coeff_array = std::array< T, numMonomials( N, M ) >;

    /// @brief Construct from left/right operands.
    constexpr BinFuncExpr( const L& l, const R& r ) noexcept : l_( l ), r_( r ) {}

    /// @brief Evaluate binary function result into `out`.
    constexpr void evalTo( coeff_array& out ) const noexcept
    {
        if constexpr ( is_leaf_v< L > && is_leaf_v< R > )
            Op::template apply< T >( out, l_.coeffs(), r_.coeffs() );
        else if constexpr ( is_leaf_v< R > )
        {
            coeff_array la{};
            l_.evalTo( la );
            Op::template apply< T >( out, la, r_.coeffs() );
        } else if constexpr ( is_leaf_v< L > )
        {
            coeff_array ra{};
            r_.evalTo( ra );
            Op::template apply< T >( out, l_.coeffs(), ra );
        } else
        {
            coeff_array la{}, ra{};
            l_.evalTo( la );
            r_.evalTo( ra );
            Op::template apply< T >( out, la, ra );
        }
    }

   private:
    stored_t< L > l_;
    stored_t< R > r_;
};

/**
 * @brief Expression node applying a ternary function kernel `Op` to three operands.
 * @details Materializes operands as needed, then calls `Op::apply(out, a, b, c)`.
 */
template < typename A, typename B, typename C, typename Op >
class TerFuncExpr
    : public tax::Expr< TerFuncExpr< A, B, C, Op >, typename A::scalar_type, A::order, A::nvars >
{
    static_assert( A::order == B::order && B::order == C::order && A::nvars == B::nvars &&
                   B::nvars == C::nvars &&
                   std::is_same_v< typename A::scalar_type, typename B::scalar_type > &&
                   std::is_same_v< typename B::scalar_type, typename C::scalar_type > );

   public:
    using T = typename A::scalar_type;
    static constexpr int N = A::order, M = A::nvars;
    using coeff_array = std::array< T, numMonomials( N, M ) >;

    /// @brief Construct from three operands.
    constexpr TerFuncExpr( const A& a, const B& b, const C& c ) noexcept : a_( a ), b_( b ), c_( c )
    {
    }

    /// @brief Evaluate ternary function result into `out`.
    constexpr void evalTo( coeff_array& out ) const noexcept
    {
        auto materialize = []( const auto& e, coeff_array& buf ) {
            if constexpr ( is_leaf_v< std::remove_cvref_t< decltype( e ) > > )
                buf = e.coeffs();
            else
                e.evalTo( buf );
        };
        coeff_array aa{}, bb{}, cc{};
        materialize( a_, aa );
        materialize( b_, bb );
        materialize( c_, cc );
        Op::template apply< T >( out, aa, bb, cc );
    }

   private:
    stored_t< A > a_;
    stored_t< B > b_;
    stored_t< C > c_;
};

}  // namespace tax::detail
