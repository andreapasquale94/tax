#pragma once

#include <tax/expr/base.hpp>

namespace tax::detail
{

/**
 * @brief Flattened variadic sum expression node.
 * @details Keeps sums flat to reduce temporary nodes and improve accumulation.
 */
template < typename... Es >
class SumExpr
    : public tax::Expr< SumExpr< Es... >,
                        typename std::tuple_element_t< 0, std::tuple< Es... > >::scalar_type,
                        std::tuple_element_t< 0, std::tuple< Es... > >::order,
                        std::tuple_element_t< 0, std::tuple< Es... > >::nvars >
{
    static_assert( sizeof...( Es ) >= 2, "SumExpr needs at least 2 operands" );
    template < typename... >
    friend class SumExpr;

   public:
    using T = typename std::tuple_element_t< 0, std::tuple< Es... > >::scalar_type;
    static constexpr int N = std::tuple_element_t< 0, std::tuple< Es... > >::order;
    static constexpr int M = std::tuple_element_t< 0, std::tuple< Es... > >::nvars;
    using coeff_array = std::array< T, numMonomials( N, M ) >;

    /// @brief Construct from operand pack.
    explicit constexpr SumExpr( stored_t< Es >... es ) noexcept : ops_( es... ) {}

    template < typename E >
    /// @brief Return a new `SumExpr` with `e` appended.
    [[nodiscard]] constexpr auto append( stored_t< E > e ) const noexcept
    {
        return std::apply(
            [&]( auto const&... x ) noexcept { return SumExpr< Es..., E >( x..., e ); }, ops_ );
    }

    template < typename E >
    /// @brief Return a new `SumExpr` with `e` prepended.
    [[nodiscard]] constexpr auto prepend( stored_t< E > e ) const noexcept
    {
        return std::apply(
            [&]( auto const&... x ) noexcept { return SumExpr< E, Es... >( e, x... ); }, ops_ );
    }

    template < typename... Rs >
    /// @brief Concatenate two flattened sums.
    [[nodiscard]] constexpr auto concat( const SumExpr< Rs... >& r ) const noexcept
    {
        return std::apply(
            [&]( auto const&... rx ) noexcept {
                return std::apply(
                    [&]( auto const&... lx ) noexcept {
                        return SumExpr< Es..., Rs... >( lx..., rx... );
                    },
                    ops_ );
            },
            r.ops_ );
    }

    /// @brief Evaluate the full sum into `out`.
    constexpr void evalTo( coeff_array& out ) const noexcept
    {
        std::get< 0 >( ops_ ).evalTo( out );
        accumRest( out, std::make_index_sequence< sizeof...( Es ) - 1 >{} );
    }

    /// @brief Accumulate this sum into `out`.
    constexpr void addTo( coeff_array& out ) const noexcept
    {
        std::apply( [&]( auto const&... e ) noexcept { ( e.addTo( out ), ... ); }, ops_ );
    }

    /// @brief Subtract this sum from `out`.
    constexpr void subTo( coeff_array& out ) const noexcept
    {
        std::apply( [&]( auto const&... e ) noexcept { ( e.subTo( out ), ... ); }, ops_ );
    }

   private:
    std::tuple< stored_t< Es >... > ops_;

    template < std::size_t... I >
    constexpr void accumRest( coeff_array& out, std::index_sequence< I... > ) const noexcept
    {
        ( accumOne< I + 1 >( out ), ... );
    }

    template < std::size_t I >
    constexpr void accumOne( coeff_array& out ) const noexcept
    {
        using E = std::tuple_element_t< I, std::tuple< Es... > >;
        if constexpr ( is_leaf_v< E > )
        {
            addInPlace< T, numMonomials( N, M ) >( out, std::get< I >( ops_ ).coeffs() );
        } else
        {
            coeff_array tmp{};
            std::get< I >( ops_ ).evalTo( tmp );
            addInPlace< T, numMonomials( N, M ) >( out, tmp );
        }
    }
};

}  // namespace tax::detail
