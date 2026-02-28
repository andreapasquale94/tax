#pragma once

#include <tax/expr/base.hpp>
#include <tax/kernels.hpp>

namespace tax
{

/**
 * @brief Materialized truncated Taylor polynomial in `M` variables up to order `N`.
 * @tparam T Scalar coefficient type.
 * @tparam N Maximum total polynomial order.
 * @tparam M Number of variables.
 * @details Coefficients are stored in graded-lex order as defined by `flatIndex`.
 */
template < typename T, int N, int M = 1 >
class TDA : public DAExpr< TDA< T, N, M >, T, N, M >, public DALeaf
{
   public:
    static_assert( N >= 0, "DA order must be non-negative" );
    static_assert( M >= 1, "Number of variables must be at least 1" );

    /// @brief Number of stored coefficients.
    static constexpr std::size_t ncoef = detail::numMonomials( N, M );
    /// @brief Coefficient storage type.
    using coeff_array = std::array< T, ncoef >;
    /// @brief Expansion point type.
    using point_type = std::array< T, M >;

    // -- Constructors ---------------------------------------------------------

    /// @brief Construct zero polynomial.
    constexpr TDA() noexcept : c_{} {}
    /// @brief Construct from a full coefficient array.
    explicit constexpr TDA( coeff_array c ) noexcept : c_( std::move( c ) ) {}
    /// @brief Construct constant polynomial with value `val`.
    /*implicit*/ constexpr TDA( T val ) noexcept : c_{} { c_[0] = val; }

    /// @brief Materialize a compatible expression in one evaluation pass.
    template < typename Derived >
    /*implicit*/ constexpr TDA( const DAExpr< Derived, T, N, M >& expr ) noexcept : c_{}
    {
        expr.self().evalTo( c_ );
    }

    // -- Variable factories ---------------------------------------------------

    /**
     * @brief Create the univariate variable expanded at `x0`.
     * @details Produces `x0 + 1*dx` (truncated to order `N`).
     */
    [[nodiscard]] static constexpr TDA variable( T x0 ) noexcept
        requires( M == 1 )
    {
        coeff_array c{};
        c[0] = x0;
        if constexpr ( N >= 1 ) c[1] = T{ 1 };
        return TDA{ c };
    }

    template < int I >
    /**
     * @brief Create variable `x_I` expanded around point `x0`.
     * @details The constant term is `x0[I]` and the `e_I` coefficient is `1`.
     */
    [[nodiscard]] static constexpr TDA variable( const point_type& x0 ) noexcept
    {
        static_assert( I >= 0 && I < M, "Variable index out of range" );
        coeff_array c{};
        c[0] = x0[I];
        if constexpr ( N >= 1 )
        {
            MultiIndex< M > ei{};
            ei[I] = 1;
            c[detail::flatIndex< M >( ei )] = T{ 1 };
        }
        return TDA{ c };
    }

    /**
     * @brief Create all coordinate variables at expansion point `x0`.
     * @return Tuple `(x_0, ..., x_{M-1})` of DA variables.
     */
    [[nodiscard]] static constexpr auto variables( const point_type& x0 ) noexcept
    {
        return [&]< std::size_t... I >( std::index_sequence< I... > ) {
            return std::tuple{ variable< int( I ) >( x0 )... };
        }( std::make_index_sequence< std::size_t( M ) >{} );
    }

    /// @brief Create a constant polynomial with value `v`.
    [[nodiscard]] static constexpr TDA constant( T v ) noexcept { return TDA{ v }; }

    // -- evalTo / addTo / subTo -----------------------------------------------

    /**
     * @brief Copy coefficients into `out`.
     * @param out Destination coefficient array.
     */
    constexpr void evalTo( coeff_array& out ) const noexcept { out = c_; }

    /**
     * @brief Add coefficients into `out`.
     * @param out Destination updated as `out += coeffs()`.
     */
    constexpr void addTo( coeff_array& out ) const noexcept
    {
        detail::addInPlace< T, ncoef >( out, c_ );
    }

    /**
     * @brief Subtract coefficients from `out`.
     * @param out Destination updated as `out -= coeffs()`.
     */
    constexpr void subTo( coeff_array& out ) const noexcept
    {
        detail::subInPlace< T, ncoef >( out, c_ );
    }

    // -- Element access -------------------------------------------------------

    /// @brief Read-only coefficient access by flat index.
    [[nodiscard]] constexpr T operator[]( std::size_t i ) const noexcept { return c_[i]; }
    /// @brief Mutable coefficient access by flat index.
    [[nodiscard]] constexpr T& operator[]( std::size_t i ) noexcept { return c_[i]; }
    /// @brief Access full coefficient array.
    [[nodiscard]] constexpr const coeff_array& coeffs() const noexcept { return c_; }
    /// @brief Value at the expansion point.
    [[nodiscard]] constexpr T value() const noexcept { return c_[0]; }

    /**
     * @brief Coefficient associated with multi-index `alpha`.
     * @param alpha Monomial exponent vector.
     */
    [[nodiscard]] constexpr T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return c_[detail::flatIndex< M >( alpha )];
    }

    /**
     * @brief Partial derivative selected by `alpha` at expansion point.
     * @details Returns `coeff(alpha) * prod_i alpha[i]!`.
     */
    [[nodiscard]] constexpr T derivative( const MultiIndex< M >& alpha ) const noexcept
    {
        const auto id = detail::flatIndex< M >( alpha );
        const auto factor = derivative_factors_[id];
        return c_[id] * T( factor );
    }

    /**
     * @brief All partial derivatives in flat monomial order.
     * @details Entry `i` equals `coeff[i] * prod_j alpha_j!` for the monomial at `i`.
     */
    [[nodiscard]] constexpr coeff_array derivatives() const noexcept
    {
        coeff_array out{};
        for ( std::size_t i = 0; i < ncoef; ++i ) out[i] = c_[i] * derivative_factors_[i];
        return out;
    }

    // -- In-place operators ---------------------------------------------------

    /// @brief In-place polynomial addition.
    constexpr TDA& operator+=( const TDA& o ) noexcept
    {
        detail::addInPlace< T, ncoef >( c_, o.c_ );
        return *this;
    }
    /// @brief In-place polynomial subtraction.
    constexpr TDA& operator-=( const TDA& o ) noexcept
    {
        detail::subInPlace< T, ncoef >( c_, o.c_ );
        return *this;
    }
    template < typename Derived >
    /// @brief In-place addition from an expression node.
    constexpr TDA& operator+=( const DAExpr< Derived, T, N, M >& e ) noexcept
    {
        coeff_array t{};
        e.self().evalTo( t );
        detail::addInPlace< T, ncoef >( c_, t );
        return *this;
    }
    template < typename Derived >
    /// @brief In-place subtraction from an expression node.
    constexpr TDA& operator-=( const DAExpr< Derived, T, N, M >& e ) noexcept
    {
        coeff_array t{};
        e.self().evalTo( t );
        detail::subInPlace< T, ncoef >( c_, t );
        return *this;
    }
    /// @brief In-place scalar multiplication.
    constexpr TDA& operator*=( T s ) noexcept
    {
        detail::scaleInPlace< T, ncoef >( c_, s );
        return *this;
    }
    /// @brief In-place scalar division.
    constexpr TDA& operator/=( T s ) noexcept
    {
        detail::scaleInPlace< T, ncoef >( c_, T{ 1 } / s );
        return *this;
    }

   private:
    [[nodiscard]] static constexpr coeff_array makeDerivativeFactors() noexcept
    {
        coeff_array factors{};
        MultiIndex< M > alpha{};

        auto fillAlpha = [&]( auto& self, int var, int rem ) constexpr -> void {
            if ( var == M - 1 )
            {
                alpha[var] = rem;
                std::size_t fac = 1;
                for ( int i = 0; i < M; ++i )
                    for ( int j = 1; j <= alpha[i]; ++j ) fac *= std::size_t( j );
                factors[detail::flatIndex< M >( alpha )] = T( fac );
                return;
            }
            for ( int k = rem; k >= 0; --k )
            {
                alpha[var] = k;
                self( self, var + 1, rem - k );
            }
        };

        for ( int d = 0; d <= N; ++d ) fillAlpha( fillAlpha, 0, d );
        return factors;
    }

    inline static constexpr coeff_array derivative_factors_ = makeDerivativeFactors();
    coeff_array c_;
};

/// @brief Univariate DA alias (`double`, order `N`, one variable).
template < int N >
using DA = TDA< double, N, 1 >;

/// @brief Multivariate DA alias (`double`, order `N`, `M` variables).
template < int N, int M >
using DAn = TDA< double, N, M >;

}  // namespace tax
