#pragma once

#include <tax/expr/base.hpp>
#include <tax/kernels.hpp>
#include <tax/utils/enumeration.hpp>
#include <tax/utils/streaming.hpp>
#include <ostream>
#include <span>
#include <type_traits>

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
class TruncatedTaylorExpansionT : public Expr< TruncatedTaylorExpansionT< T, N, M >, T, N, M >, public ExpansionLeaf
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
    constexpr TruncatedTaylorExpansionT() noexcept : c_{} {}
    /// @brief Construct from a full coefficient array.
    explicit constexpr TruncatedTaylorExpansionT( coeff_array c ) noexcept : c_( std::move( c ) ) {}
    /// @brief Construct constant polynomial with value `val`.
    /*implicit*/ constexpr TruncatedTaylorExpansionT( T val ) noexcept : c_{} { c_[0] = val; }

    /// @brief Materialize a compatible expression in one evaluation pass.
    template < typename Derived >
    /*implicit*/ constexpr TruncatedTaylorExpansionT( const Expr< Derived, T, N, M >& expr ) noexcept : c_{}
    {
        expr.self().evalTo( c_ );
    }

    // -- Variable factories ---------------------------------------------------

    /**
     * @brief Create the univariate variable expanded at `x0`.
     * @details Produces `x0 + 1*dx` (truncated to order `N`).
     */
    [[nodiscard]] static constexpr TruncatedTaylorExpansionT variable( T x0 ) noexcept
        requires( M == 1 )
    {
        coeff_array c{};
        c[0] = x0;
        if constexpr ( N >= 1 ) c[1] = T{ 1 };
        return TruncatedTaylorExpansionT{ c };
    }

    /**
     * @brief Create variable `x_I` expanded around point `x0`.
     * @details The constant term is `x0[I]` and the `e_I` coefficient is `1`.
     */
    template < int I >
    [[nodiscard]] static constexpr TruncatedTaylorExpansionT variable( const point_type& x0 ) noexcept
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
        return TruncatedTaylorExpansionT{ c };
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
    [[nodiscard]] static constexpr TruncatedTaylorExpansionT constant( T v ) noexcept { return TruncatedTaylorExpansionT{ v }; }

    /// @brief Create a constant polynomial with value `0`.
    [[nodiscard]] static constexpr TruncatedTaylorExpansionT zero() noexcept { return TruncatedTaylorExpansionT{ 0 }; }

    /// @brief Create a constant polynomial with value `1`.
    [[nodiscard]] static constexpr TruncatedTaylorExpansionT one() noexcept { return TruncatedTaylorExpansionT{ 1 }; }

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
    /// @brief Read-only span over the coefficient storage.
    [[nodiscard]] constexpr std::span< const T > coeffSpan() const noexcept
    {
        return { c_.data(), ncoef };
    }
    /// @brief Mutable span over the coefficient storage.
    [[nodiscard]] constexpr std::span< T > coeffSpan() noexcept { return { c_.data(), ncoef }; }
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
     * @brief Coefficient selected by compile-time multi-index.
     * @tparam Alpha Monomial exponents for each variable (size must be `M`).
     */
    template < int... Alpha >
    [[nodiscard]] constexpr T coeff() const noexcept
    {
        static_assert( sizeof...( Alpha ) == M,
                       "Coefficient multi-index arity must match number of variables" );
        static_assert( ( ( Alpha >= 0 ) && ... ), "Coefficient exponents must be non-negative" );
        constexpr int total_order = ( Alpha + ... + 0 );
        static_assert( total_order <= N,
                       "Coefficient multi-index total order exceeds DA truncation order" );
        constexpr MultiIndex< M > alpha{ Alpha... };
        return coeff( alpha );
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
     * @brief Partial derivative selected by compile-time multi-index.
     * @tparam Alpha Derivative orders for each variable (size must be `M`).
     */
    template < int... Alpha >
    [[nodiscard]] constexpr T derivative() const noexcept
    {
        static_assert( sizeof...( Alpha ) == M,
                       "Derivative multi-index arity must match number of variables" );
        static_assert( ( ( Alpha >= 0 ) && ... ), "Derivative orders must be non-negative" );
        constexpr int total_order = ( Alpha + ... + 0 );
        static_assert( total_order <= N, "Derivative total order exceeds DA truncation order" );
        constexpr MultiIndex< M > alpha{ Alpha... };
        return derivative( alpha );
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

    // -- Evaluation -----------------------------------------------------------

    /**
     * @brief Evaluate the polynomial at displacement `dx` from expansion point.
     * @param dx Displacement value (univariate).
     * @return f(x0 + dx) truncated to order N.
     */
    [[nodiscard]] constexpr T eval( T dx ) const noexcept
        requires( M == 1 )
    {
        // Horner's method: c[N]*dx + c[N-1] ... *dx + c[0]
        T result = c_[N];
        for ( int i = N - 1; i >= 0; --i ) result = result * dx + c_[i];
        return result;
    }

    /**
     * @brief Evaluate the polynomial at displacement `dx` from expansion point.
     * @param dx Displacement vector (multivariate).
     * @return f(x0 + dx) truncated to order N.
     */
    [[nodiscard]] constexpr T eval( const point_type& dx ) const noexcept
    {
        if constexpr ( M == 1 )
        {
            return eval( dx[0] );
        } else
        {
            T result{};
            MultiIndex< M > alpha{};

            auto accumulate = [&]( auto& self, int var, int rem ) constexpr -> void {
                if ( var == M - 1 )
                {
                    alpha[var] = rem;
                    // Compute dx^alpha = product of dx[i]^alpha[i]
                    T monomial{ 1 };
                    for ( int i = 0; i < M; ++i )
                        for ( int j = 0; j < alpha[i]; ++j ) monomial *= dx[i];
                    result += c_[detail::flatIndex< M >( alpha )] * monomial;
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

    // -- In-place operators ---------------------------------------------------

    /// @brief In-place polynomial addition.
    constexpr TruncatedTaylorExpansionT& operator+=( const TruncatedTaylorExpansionT& o ) noexcept
    {
        detail::addInPlace< T, ncoef >( c_, o.c_ );
        return *this;
    }
    /// @brief In-place polynomial subtraction.
    constexpr TruncatedTaylorExpansionT& operator-=( const TruncatedTaylorExpansionT& o ) noexcept
    {
        detail::subInPlace< T, ncoef >( c_, o.c_ );
        return *this;
    }
    template < typename Derived >
    /// @brief In-place addition from an expression node.
    constexpr TruncatedTaylorExpansionT& operator+=( const Expr< Derived, T, N, M >& e ) noexcept
    {
        coeff_array t{};
        e.self().evalTo( t );
        detail::addInPlace< T, ncoef >( c_, t );
        return *this;
    }
    template < typename Derived >
    /// @brief In-place subtraction from an expression node.
    constexpr TruncatedTaylorExpansionT& operator-=( const Expr< Derived, T, N, M >& e ) noexcept
    {
        coeff_array t{};
        e.self().evalTo( t );
        detail::subInPlace< T, ncoef >( c_, t );
        return *this;
    }
    /// @brief In-place scalar multiplication.
    constexpr TruncatedTaylorExpansionT& operator*=( T s ) noexcept
    {
        detail::scaleInPlace< T, ncoef >( c_, s );
        return *this;
    }
    /// @brief In-place polynomial multiplication (Cauchy product).
    constexpr TruncatedTaylorExpansionT& operator*=( const TruncatedTaylorExpansionT& o ) noexcept
    {
        coeff_array tmp{};
        detail::cauchyProduct< T, N, M >( tmp, c_, o.c_ );
        c_ = tmp;
        return *this;
    }
    /// @brief In-place scalar division.
    constexpr TruncatedTaylorExpansionT& operator/=( T s ) noexcept
    {
        detail::scaleInPlace< T, ncoef >( c_, T{ 1 } / s );
        return *this;
    }
    /// @brief In-place polynomial division.
    constexpr TruncatedTaylorExpansionT& operator/=( const TruncatedTaylorExpansionT& o ) noexcept
    {
        coeff_array rec{};
        detail::seriesReciprocal< T, N, M >( rec, o.c_ );
        coeff_array tmp{};
        detail::cauchyProduct< T, N, M >( tmp, c_, rec );
        c_ = tmp;
        return *this;
    }

    // -- Comparison operators (on constant term) --------------------------------

    [[nodiscard]] friend constexpr bool operator==( const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b ) noexcept
    {
        return a.value() == b.value();
    }
    [[nodiscard]] friend constexpr bool operator!=( const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b ) noexcept
    {
        return a.value() != b.value();
    }
    [[nodiscard]] friend constexpr bool operator<( const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b ) noexcept
    {
        return a.value() < b.value();
    }
    [[nodiscard]] friend constexpr bool operator>( const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b ) noexcept
    {
        return a.value() > b.value();
    }
    [[nodiscard]] friend constexpr bool operator<=( const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b ) noexcept
    {
        return a.value() <= b.value();
    }
    [[nodiscard]] friend constexpr bool operator>=( const TruncatedTaylorExpansionT& a, const TruncatedTaylorExpansionT& b ) noexcept
    {
        return a.value() >= b.value();
    }

    // DA vs scalar (T)
    [[nodiscard]] friend constexpr bool operator==( const TruncatedTaylorExpansionT& a, const T& s ) noexcept
    {
        return a.value() == s;
    }
    [[nodiscard]] friend constexpr bool operator!=( const TruncatedTaylorExpansionT& a, const T& s ) noexcept
    {
        return a.value() != s;
    }
    [[nodiscard]] friend constexpr bool operator<( const TruncatedTaylorExpansionT& a, const T& s ) noexcept
    {
        return a.value() < s;
    }
    [[nodiscard]] friend constexpr bool operator>( const TruncatedTaylorExpansionT& a, const T& s ) noexcept
    {
        return a.value() > s;
    }
    [[nodiscard]] friend constexpr bool operator<=( const TruncatedTaylorExpansionT& a, const T& s ) noexcept
    {
        return a.value() <= s;
    }
    [[nodiscard]] friend constexpr bool operator>=( const TruncatedTaylorExpansionT& a, const T& s ) noexcept
    {
        return a.value() >= s;
    }

    // scalar (T) vs DA
    [[nodiscard]] friend constexpr bool operator==( const T& s, const TruncatedTaylorExpansionT& a ) noexcept
    {
        return s == a.value();
    }
    [[nodiscard]] friend constexpr bool operator!=( const T& s, const TruncatedTaylorExpansionT& a ) noexcept
    {
        return s != a.value();
    }
    [[nodiscard]] friend constexpr bool operator<( const T& s, const TruncatedTaylorExpansionT& a ) noexcept
    {
        return s < a.value();
    }
    [[nodiscard]] friend constexpr bool operator>( const T& s, const TruncatedTaylorExpansionT& a ) noexcept
    {
        return s > a.value();
    }
    [[nodiscard]] friend constexpr bool operator<=( const T& s, const TruncatedTaylorExpansionT& a ) noexcept
    {
        return s <= a.value();
    }
    [[nodiscard]] friend constexpr bool operator>=( const T& s, const TruncatedTaylorExpansionT& a ) noexcept
    {
        return s >= a.value();
    }

    /// @brief Stream as polynomial in `dx`, using superscripts for powers and subscripts for variable indices.
    friend std::ostream& operator<<( std::ostream& os, const TruncatedTaylorExpansionT& a )
    {
        bool write = false;
        for ( int d = 0; d <= N; ++d )
        {
            detail::forEachMonomial< M >( d, [&]( const MultiIndex< M >& alpha, std::size_t ai ) {
                const T coeff = a.c_[ai];
                if ( coeff == T{} ) return;

                const bool has_monomial = ( d > 0 );
                if constexpr ( std::is_arithmetic_v< T > )
                {
                    const bool neg = coeff < T{};
                    const T abs_coeff = neg ? -coeff : coeff;
                    const bool print_coeff = !has_monomial || abs_coeff != T{ 1 };

                    if ( write )
                    {
                        os << ( neg ? " - " : " + " );
                    } else if ( neg )
                    {
                        os << '-';
                    }

                    if ( print_coeff ) os << abs_coeff;
                    if ( has_monomial )
                    {
                        if ( print_coeff ) os << "·";
                        detail::writeMonomial< M >( os, alpha );
                    }
                } else
                {
                    if ( write ) os << " + ";
                    os << coeff;
                    if ( has_monomial )
                    {
                        os << '*';
                        detail::writeMonomial< M >( os, alpha );
                    }
                }

                write = true;
            } );
        }

        if ( !write ) os << T{};
        os << " + ";
        detail::writeTruncationRemainder< M >( os, N + 1 );
        return os;
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

/// @brief Univariate TE alias (`double`, order `N`, one variable).
template < int N >
using TE = TruncatedTaylorExpansionT< double, N, 1 >;

/// @brief Multivariate TEn alias (`double`, order `N`, `M` variables).
template < int N, int M >
using TEn = TruncatedTaylorExpansionT< double, N, M >;

}  // namespace tax
