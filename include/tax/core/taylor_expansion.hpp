#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/scheme.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/la/types.hpp>
#include <type_traits>
#include <utility>

namespace tax
{

/// A truncated Taylor expansion over an index `Scheme`, with dense storage.
template < typename T, typename Scheme >
    requires Scalar< T > && IndexScheme< Scheme >
class TaylorExpansion
{
   public:
    static_assert( Scheme::order >= 0, "TaylorExpansion order must be non-negative" );
    static_assert( Scheme::vars >= 1, "TaylorExpansion variable count must be at least 1" );

    using scheme = Scheme;
    using scalar_type = T;
    using container_t = storage::DenseContainer< T, Scheme::nCoeff >;
    using Input = std::array< T, std::size_t( Scheme::vars ) >;
    using Data = std::array< T, Scheme::nCoeff >;

    static constexpr int order_v = Scheme::order;
    static constexpr int vars_v = Scheme::vars;
    static constexpr std::size_t nCoefficients = Scheme::nCoeff;

    constexpr TaylorExpansion() noexcept = default;

    /// Constant expansion: value `val`, all higher-order coefficients zero.
    /*implicit*/ constexpr TaylorExpansion( T val ) noexcept { c_.set( 0, val ); }

    explicit constexpr TaylorExpansion( Data c ) noexcept : c_{ c } {}

    [[nodiscard]] static constexpr TaylorExpansion zero() noexcept { return {}; }

    [[nodiscard]] static constexpr TaylorExpansion constant( T v ) noexcept
    {
        return TaylorExpansion{ v };
    }

    /// Univariate variable: `x = x0 + 1*dx`.
    [[nodiscard]] static constexpr TaylorExpansion variable( T x0 ) noexcept
        requires( Scheme::isUnivariate )
    {
        TaylorExpansion r{ x0 };
        if constexpr ( Scheme::order >= 1 ) r.c_.set( 1, T{ 1 } );
        return r;
    }

    /// Multivariate variable: the I-th coordinate variable at point `p`.
    template < int I >
    [[nodiscard]] static constexpr TaylorExpansion variable( const Input& p ) noexcept
        requires( I >= 0 && I < Scheme::vars )
    {
        TaylorExpansion r{};
        r.c_.set( 0, p[std::size_t( I )] );
        if constexpr ( Scheme::order >= 1 )
        {
            constexpr MultiIndex< Scheme::vars > alpha = [] {
                MultiIndex< Scheme::vars > a{};
                a[std::size_t( I )] = 1;
                return a;
            }();
            static_assert( Scheme::flatOf( alpha ) != Scheme::kNotInBox,
                           "variable: coordinate's group has order 0" );
            r.c_.set( Scheme::flatOf( alpha ), T{ 1 } );
        }
        return r;
    }

    /// Runtime-indexed coordinate variable: `x_i = x0 + 1*dx_i`.
    [[nodiscard]] static TaylorExpansion variable( T x0, int var_idx )
    {
        if ( var_idx < 0 || var_idx >= Scheme::vars )
            throw std::out_of_range( "variable(): var_idx out of range" );
        TaylorExpansion r{};
        r.c_.set( 0, x0 );
        if constexpr ( Scheme::order >= 1 )
        {
            MultiIndex< Scheme::vars > alpha{};
            alpha[std::size_t( var_idx )] = 1;
            const std::size_t k = Scheme::flatOf( alpha );
            if ( k != Scheme::kNotInBox ) r.c_.set( k, T{ 1 } );
        }
        return r;
    }

    /// Constant (zeroth) coefficient, i.e. f(x0).
    [[nodiscard]] constexpr T value() const noexcept { return c_.value(); }

    [[nodiscard]] constexpr T operator[]( std::size_t k ) const noexcept { return c_[k]; }
    [[nodiscard]] constexpr T& operator[]( std::size_t k ) noexcept { return c_[k]; }

    /// Runtime multi-index coefficient lookup.
    [[nodiscard]] constexpr T coeff( const MultiIndex< Scheme::vars >& alpha ) const noexcept
    {
        const std::size_t k = Scheme::flatOf( alpha );
        return k == Scheme::kNotInBox ? T{} : c_[k];
    }

    /// Compile-time multi-index coefficient lookup.
    template < int... Alpha >
    [[nodiscard]] constexpr T coeff() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( Scheme::vars ),
                       "coeff<Alpha...>(): arity must match variable count" );
        static_assert( ( ( Alpha >= 0 ) && ... ), "coeff<Alpha...>(): negative exponent" );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= Scheme::order, "coeff<Alpha...>(): total degree exceeds N" );
        constexpr MultiIndex< Scheme::vars > a{ Alpha... };
        constexpr std::size_t k = Scheme::flatOf( a );
        if constexpr ( k == Scheme::kNotInBox )
            return T{};
        else
            return c_[k];
    }

    // Derivative accessors (apply k! scaling to raw coefficients)

    /// Runtime partial derivative value `d^|alpha| f / dx^alpha` at x0.
    [[nodiscard]] constexpr T derivative( const MultiIndex< Scheme::vars >& alpha ) const noexcept
    {
        // Accumulate the factorial in T: std::size_t overflows at 21! on 64-bit,
        // silently corrupting high-order derivatives.
        T fac = T{ 1 };
        for ( int i = 0; i < Scheme::vars; ++i )
            for ( int j = 1; j <= alpha[std::size_t( i )]; ++j ) fac *= T( j );
        return coeff( alpha ) * fac;
    }

    /// Compile-time partial derivative value.
    template < int... Alpha >
    [[nodiscard]] constexpr T derivative() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( Scheme::vars ),
                       "derivative<Alpha...>(): arity must match variable count" );
        static_assert( ( ( Alpha >= 0 ) && ... ), "derivative<Alpha...>(): negative exponent" );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= Scheme::order, "derivative<Alpha...>(): total degree exceeds N" );

        // Accumulate in T to avoid std::size_t factorial overflow (21! > UINT64_MAX).
        constexpr auto factorial = []( int n ) constexpr noexcept -> T {
            T r = T{ 1 };
            for ( int i = 2; i <= n; ++i ) r *= T( i );
            return r;
        };
        constexpr T fac = ( factorial( Alpha ) * ... * T( 1 ) );
        return coeff< Alpha... >() * fac;
    }

    /// Evaluate the polynomial at displacement `dx` from the expansion point.
    [[nodiscard]] constexpr T eval( const Input& dx ) const noexcept
    {
        constexpr int N = Scheme::order;
        constexpr int M = Scheme::vars;
        if constexpr ( Scheme::isUnivariate )
        {
            // Horner's method
            T result = c_[std::size_t( N )];
            for ( int i = N - 1; i >= 0; --i ) result = result * dx[0] + c_[std::size_t( i )];
            return result;
        } else
        {
            T result{};

            // Power table pw[i][j] = dx_i^j: each monomial then costs one
            // multiply via the partial product carried down the recursion,
            // instead of rebuilding dx^alpha from |alpha| factors.
            std::array< std::array< T, std::size_t( N ) + 1 >, std::size_t( M ) > pw{};
            for ( int i = 0; i < M; ++i )
            {
                pw[std::size_t( i )][0] = T{ 1 };
                for ( int j = 1; j <= N; ++j )
                    pw[std::size_t( i )][std::size_t( j )] =
                        pw[std::size_t( i )][std::size_t( j - 1 )] * dx[std::size_t( i )];
            }

            // Degree-by-degree accumulation: enumerate all monomials of total degree d
            // and accumulate c_alpha * dx^alpha. Indices outside the kept set
            // (kNotInBox) contribute nothing.
            auto accumulate = [&]( auto& self, int var, int rem, MultiIndex< M > alpha,
                                   T partial ) constexpr -> void {
                if ( var == M - 1 )
                {
                    alpha[std::size_t( var )] = rem;
                    const std::size_t k = Scheme::flatOf( alpha );
                    if ( k != Scheme::kNotInBox )
                        result += c_[k] * partial * pw[std::size_t( var )][std::size_t( rem )];
                    return;
                }
                for ( int k = rem; k >= 0; --k )
                {
                    auto a2 = alpha;
                    a2[std::size_t( var )] = k;
                    self( self, var + 1, rem - k, a2,
                          partial * pw[std::size_t( var )][std::size_t( k )] );
                }
            };

            for ( int d = 0; d <= N; ++d )
                accumulate( accumulate, 0, d, MultiIndex< M >{}, T{ 1 } );
            return result;
        }
    }

    /// Evaluate the polynomial at displacement given as an Eigen vector.
    template < typename DxDerived >
    [[nodiscard]] T eval( const Eigen::MatrixBase< DxDerived >& dx ) const
    {
        static_assert( DxDerived::SizeAtCompileTime == Scheme::vars ||
                           DxDerived::SizeAtCompileTime == Eigen::Dynamic,
                       "eval(Eigen): size must match number of variables M" );
        Input p{};
        for ( int i = 0; i < Scheme::vars; ++i ) p[std::size_t( i )] = T( dx( i ) );
        return eval( p );
    }

    /// Partial derivative polynomial with respect to variable `I`.
    template < int I >
    [[nodiscard]] constexpr TaylorExpansion deriv() const noexcept
        requires( I >= 0 && I < Scheme::vars )
    {
        return derivImpl( I );
    }

    /// Partial derivative polynomial with respect to variable `var`. Throws std::out_of_range if
    /// `var` is outside [0, M).
    [[nodiscard]] constexpr TaylorExpansion deriv( int var ) const
    {
        if ( var < 0 || var >= Scheme::vars )
            throw std::out_of_range( "tax::TaylorExpansion::deriv(var): var must be in [0, M)" );
        return derivImpl( var );
    }

    /// Indefinite integral polynomial with respect to variable `I`.
    template < int I >
    [[nodiscard]] constexpr TaylorExpansion integ() const noexcept
        requires( I >= 0 && I < Scheme::vars )
    {
        return integImpl( I );
    }

    /// Indefinite integral polynomial with respect to variable `var`. Throws std::out_of_range if
    /// `var` is outside [0, M).
    [[nodiscard]] constexpr TaylorExpansion integ( int var ) const
    {
        if ( var < 0 || var >= Scheme::vars )
            throw std::out_of_range( "tax::TaylorExpansion::integ(var): var must be in [0, M)" );
        return integImpl( var );
    }

    /// Order-reducing truncation: drop monomials of degree > N2, yielding a lower-order expansion.
    /// Isotropic-only: order reduction is defined for the single-order graded-lex layout.
    template < int N2 >
    [[nodiscard]] constexpr TaylorExpansion< T, IsotropicScheme< N2, Scheme::vars > > truncate()
        const noexcept
        requires( is_isotropic_scheme_v< Scheme > && N2 >= 0 && N2 <= Scheme::order )
    {
        using Out = TaylorExpansion< T, IsotropicScheme< N2, Scheme::vars > >;
        typename Out::Data out{};
        // Graded-lex: degree-<=N2 monomials are a shared prefix of the order-N layout.
        for ( std::size_t k = 0; k < numMonomials( N2, Scheme::vars ); ++k ) out[k] = c_[k];
        return Out{ out };
    }

    /// Same-order truncation: zero every coefficient of total degree > d (d>=N copies, d<0 zeroes).
    /// Isotropic-only: relies on the contiguous degree-block layout of the graded-lex order.
    [[nodiscard]] constexpr TaylorExpansion truncate( int d ) const noexcept
        requires( is_isotropic_scheme_v< Scheme > )
    {
        if ( d >= Scheme::order ) return *this;
        Data out{};
        if ( d >= 0 )
            for ( std::size_t k = 0; k < numMonomials( d, Scheme::vars ); ++k ) out[k] = c_[k];
        return TaylorExpansion{ out };
    }

    // Gradient and Hessian (require Eigen/Core, already included above)

    /// Compute the gradient vector `[df/dx_0, ..., df/dx_{M-1}]` at the expansion point.
    [[nodiscard]] tax::la::VecNT< Scheme::vars, T > gradient() const noexcept
    {
        tax::la::VecNT< Scheme::vars, T > g;
        MultiIndex< Scheme::vars > alpha{};
        for ( int i = 0; i < Scheme::vars; ++i )
        {
            alpha[std::size_t( i )] = 1;
            g( i ) = derivative( alpha );
            alpha[std::size_t( i )] = 0;
        }
        return g;
    }

    /// Compute the Hessian matrix `H(i,j) = d^2 f / (dx_i dx_j)` at the expansion point.
    [[nodiscard]] tax::la::MatNT< Scheme::vars, T > hessian() const noexcept
    {
        tax::la::MatNT< Scheme::vars, T > H;
        for ( int i = 0; i < Scheme::vars; ++i )
        {
            for ( int j = 0; j < Scheme::vars; ++j )
            {
                MultiIndex< Scheme::vars > alpha{};
                alpha[std::size_t( i )] += 1;
                alpha[std::size_t( j )] += 1;
                H( i, j ) = derivative( alpha );
            }
        }
        return H;
    }

    [[nodiscard]] constexpr const container_t& container() const noexcept { return c_; }
    [[nodiscard]] constexpr container_t& container() noexcept { return c_; }

    /// Raw coefficient array — convenience accessor used by kernels.
    [[nodiscard]] constexpr const Data& coefficients() const noexcept { return c_.data; }
    [[nodiscard]] constexpr Data& coefficients() noexcept { return c_.data; }

   private:
    /// Shared body of the compile-time-index and runtime-index deriv overloads.
    [[nodiscard]] constexpr TaylorExpansion derivImpl( int var ) const noexcept
    {
        Data out{};
        for ( std::size_t i = 0; i < nCoefficients; ++i )
        {
            if ( c_[i] == T{} ) continue;
            auto alpha = Scheme::multiOf( i );
            const int exp = alpha[std::size_t( var )];
            if ( exp == 0 ) continue;
            alpha[std::size_t( var )] = exp - 1;
            out[Scheme::flatOf( alpha )] += c_[i] * T( exp );
        }
        return TaylorExpansion{ out };
    }

    /// Shared body of the compile-time-index and runtime-index integ overloads.
    [[nodiscard]] constexpr TaylorExpansion integImpl( int var ) const noexcept
    {
        Data out{};
        for ( std::size_t i = 0; i < nCoefficients; ++i )
        {
            if ( c_[i] == T{} ) continue;
            auto alpha = Scheme::multiOf( i );
            const int exp = alpha[std::size_t( var )];
            alpha[std::size_t( var )] = exp + 1;
            const std::size_t k = Scheme::flatOf( alpha );
            if ( k == Scheme::kNotInBox ) continue;  // would exceed the kept set
            out[k] = c_[i] / T( exp + 1 );
        }
        return TaylorExpansion{ out };
    }

    container_t c_{};
};

// Convenience aliases

/// `TE<N, M>` — order-N, M-variate `double` expansion. `M` defaults to 1.
template < int N, int M = 1 >
using TE = TaylorExpansion< double, IsotropicScheme< N, M > >;

/// `TEn<N, M>` — explicit M-variate alias, same as `TE<N, M>`.
template < int N, int M >
using TEn = TaylorExpansion< double, IsotropicScheme< N, M > >;

/// `MixedTE<Groups...>` — an anisotropic (per-group order) `double` expansion.
template < typename... Groups >
using MixedTE = TaylorExpansion< double, MixedScheme< Groups... > >;

// Free-function variable factories (unnamed, integer-indexed)

/// Univariate variable `x = x0 + 1*dx` of an order-`N` expansion.
template < int N, Scalar T = double >
[[nodiscard]] constexpr auto variable( T x0 ) noexcept
{
    return TaylorExpansion< T, IsotropicScheme< N, 1 > >::variable( x0 );
}

/// The `I`-th coordinate variable of an order-`N`, `M`-variate expansion at point `p`.
template < int I, int N, int M, Scalar T = double >
[[nodiscard]] constexpr auto variable( const std::array< T, std::size_t( M ) >& p ) noexcept
{
    return TaylorExpansion< T, IsotropicScheme< N, M > >::template variable< I >( p );
}

/// All `M` coordinate variables of an order-`N`, `M`-variate expansion at point `p`.
template < int N, int M, Scalar T = double >
[[nodiscard]] constexpr auto variables( const std::array< T, std::size_t( M ) >& p ) noexcept
{
    using E = TaylorExpansion< T, IsotropicScheme< N, M > >;
    std::array< E, std::size_t( M ) > out{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out[I] = E::template variable< int( I ) >( p ) ), ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );
    return out;
}

}  // namespace tax
