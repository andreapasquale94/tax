#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/scheme.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/core/storage/sparse.hpp>
#include <tax/la/types.hpp>
#include <type_traits>
#include <utility>

namespace tax
{

// Primary template (forward declaration for partial specialisations).
template < typename T, typename Scheme, typename Storage = storage::Dense >
    requires IndexScheme< Scheme >
class TaylorExpansion;

// Dense specialisation

/// A truncated Taylor expansion over an index `Scheme` with dense storage.
template < typename T, typename Scheme >
    requires Scalar< T > && IndexScheme< Scheme >
class TaylorExpansion< T, Scheme, storage::Dense >
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
    [[nodiscard]] constexpr TaylorExpansion< T, IsotropicScheme< N2, Scheme::vars >,
                                             storage::Dense >
    truncate() const noexcept
        requires( is_isotropic_scheme_v< Scheme > && N2 >= 0 && N2 <= Scheme::order )
    {
        using Out = TaylorExpansion< T, IsotropicScheme< N2, Scheme::vars >, storage::Dense >;
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

// Convenience aliases  (dense)

/// `TE<N, M>` — order-N, M-variate dense `double` expansion. `M` defaults to 1.
template < int N, int M = 1 >
using TE = TaylorExpansion< double, IsotropicScheme< N, M >, storage::Dense >;

/// `TEn<N, M>` — explicit M-variate alias, same as `TE<N, M>`.
template < int N, int M >
using TEn = TaylorExpansion< double, IsotropicScheme< N, M >, storage::Dense >;

// Sparse specialisation

/// A truncated Taylor expansion in M variables of order N with sparse storage.
/// Sparse storage is isotropic-only (single-order graded-lex layout).
template < typename T, typename Scheme >
    requires Scalar< T > && IndexScheme< Scheme >
class TaylorExpansion< T, Scheme, storage::Sparse >
{
    static_assert( is_isotropic_scheme_v< Scheme >,
                   "Sparse TaylorExpansion is supported only for IsotropicScheme<N, M>" );

    static constexpr int N = Scheme::order;
    static constexpr int M = Scheme::vars;

   public:
    static_assert( N >= 0, "TaylorExpansion<Sparse> order must be non-negative" );
    static_assert( M >= 1, "TaylorExpansion<Sparse> variable count must be at least 1" );

    using scheme = Scheme;
    using scalar_type = T;
    using container_t = storage::SparseContainer< T, N, M >;
    using Input = std::array< T, std::size_t( M ) >;
    using Dense = TaylorExpansion< T, Scheme, storage::Dense >;

    static constexpr int order_v = N;
    static constexpr int vars_v = M;
    /// Dense-equivalent upper bound on the number of monomials.
    static constexpr std::size_t nCoefficients = numMonomials( N, M );

    constexpr TaylorExpansion() = default;

    /// Constant polynomial with value `c`.
    /*implicit*/ TaylorExpansion( T c )
    {
        if ( c != T{ 0 } ) c_.set( 0, c );
    }

    /// Lift a dense polynomial into sparse storage (drops exact zeros).
    explicit TaylorExpansion( const Dense& d )
    {
        for ( std::size_t k = 0; k < Dense::nCoefficients; ++k )
        {
            if ( d[k] != T{ 0 } ) c_.set( k, d[k] );
        }
    }

    [[nodiscard]] static TaylorExpansion zero() noexcept { return {}; }
    [[nodiscard]] static TaylorExpansion constant( T c ) { return TaylorExpansion{ c }; }

    /// Univariate variable: `x = x0 + 1*dx`.
    [[nodiscard]] static TaylorExpansion variable( T x0 ) noexcept
        requires( M == 1 )
    {
        TaylorExpansion r;
        if ( x0 != T{ 0 } ) r.c_.set( 0, x0 );
        if constexpr ( N >= 1 ) r.c_.set( 1, T{ 1 } );
        return r;
    }

    /// Coordinate variable `x_I` at expansion point `p` (compile-time index).
    template < int I >
    [[nodiscard]] static TaylorExpansion variable( const Input& p ) noexcept
        requires( M >= 1 && I >= 0 && I < M )
    {
        TaylorExpansion r;
        if ( p[std::size_t( I )] != T{ 0 } ) r.c_.set( 0, p[std::size_t( I )] );
        if constexpr ( N >= 1 )
        {
            MultiIndex< M > alpha{};
            alpha[std::size_t( I )] = 1;
            r.c_.set( flatIndex< M >( alpha ), T{ 1 } );
        }
        return r;
    }

    /// Number of currently stored nonzero monomials.
    [[nodiscard]] std::size_t nnz() const noexcept { return c_.nnz(); }

    /// Constant (zeroth) coefficient; returns 0 if the slot is absent.
    [[nodiscard]] T value() const noexcept { return c_.value(); }

    /// Runtime multi-index coefficient lookup (O(log nnz)).
    [[nodiscard]] T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return c_.coeffAtFlat( flatIndex< M >( alpha ) );
    }

    /// Compile-time multi-index coefficient lookup (O(log nnz)).
    template < int... Alpha >
    [[nodiscard]] T coeff() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ),
                       "coeff<Alpha...>(): arity must match variable count" );
        static_assert( ( ( Alpha >= 0 ) && ... ), "coeff<Alpha...>(): negative exponent" );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= N, "coeff<Alpha...>(): total degree exceeds N" );
        constexpr MultiIndex< M > a{ Alpha... };
        return c_.coeffAtFlat( flatIndex< M >( a ) );
    }

    // Derivative accessors (apply k! scaling to raw coefficients)

    /// Runtime partial derivative value `d^|alpha| f / dx^alpha` at x0.
    [[nodiscard]] T derivative( const MultiIndex< M >& alpha ) const noexcept
    {
        // Accumulate in T: std::size_t overflows at 21! (see dense variant).
        T fac = T{ 1 };
        for ( int i = 0; i < M; ++i )
            for ( int j = 1; j <= alpha[std::size_t( i )]; ++j ) fac *= T( j );
        return coeff( alpha ) * fac;
    }

    /// Compile-time partial derivative value.
    template < int... Alpha >
    [[nodiscard]] T derivative() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ) );
        static_assert( ( ( Alpha >= 0 ) && ... ) );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= N );

        // Accumulate in T to avoid std::size_t factorial overflow (21! > UINT64_MAX).
        constexpr auto factorial = []( int n ) constexpr noexcept -> T {
            T r = T{ 1 };
            for ( int i = 2; i <= n; ++i ) r *= T( i );
            return r;
        };
        constexpr T fac = ( factorial( Alpha ) * ... * T( 1 ) );
        return coeff< Alpha... >() * fac;
    }

    /// Read-only view of the sorted flat indices of all nonzero slots.
    [[nodiscard]] std::span< const storage::flat_index_t > support() const noexcept
    {
        return c_.support();
    }

    /// Read-only view of the coefficient values aligned with `support()`.
    [[nodiscard]] std::span< const T > values() const noexcept { return c_.values(); }

    /// Materialise a dense `TaylorExpansion<T, N, M, Dense>` from this sparse polynomial.
    [[nodiscard]] Dense dense() const noexcept
    {
        Dense r;
        c_.forEachNonzero( [&]( std::size_t k, T v ) { r[k] = v; } );
        return r;
    }

    /// Evaluate the polynomial at displacement `dx` from the expansion point.
    [[nodiscard]] T eval( const Input& dx ) const noexcept { return dense().eval( dx ); }

    /// Evaluate the polynomial at displacement given as an Eigen vector.
    template < typename DxDerived >
    [[nodiscard]] T eval( const Eigen::MatrixBase< DxDerived >& dx ) const
    {
        return dense().eval( dx );
    }

    /// Partial derivative polynomial with respect to variable `I`.
    template < int I >
    [[nodiscard]] TaylorExpansion deriv() const noexcept
        requires( I >= 0 && I < M )
    {
        return derivImpl( I );
    }

    /// Partial derivative polynomial with respect to variable `var`. Throws std::out_of_range if
    /// `var` is outside [0, M).
    [[nodiscard]] TaylorExpansion deriv( int var ) const
    {
        if ( var < 0 || var >= M )
            throw std::out_of_range( "tax::TaylorExpansion::deriv(var): var must be in [0, M)" );
        return derivImpl( var );
    }

    /// Indefinite integral polynomial with respect to variable `I`.
    template < int I >
    [[nodiscard]] TaylorExpansion integ() const noexcept
        requires( I >= 0 && I < M )
    {
        return integImpl( I );
    }

    /// Indefinite integral polynomial with respect to variable `var`. Throws std::out_of_range if
    /// `var` is outside [0, M).
    [[nodiscard]] TaylorExpansion integ( int var ) const
    {
        if ( var < 0 || var >= M )
            throw std::out_of_range( "tax::TaylorExpansion::integ(var): var must be in [0, M)" );
        return integImpl( var );
    }

    /// Compute the gradient vector `[df/dx_0, ..., df/dx_{M-1}]` at the expansion point.
    [[nodiscard]] tax::la::VecNT< M, T > gradient() const noexcept
    {
        tax::la::VecNT< M, T > g;
        MultiIndex< M > alpha{};
        for ( int i = 0; i < M; ++i )
        {
            alpha[std::size_t( i )] = 1;
            g( i ) = derivative( alpha );
            alpha[std::size_t( i )] = 0;
        }
        return g;
    }

    /// Compute the Hessian matrix `H(i,j) = d^2 f / (dx_i dx_j)` at the expansion point.
    [[nodiscard]] tax::la::MatNT< M, T > hessian() const noexcept
    {
        tax::la::MatNT< M, T > H;
        for ( int i = 0; i < M; ++i )
        {
            for ( int j = 0; j < M; ++j )
            {
                MultiIndex< M > alpha{};
                alpha[std::size_t( i )] += 1;
                alpha[std::size_t( j )] += 1;
                H( i, j ) = derivative( alpha );
            }
        }
        return H;
    }

    /// Order-reducing truncation: drop monomials of degree > N2, yielding a lower-order expansion.
    template < int N2 >
    [[nodiscard]] TaylorExpansion< T, IsotropicScheme< N2, M >, storage::Sparse > truncate()
        const noexcept
        requires( N2 >= 0 && N2 <= N )
    {
        return truncatedBelow< TaylorExpansion< T, IsotropicScheme< N2, M >, storage::Sparse > >(
            numMonomials( N2, M ) );
    }

    /// Same-order truncation: zero every coefficient of total degree > d (d>=N copies, d<0 zeroes).
    [[nodiscard]] TaylorExpansion truncate( int d ) const noexcept
    {
        if ( d >= N ) return *this;
        return truncatedBelow< TaylorExpansion >( d >= 0 ? numMonomials( d, M ) : 0 );
    }

    [[nodiscard]] const container_t& container() const noexcept { return c_; }
    [[nodiscard]] container_t& container() noexcept { return c_; }

   private:
    /// Shared body of the compile-time-index and runtime-index deriv overloads.
    /// Decrementing exponent `var` is injective on the support, so no two source
    /// monomials collide — `accumulate` keeps the result sorted and zero-free.
    [[nodiscard]] TaylorExpansion derivImpl( int var ) const noexcept
    {
        TaylorExpansion r;
        c_.forEachNonzero( [&]( std::size_t k, T v ) {
            auto alpha = unflatIndex< M >( k );
            const int e = alpha[std::size_t( var )];
            if ( e == 0 ) return;
            alpha[std::size_t( var )] = e - 1;
            r.c_.accumulate( flatIndex< M >( alpha ), v * T( e ) );
        } );
        return r;
    }

    /// Shared body of the compile-time-index and runtime-index integ overloads.
    [[nodiscard]] TaylorExpansion integImpl( int var ) const noexcept
    {
        TaylorExpansion r;
        c_.forEachNonzero( [&]( std::size_t k, T v ) {
            auto alpha = unflatIndex< M >( k );
            int deg = 0;
            for ( int i = 0; i < M; ++i ) deg += alpha[std::size_t( i )];
            if ( deg + 1 > N ) return;  // would exceed the kept set
            const int e = alpha[std::size_t( var )];
            alpha[std::size_t( var )] = e + 1;
            r.c_.accumulate( flatIndex< M >( alpha ), v / T( e + 1 ) );
        } );
        return r;
    }

    /// Copy the sorted prefix of stored slots with flat index < `limit` into a fresh result.
    template < typename Result >
    [[nodiscard]] Result truncatedBelow( std::size_t limit ) const noexcept
    {
        Result r;
        const auto sup = support();
        const auto vals = values();
        // Support is sorted ascending: the kept slots are the prefix below `limit`.
        const auto n = std::size_t(
            std::ranges::lower_bound( sup, storage::flat_index_t( limit ) ) - sup.begin() );
        r.container().rawIndices().assign( sup.begin(), sup.begin() + std::ptrdiff_t( n ) );
        r.container().rawValues().assign( vals.begin(), vals.begin() + std::ptrdiff_t( n ) );
        return r;
    }

    container_t c_;
};

// Convenience aliases  (sparse)

/// `STE<N>` — univariate sparse `double` expansion of order N.
/// `STE<N, M>` — M-variate sparse `double` expansion of order N.
template < int N, int M = 1 >
using STE = TaylorExpansion< double, IsotropicScheme< N, M >, storage::Sparse >;

/// `MixedTE<Groups...>` — an anisotropic (per-group order) dense `double` expansion.
template < typename... Groups >
using MixedTE = TaylorExpansion< double, MixedScheme< Groups... >, storage::Dense >;

// Free-function variable factories (unnamed, integer-indexed)

/// Univariate variable `x = x0 + 1*dx` of an order-`N` dense expansion.
template < int N, Scalar T = double >
[[nodiscard]] constexpr auto variable( T x0 ) noexcept
{
    return TaylorExpansion< T, IsotropicScheme< N, 1 > >::variable( x0 );
}

/// The `I`-th coordinate variable of an order-`N`, `M`-variate dense expansion at point `p`.
template < int I, int N, int M, Scalar T = double >
[[nodiscard]] constexpr auto variable( const std::array< T, std::size_t( M ) >& p ) noexcept
{
    return TaylorExpansion< T, IsotropicScheme< N, M > >::template variable< I >( p );
}

/// All `M` coordinate variables of an order-`N`, `M`-variate dense expansion at point `p`.
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

// Conversion helper: dense -> sparse

/// Convert a dense polynomial to sparse storage (drops exact zeros).
template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme, storage::Sparse > sparse(
    const TaylorExpansion< T, Scheme, storage::Dense >& d ) noexcept
{
    return TaylorExpansion< T, Scheme, storage::Sparse >( d );
}

}  // namespace tax
