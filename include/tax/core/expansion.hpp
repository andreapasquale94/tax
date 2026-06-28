#pragma once

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
#include <tax/series/basis.hpp>
#include <tax/series/taylor_basis.hpp>
#include <type_traits>
#include <utility>

namespace tax
{

// Forward declaration so the public `TE` alias can name the batched coefficient
// type without including <tax/core/batch.hpp> (which includes this header).
template < typename T, int K >
struct Batch;

// ---------------------------------------------------------------------------
// Expansion<T, Basis, Scheme, Storage> — a truncated expansion in a chosen
// polynomial basis over an index scheme. `TaylorExpansion` is the TaylorBasis
// instance (see the alias below); the same class carries every other basis.
// ---------------------------------------------------------------------------

// Primary template (forward declaration for partial specialisations).
template < typename T, typename Basis, typename Scheme, typename Storage = storage::Dense >
    requires Scalar< T > && tax::Basis< Basis > && IndexScheme< Scheme >
class Expansion;

// ---------------------------------------------------------------------------
// Dense specialisation
// ---------------------------------------------------------------------------

/// A truncated expansion over an index `Scheme` with dense storage.
template < typename T, typename Basis, typename Scheme >
    requires Scalar< T > && tax::Basis< Basis > && IndexScheme< Scheme >
class Expansion< T, Basis, Scheme, storage::Dense >
{
   public:
    static_assert( Scheme::order >= 0, "Expansion order must be non-negative" );
    static_assert( Scheme::vars >= 1, "Expansion variable count must be at least 1" );

    // ------------------------------------------------------------------
    // Associated types
    // ------------------------------------------------------------------
    using basis = Basis;
    using scheme = Scheme;
    using scalar_type = T;
    using container_t = storage::DenseContainer< T, Scheme::nCoeff >;
    using Input = std::array< T, std::size_t( Scheme::vars ) >;
    using Data = std::array< T, Scheme::nCoeff >;

    // ------------------------------------------------------------------
    // Compile-time properties
    // ------------------------------------------------------------------
    static constexpr int order_v = Scheme::order;
    static constexpr int vars_v = Scheme::vars;
    static constexpr std::size_t nCoefficients = Scheme::nCoeff;

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Zero-initialise all coefficients.
    constexpr Expansion() noexcept = default;

    /// Constant expansion: value `val`, all higher-order coefficients zero.
    /*implicit*/ constexpr Expansion( T val ) noexcept { c_.set( 0, val ); }

    /// Construct directly from a raw coefficient array.
    explicit constexpr Expansion( Data c ) noexcept : c_{ c } {}

    // ------------------------------------------------------------------
    // Named factories
    // ------------------------------------------------------------------

    [[nodiscard]] static constexpr Expansion zero() noexcept { return {}; }

    [[nodiscard]] static constexpr Expansion constant( T v ) noexcept { return Expansion{ v }; }

    /// Univariate identity map  f(x) = x   (1·P_1, with P_1 ≡ x in every basis).
    [[nodiscard]] static constexpr Expansion variable() noexcept
        requires( Scheme::isUnivariate )
    {
        Expansion r{};
        if constexpr ( Scheme::order >= 1 ) r.c_.set( 1, T{ 1 } );
        return r;
    }

    /// The I-th coordinate identity map  f(x) = x_I.
    template < int I >
    [[nodiscard]] static constexpr Expansion variable() noexcept
        requires( I >= 0 && I < Scheme::vars )
    {
        Expansion r{};
        if constexpr ( Scheme::order >= 1 )
        {
            MultiIndex< Scheme::vars > e{};
            e[std::size_t( I )] = 1;
            const std::size_t k = Scheme::flatOf( e );
            if ( k != Scheme::kNotInBox ) r.c_.set( k, T{ 1 } );
        }
        return r;
    }

    /// Univariate variable centred at `x0`: `x = x0 + 1·P_1` (Taylor: x0 + dx).
    [[nodiscard]] static constexpr Expansion variable( T x0 ) noexcept
        requires( Scheme::isUnivariate )
    {
        Expansion r{ x0 };
        if constexpr ( Scheme::order >= 1 ) r.c_.set( 1, T{ 1 } );
        return r;
    }

    /// Multivariate variable: the I-th coordinate variable at point `p`.
    template < int I >
    [[nodiscard]] static constexpr Expansion variable( const Input& p ) noexcept
        requires( I >= 0 && I < Scheme::vars )
    {
        Expansion r{};
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

    /// Runtime-indexed coordinate variable: `x_i = x0 + 1·dx_i`.
    [[nodiscard]] static Expansion variable( T x0, int var_idx )
    {
        if ( var_idx < 0 || var_idx >= Scheme::vars )
            throw std::out_of_range( "variable(): var_idx out of range" );
        Expansion r{};
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

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    /// Constant (zeroth) coefficient.
    [[nodiscard]] constexpr T value() const noexcept { return c_.value(); }

    /// Read coefficient at flat index `k`.
    [[nodiscard]] constexpr T operator[]( std::size_t k ) const noexcept { return c_[k]; }

    /// Write coefficient at flat index `k`.
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

    // ------------------------------------------------------------------
    // Derivative accessors (apply k! scaling to raw coefficients) — Taylor
    // value semantics; meaningful for the monomial basis.
    // ------------------------------------------------------------------

    /// Runtime partial derivative value `d^|alpha| f / dx^alpha` at x0.
    [[nodiscard]] constexpr T derivative( const MultiIndex< Scheme::vars >& alpha ) const noexcept
    {
        // Accumulate the factorial in T: std::size_t overflows at 21! on 64-bit.
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

        constexpr auto factorial = []( int n ) constexpr noexcept -> T {
            T r = T{ 1 };
            for ( int i = 2; i <= n; ++i ) r *= T( i );
            return r;
        };
        constexpr T fac = ( factorial( Alpha ) * ... * T( 1 ) );
        return coeff< Alpha... >() * fac;
    }

    // ------------------------------------------------------------------
    // Polynomial evaluation (delegated to the basis policy)
    // ------------------------------------------------------------------

    /// Evaluate the polynomial at the point vector `x` (Taylor: displacement from x0).
    [[nodiscard]] constexpr T eval( const Input& x ) const noexcept
    {
        return Basis::template eval< T, Scheme >( c_.data, x );
    }

    /// Univariate convenience: evaluate at the scalar point `x`.
    [[nodiscard]] constexpr T eval( T x ) const noexcept
        requires( Scheme::isUnivariate )
    {
        return Basis::template eval< T, Scheme >( c_.data, Input{ x } );
    }

    /// Evaluate at a displacement given as an Eigen vector.
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

    // ------------------------------------------------------------------
    // Differentiation and integration (delegated to the basis policy)
    // ------------------------------------------------------------------

    /// Partial derivative polynomial with respect to variable `I`.
    template < int I = 0 >
    [[nodiscard]] constexpr Expansion deriv() const noexcept
        requires( I >= 0 && I < Scheme::vars )
    {
        Data out{};
        Basis::template derivative< T, Scheme >( out, c_.data, I );
        return Expansion{ out };
    }

    /// Partial derivative polynomial with respect to variable `var`.
    [[nodiscard]] Expansion deriv( int var ) const
    {
        if ( var < 0 || var >= Scheme::vars )
            throw std::out_of_range( "tax::Expansion::deriv(var): var must be in [0, M)" );
        Data out{};
        Basis::template derivative< T, Scheme >( out, c_.data, var );
        return Expansion{ out };
    }

    /// Indefinite integral polynomial with respect to variable `I`.
    template < int I = 0 >
    [[nodiscard]] constexpr Expansion integ() const noexcept
        requires( I >= 0 && I < Scheme::vars )
    {
        Data out{};
        Basis::template integral< T, Scheme >( out, c_.data, I );
        return Expansion{ out };
    }

    /// Indefinite integral polynomial with respect to variable `var`.
    [[nodiscard]] Expansion integ( int var ) const
    {
        if ( var < 0 || var >= Scheme::vars )
            throw std::out_of_range( "tax::Expansion::integ(var): var must be in [0, M)" );
        Data out{};
        Basis::template integral< T, Scheme >( out, c_.data, var );
        return Expansion{ out };
    }

    // ------------------------------------------------------------------
    // Truncation
    // ------------------------------------------------------------------

    /// Order-reducing truncation: drop monomials of degree > N2.
    template < int N2 >
    [[nodiscard]] constexpr Expansion< T, Basis, IsotropicScheme< N2, Scheme::vars >,
                                       storage::Dense >
    truncate() const noexcept
        requires( is_isotropic_scheme_v< Scheme > && N2 >= 0 && N2 <= Scheme::order )
    {
        using Out = Expansion< T, Basis, IsotropicScheme< N2, Scheme::vars >, storage::Dense >;
        typename Out::Data out{};
        for ( std::size_t k = 0; k < numMonomials( N2, Scheme::vars ); ++k ) out[k] = c_[k];
        return Out{ out };
    }

    /// Same-order truncation: zero every coefficient of total degree > d.
    [[nodiscard]] constexpr Expansion truncate( int d ) const noexcept
        requires( is_isotropic_scheme_v< Scheme > )
    {
        if ( d >= Scheme::order ) return *this;
        Data out{};
        if ( d >= 0 )
            for ( std::size_t k = 0; k < numMonomials( d, Scheme::vars ); ++k ) out[k] = c_[k];
        return Expansion{ out };
    }

    // ------------------------------------------------------------------
    // Gradient and Hessian (Taylor value semantics)
    // ------------------------------------------------------------------

    /// Gradient vector `[df/dx_0, ..., df/dx_{M-1}]` at the expansion point.
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

    /// Hessian matrix `H(i,j) = d^2 f / (dx_i dx_j)` at the expansion point.
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

    // ------------------------------------------------------------------
    // Container access
    // ------------------------------------------------------------------

    [[nodiscard]] constexpr const container_t& container() const noexcept { return c_; }
    [[nodiscard]] constexpr container_t& container() noexcept { return c_; }

    /// Raw coefficient array — convenience accessor used by kernels.
    [[nodiscard]] constexpr const Data& coefficients() const noexcept { return c_.data; }
    [[nodiscard]] constexpr Data& coefficients() noexcept { return c_.data; }

   private:
    container_t c_{};
};

// ---------------------------------------------------------------------------
// Public alias: TaylorExpansion is the TaylorBasis instance of Expansion.
// ---------------------------------------------------------------------------

template < typename T, typename Scheme, typename Storage = storage::Dense >
using TaylorExpansion = Expansion< T, TaylorBasis, Scheme, Storage >;

// ---------------------------------------------------------------------------
// Convenience aliases  (dense)
// ---------------------------------------------------------------------------

/// `TE<N, M, K>` — order-N, M-variate dense `double` Taylor expansion.
template < int N, int M = 1, int K = 1 >
using TE = Expansion< std::conditional_t< K == 1, double, Batch< double, K > >, TaylorBasis,
                      IsotropicScheme< N, M >, storage::Dense >;

/// `TEn<N, M>` — explicit M-variate alias, same as `TE<N, M>`.
template < int N, int M >
using TEn = Expansion< double, TaylorBasis, IsotropicScheme< N, M >, storage::Dense >;

// ---------------------------------------------------------------------------
// Sparse specialisation (TaylorBasis only)
// ---------------------------------------------------------------------------

/// A truncated Taylor expansion in M variables of order N with sparse storage.
template < typename T, typename Basis, typename Scheme >
    requires Scalar< T > && tax::Basis< Basis > && IndexScheme< Scheme >
class Expansion< T, Basis, Scheme, storage::Sparse >
{
    static_assert( std::is_same_v< Basis, TaylorBasis >,
                   "Sparse Expansion is supported only for the TaylorBasis" );
    static_assert( is_isotropic_scheme_v< Scheme >,
                   "Sparse Expansion is supported only for IsotropicScheme<N, M>" );

    static constexpr int N = Scheme::order;
    static constexpr int M = Scheme::vars;

   public:
    static_assert( N >= 0, "Expansion<Sparse> order must be non-negative" );
    static_assert( M >= 1, "Expansion<Sparse> variable count must be at least 1" );

    // ------------------------------------------------------------------
    // Associated types
    // ------------------------------------------------------------------
    using basis = Basis;
    using scheme = Scheme;
    using scalar_type = T;
    using container_t = storage::SparseContainer< T, N, M >;
    using Input = std::array< T, std::size_t( M ) >;
    using Dense = Expansion< T, Basis, Scheme, storage::Dense >;

    // ------------------------------------------------------------------
    // Compile-time properties
    // ------------------------------------------------------------------
    static constexpr int order_v = N;
    static constexpr int vars_v = M;
    static constexpr std::size_t nCoefficients = numMonomials( N, M );

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    constexpr Expansion() = default;

    /*implicit*/ Expansion( T c )
    {
        if ( c != T{ 0 } ) c_.set( 0, c );
    }

    explicit Expansion( const Dense& d )
    {
        for ( std::size_t k = 0; k < Dense::nCoefficients; ++k )
        {
            if ( d[k] != T{ 0 } ) c_.set( k, d[k] );
        }
    }

    // ------------------------------------------------------------------
    // Named factories
    // ------------------------------------------------------------------

    [[nodiscard]] static Expansion zero() noexcept { return {}; }
    [[nodiscard]] static Expansion constant( T c ) { return Expansion{ c }; }

    [[nodiscard]] static Expansion variable( T x0 ) noexcept
        requires( M == 1 )
    {
        Expansion r;
        if ( x0 != T{ 0 } ) r.c_.set( 0, x0 );
        if constexpr ( N >= 1 ) r.c_.set( 1, T{ 1 } );
        return r;
    }

    template < int I >
    [[nodiscard]] static Expansion variable( const Input& p ) noexcept
        requires( M >= 1 && I >= 0 && I < M )
    {
        Expansion r;
        if ( p[std::size_t( I )] != T{ 0 } ) r.c_.set( 0, p[std::size_t( I )] );
        if constexpr ( N >= 1 )
        {
            MultiIndex< M > alpha{};
            alpha[std::size_t( I )] = 1;
            r.c_.set( flatIndex< M >( alpha ), T{ 1 } );
        }
        return r;
    }

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    [[nodiscard]] std::size_t nnz() const noexcept { return c_.nnz(); }
    [[nodiscard]] T value() const noexcept { return c_.value(); }

    [[nodiscard]] T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return c_.coeffAtFlat( flatIndex< M >( alpha ) );
    }

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

    // ------------------------------------------------------------------
    // Derivative accessors (apply k! scaling to raw coefficients)
    // ------------------------------------------------------------------

    [[nodiscard]] T derivative( const MultiIndex< M >& alpha ) const noexcept
    {
        T fac = T{ 1 };
        for ( int i = 0; i < M; ++i )
            for ( int j = 1; j <= alpha[std::size_t( i )]; ++j ) fac *= T( j );
        return coeff( alpha ) * fac;
    }

    template < int... Alpha >
    [[nodiscard]] T derivative() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ) );
        static_assert( ( ( Alpha >= 0 ) && ... ) );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= N );

        constexpr auto factorial = []( int n ) constexpr noexcept -> T {
            T r = T{ 1 };
            for ( int i = 2; i <= n; ++i ) r *= T( i );
            return r;
        };
        constexpr T fac = ( factorial( Alpha ) * ... * T( 1 ) );
        return coeff< Alpha... >() * fac;
    }

    // ------------------------------------------------------------------
    // Sparse-specific accessors
    // ------------------------------------------------------------------

    [[nodiscard]] std::span< const storage::flat_index_t > support() const noexcept
    {
        return c_.support();
    }

    [[nodiscard]] std::span< const T > values() const noexcept { return c_.values(); }

    // ------------------------------------------------------------------
    // Conversion
    // ------------------------------------------------------------------

    [[nodiscard]] Dense dense() const noexcept
    {
        Dense r;
        c_.forEachNonzero( [&]( std::size_t k, T v ) { r[k] = v; } );
        return r;
    }

    // ------------------------------------------------------------------
    // Truncation
    // ------------------------------------------------------------------

    template < int N2 >
    [[nodiscard]] Expansion< T, Basis, IsotropicScheme< N2, M >, storage::Sparse > truncate()
        const noexcept
        requires( N2 >= 0 && N2 <= N )
    {
        return truncatedBelow< Expansion< T, Basis, IsotropicScheme< N2, M >, storage::Sparse > >(
            numMonomials( N2, M ) );
    }

    [[nodiscard]] Expansion truncate( int d ) const noexcept
    {
        if ( d >= N ) return *this;
        return truncatedBelow< Expansion >( d >= 0 ? numMonomials( d, M ) : 0 );
    }

    // ------------------------------------------------------------------
    // Container access
    // ------------------------------------------------------------------

    [[nodiscard]] const container_t& container() const noexcept { return c_; }
    [[nodiscard]] container_t& container() noexcept { return c_; }

   private:
    template < typename Result >
    [[nodiscard]] Result truncatedBelow( std::size_t limit ) const noexcept
    {
        Result r;
        const auto cap = storage::flat_index_t( limit );
        auto& oi = r.container().rawIndices();
        auto& ov = r.container().rawValues();
        const auto sup = support();
        const auto vals = values();
        for ( std::size_t i = 0; i < sup.size(); ++i )
        {
            if ( sup[i] >= cap ) break;
            oi.push_back( sup[i] );
            ov.push_back( vals[i] );
        }
        return r;
    }

    container_t c_;
};

// ---------------------------------------------------------------------------
// Convenience aliases  (sparse)
// ---------------------------------------------------------------------------

/// `STE<N>` / `STE<N, M>` — sparse `double` Taylor expansion.
template < int N, int M = 1 >
using STE = Expansion< double, TaylorBasis, IsotropicScheme< N, M >, storage::Sparse >;

/// `MixedTE<Groups...>` — an anisotropic (per-group order) dense `double` expansion.
template < typename... Groups >
using MixedTE = Expansion< double, TaylorBasis, MixedScheme< Groups... >, storage::Dense >;

// ---------------------------------------------------------------------------
// Free-function variable factories (unnamed, integer-indexed)
// ---------------------------------------------------------------------------

/// Univariate variable `x = x0 + 1·dx` of an order-`N` dense expansion.
template < int N, Scalar T = double >
[[nodiscard]] constexpr auto variable( T x0 ) noexcept
{
    return Expansion< T, TaylorBasis, IsotropicScheme< N, 1 > >::variable( x0 );
}

/// The `I`-th coordinate variable of an order-`N`, `M`-variate dense expansion at point `p`.
template < int I, int N, int M, Scalar T = double >
[[nodiscard]] constexpr auto variable( const std::array< T, std::size_t( M ) >& p ) noexcept
{
    return Expansion< T, TaylorBasis, IsotropicScheme< N, M > >::template variable< I >( p );
}

/// All `M` coordinate variables of an order-`N`, `M`-variate dense expansion at point `p`.
template < int N, int M, Scalar T = double >
[[nodiscard]] constexpr auto variables( const std::array< T, std::size_t( M ) >& p ) noexcept
{
    using E = Expansion< T, TaylorBasis, IsotropicScheme< N, M > >;
    std::array< E, std::size_t( M ) > out{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out[I] = E::template variable< int( I ) >( p ) ), ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );
    return out;
}

// ---------------------------------------------------------------------------
// Conversion helper: dense -> sparse
// ---------------------------------------------------------------------------

/// Convert a dense polynomial to sparse storage (drops exact zeros).
template < typename T, typename Basis, IndexScheme Scheme >
[[nodiscard]] Expansion< T, Basis, Scheme, storage::Sparse > sparse(
    const Expansion< T, Basis, Scheme, storage::Dense >& d ) noexcept
{
    return Expansion< T, Basis, Scheme, storage::Sparse >( d );
}

}  // namespace tax
