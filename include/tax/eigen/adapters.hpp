#pragma once

#include <tax/eigen/num_traits.hpp>
#include <cassert>
#include <concepts>
#include <tuple>
#include <utility>

namespace tax
{

namespace detail::eigen
{

/// @brief Traits for extracting scalar type, order, and number of variables from a TruncatedTaylorExpansionT type.
template < typename >
struct da_traits;

template < typename T, int N, int M >
struct da_traits< TruncatedTaylorExpansionT< T, N, M > >
{
    using scalar_type = T;
    static constexpr int order = N;
    static constexpr int vars = M;
};

/// @brief True if `DA` is a `TruncatedTaylorExpansionT<T, N, M>` specialization.
template < typename >
inline constexpr bool is_da_v = false;

template < typename T, int N, int M >
inline constexpr bool is_da_v< TruncatedTaylorExpansionT< T, N, M > > = true;

/// @brief Rebind an Eigen matrix type to use a different scalar.
template < typename Derived, typename Scalar >
using rebind_matrix_t =
    Eigen::Matrix< Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime,
                   Derived::Options, Derived::MaxRowsAtCompileTime,
                   Derived::MaxColsAtCompileTime >;

/// @brief Concept for Eigen dense expression types.
template < typename T >
concept EigenDenseExpr = requires( const T& t ) {
    typename T::Scalar;
    t.derived();
};

}  // namespace detail::eigen

/**
 * @brief Create a DA tensor from a compile-time-sized Eigen vector/matrix expansion point.
 * @tparam DA The TruncatedTaylorExpansionT type (e.g., `TEn<2, 4>`). `M` must equal the input size.
 * @param x0 Eigen matrix/vector with `M` entries.
 * @return Eigen matrix of same shape with DA variable entries.
 */
template < typename DA, typename Derived >
[[nodiscard]] auto tensor( const Eigen::DenseBase< Derived >& x0 ) noexcept
    requires( detail::eigen::is_da_v< DA > &&
              std::convertible_to< typename Derived::Scalar,
                                   typename detail::eigen::da_traits< DA >::scalar_type > )
{
    using T = typename detail::eigen::da_traits< DA >::scalar_type;
    constexpr int M = detail::eigen::da_traits< DA >::vars;
    constexpr int Rows = Derived::RowsAtCompileTime;
    constexpr int Cols = Derived::ColsAtCompileTime;
    static_assert( Rows != Eigen::Dynamic && Cols != Eigen::Dynamic,
                   "tensor(x0) requires compile-time-sized Eigen inputs" );
    static_assert( Rows >= 1 && Cols >= 1, "tensor(x0) requires non-empty Eigen inputs" );
    static_assert( Rows * Cols == M, "tensor(x0) size must match number of variables M" );

    using Out = Eigen::Matrix< DA, Rows, Cols, Derived::Options, Derived::MaxRowsAtCompileTime,
                               Derived::MaxColsAtCompileTime >;
    typename DA::point_type p{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( [&] {
            if constexpr ( Rows == 1 )
                p[I] = static_cast< T >( x0( Eigen::Index( 0 ), Eigen::Index( I ) ) );
            else if constexpr ( Cols == 1 )
                p[I] = static_cast< T >( x0( Eigen::Index( I ), Eigen::Index( 0 ) ) );
            else
                p[I] = static_cast< T >(
                    x0( Eigen::Index( int( I ) / Cols ), Eigen::Index( int( I ) % Cols ) ) );
        }(),
          ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );

    Out out{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out( Eigen::Index( int( I ) / Cols ), Eigen::Index( int( I ) % Cols ) ) =
                DA::template variable< int( I ) >( p ) ),
          ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );
    return out;
}

/**
 * @brief Create all coordinate variables from an Eigen vector expansion point.
 * @tparam DA The TruncatedTaylorExpansionT type (e.g., `TEn<2, 3>`). `M` must match the vector size.
 * @param x0 Eigen vector with `M` entries.
 * @return Tuple `(x_0, ..., x_{M-1})` of DA variables.
 */
template < typename DA, typename Derived >
[[nodiscard]] auto variables( const Eigen::DenseBase< Derived >& x0 ) noexcept
    requires( detail::eigen::is_da_v< DA > &&
              std::convertible_to< typename Derived::Scalar,
                                   typename detail::eigen::da_traits< DA >::scalar_type > )
{
    using T = typename detail::eigen::da_traits< DA >::scalar_type;
    constexpr int M = detail::eigen::da_traits< DA >::vars;

    static_assert( Derived::RowsAtCompileTime == 1 || Derived::ColsAtCompileTime == 1 ||
                       Derived::RowsAtCompileTime == Eigen::Dynamic ||
                       Derived::ColsAtCompileTime == Eigen::Dynamic,
                   "Eigen input must be a vector expression" );
    static_assert(
        Derived::SizeAtCompileTime == Eigen::Dynamic || Derived::SizeAtCompileTime == M,
        "Eigen vector size must match number of variables" );
    assert( x0.rows() == 1 || x0.cols() == 1 );
    assert( x0.size() == Eigen::Index( M ) );

    typename DA::point_type p{};
    if ( x0.cols() == 1 )
    {
        for ( int i = 0; i < M; ++i )
            p[std::size_t( i )] =
                static_cast< T >( x0( Eigen::Index( i ), Eigen::Index( 0 ) ) );
    } else
    {
        for ( int i = 0; i < M; ++i )
            p[std::size_t( i )] =
                static_cast< T >( x0( Eigen::Index( 0 ), Eigen::Index( i ) ) );
    }
    return DA::variables( p );
}

/**
 * @brief Evaluate a DA polynomial at displacement given as an Eigen vector.
 * @param f The DA polynomial.
 * @param dx Eigen vector with `M` entries holding the displacement.
 * @return `f(x0 + dx)` truncated to order `N`.
 */
template < typename T, int N, int M, typename Derived >
[[nodiscard]] T eval( const TruncatedTaylorExpansionT< T, N, M >& f, const Eigen::DenseBase< Derived >& dx ) noexcept
    requires( M > 1 && std::convertible_to< typename Derived::Scalar, T > )
{
    typename TruncatedTaylorExpansionT< T, N, M >::point_type p{};
    for ( int i = 0; i < M; ++i )
        p[std::size_t( i )] = static_cast< T >( dx( Eigen::Index( i ) ) );
    return f.eval( p );
}

/**
 * @brief Extract the k-th Taylor coefficient from every element of a DA column-vector.
 *
 * Gathers `v(i).coeffs()[k]` for all `i` into a contiguous Eigen vector, making the
 * extracted slice available for vectorised arithmetic (scaling, norms, etc.).
 *
 * @tparam T    Scalar type.
 * @tparam N    Taylor order of the DA elements.
 * @tparam Dim  Static row count; use `Eigen::Dynamic` for runtime-sized inputs.
 * @param  v    DA-typed column vector.
 * @param  k    Coefficient index (0 ≤ k ≤ N).
 * @return      `Eigen::Matrix<T, Dim, 1>` containing the k-th coefficients.
 */
template < typename T, int N, int Dim >
[[nodiscard]] Eigen::Matrix< T, Dim, 1 >
coeffRow( const Eigen::Matrix< TruncatedTaylorExpansionT< T, N, 1 >, Dim, 1 >& v, int k ) noexcept
{
    const Eigen::Index dim = v.size();
    Eigen::Matrix< T, Dim, 1 > out( dim );
    for ( Eigen::Index i = 0; i < dim; ++i )
        out( i ) = v( i ).coeffSpan()[std::size_t( k )];
    return out;
}

/**
 * @brief Scatter values into the k-th Taylor coefficient slot of every element of a
 *        DA column-vector.
 *
 * Sets `v(i).coeffSpan()[k] = vals(i)` for all `i`.  Accepts any Eigen
 * column-vector *expression* for `vals` so callers can pass temporaries such as
 * `coeffRow(f_da, k) / (k + 1)` without an extra intermediate allocation.
 *
 * @tparam T       Scalar type.
 * @tparam N       Taylor order of the DA elements.
 * @tparam Dim     Static row count of the target vector.
 * @tparam ValsDer Derived type of the source Eigen expression.
 * @param  v       DA-typed column vector to modify in-place.
 * @param  k       Coefficient index (0 ≤ k ≤ N).
 * @param  vals    Source expression; must have the same runtime size as `v`.
 */
template < typename T, int N, int Dim, typename ValsDer >
void setCoeffRow( Eigen::Matrix< TruncatedTaylorExpansionT< T, N, 1 >, Dim, 1 >& v, int k,
                  const Eigen::MatrixBase< ValsDer >& vals ) noexcept
{
    assert( v.size() == vals.size() );
    for ( Eigen::Index i = 0; i < v.size(); ++i )
        v( i ).coeffSpan()[std::size_t( k )] = vals.derived()( i );
}

/**
 * @brief Vectorised evaluation of a univariate DA column-vector at scalar @p h.
 *
 * All @p Dim polynomial evaluations are performed simultaneously via a single
 * matrix–vector product:
 * @code
 *   result = C * [1, h, h², …, hᴺ]ᵀ
 * @endcode
 * where `C(i, k) = y_da(i)[k]`.
 *
 * Exposing @p T, @p N, and @p Dim as explicit template parameters (rather than
 * deducing them from a generic vector type) lets Eigen select fully-specialised,
 * SIMD-accelerated kernels whenever the state dimension and polynomial order are
 * known at compile time.  In particular:
 *   - `C` has compile-time column count `N+1`, so the inner accumulation loop is
 *     always unrolled.
 *   - When @p Dim is a fixed positive integer the whole product is unrolled
 *     into register-level SIMD operations with no heap allocation.
 *   - When @p Dim is `Eigen::Dynamic` the kernel still has a fixed-width inner
 *     loop (`N+1` iterations) that the compiler can auto-vectorise.
 *
 * @tparam T    Scalar (coefficient) type (e.g. `double`).
 * @tparam N    Taylor order of the DA elements.
 * @tparam Dim  Static row count of the state vector; use `Eigen::Dynamic` for
 *              runtime-sized vectors.
 * @param  y_da Eigen column-vector of `TruncatedTaylorExpansionT<T,N,1>` elements (length @p Dim).
 * @param  h    Scalar step size (displacement from the Taylor expansion point).
 * @return      `Eigen::Matrix<T, Dim, 1>` with the evaluated polynomial values.
 */
template < typename T, int N, int Dim >
[[nodiscard]] Eigen::Matrix< T, Dim, 1 >
evalSeries( const Eigen::Matrix< TruncatedTaylorExpansionT< T, N, 1 >, Dim, 1 >& y_da, T h ) noexcept
{
    constexpr int N1       = N + 1;
    const Eigen::Index dim = y_da.size();

    // Coefficient matrix: row i holds the N+1 Taylor coefficients of y_da(i).
    // Fixed column count N+1 lets Eigen fully unroll the inner GEMV loop.
    Eigen::Matrix< T, Dim, N1 > C( dim, N1 );
    for ( Eigen::Index i = 0; i < dim; ++i )
    {
        const auto sp = y_da( i ).coeffSpan();
        for ( int k = 0; k < N1; ++k )
            C( i, k ) = sp[std::size_t( k )];
    }

    // h-powers vector (fixed size N+1): hp[k] = hᵏ.
    Eigen::Matrix< T, N1, 1 > hp;
    hp[0] = T( 1 );
    for ( int k = 1; k < N1; ++k )
        hp[k] = hp[k - 1] * h;

    // Matrix–vector product with full compile-time dimension knowledge.
    return C * hp;
}

}  // namespace tax
