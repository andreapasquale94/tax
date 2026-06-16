// include/tax/stats/expectation.hpp
//
// Non-Monte-Carlo statistics of a Taylor-polynomial map. Following Valli,
// Armellin, Di Lizia & Lavagna, "Nonlinear Mapping of Uncertainties in
// Celestial Mechanics" (JGCD 2013), the statistics of f(dx) are obtained by
// contracting the Taylor coefficients c_alpha against the raw moments
// <dx^alpha> of the input distribution:
//
//   E[f]            = sum_alpha c_alpha <dx^alpha>
//   E[f_i f_j]      = sum_{alpha,beta} c^i_alpha c^j_beta <dx^{alpha+beta}>
//   Cov[f_i, f_j]   = E[f_i f_j] - E[f_i] E[f_j]
//
// The mean is a contraction with moments up to order N; the (co)variance is the
// exact double contraction with moments up to order 2N (no polynomial-product
// truncation). Central moments of order p >= 3 are formed from the truncated
// Taylor power (f - E[f])^p and are therefore consistent with the order-N map.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <tax/core/multi_index.hpp>
#include <tax/la/num_traits.hpp>
#include <tax/stats/concepts.hpp>
#include <vector>

namespace tax::stats
{

namespace detail
{

// Cache of the (multi-index, flatIndex) pairs for every monomial up to order N
// in M variables. Built once per (N, M) and reused across contractions.
template < int N, int M >
struct MonomialTable
{
    static constexpr std::size_t size = numMonomials( N, M );
    std::array< MultiIndex< M >, size > idx{};

    constexpr MonomialTable() noexcept
    {
        for ( std::size_t k = 0; k < size; ++k ) idx[k] = unflatIndex< M >( k );
    }
};

template < int N, int M >
inline constexpr MonomialTable< N, M > monomialTable{};

// Linear contraction sum_k c_k * moment_k of a TE against a moment table.
template < typename T, int N, int M, typename S >
[[nodiscard]] T contract( const TaylorExpansion< T, N, M, S >& f,
                          const std::vector< T >& moments ) noexcept
{
    const auto& mono = monomialTable< N, M >;
    T acc = T( 0 );
    for ( std::size_t k = 0; k < TaylorExpansion< T, N, M, S >::nCoefficients; ++k )
        acc += f.coeff( mono.idx[k] ) * moments[k];
    return acc;
}

// Exact second central moment (variance) of a scalar TE, given the order-2N
// moment table. Shared by `variance` and the `moments` summary so the table is
// built only once. `moments` must cover degree 2N.
template < typename T, int N, int M, typename S >
[[nodiscard]] T varianceFromTable( const TaylorExpansion< T, N, M, S >& f,
                                   const std::vector< T >& moments ) noexcept
{
    constexpr std::size_t nc = numMonomials( N, M );
    const auto& mono = monomialTable< N, M >;

    std::array< T, nc > c{};
    T mean = T( 0 );
    for ( std::size_t k = 0; k < nc; ++k )
    {
        c[k] = f.coeff( mono.idx[k] );
        mean += c[k] * moments[k];
    }
    T e2 = T( 0 );
    for ( std::size_t a = 0; a < nc; ++a )
    {
        if ( c[a] == T( 0 ) ) continue;
        for ( std::size_t b = 0; b < nc; ++b )
        {
            if ( c[b] == T( 0 ) ) continue;
            MultiIndex< M > sum{};
            for ( std::size_t v = 0; v < std::size_t( M ); ++v )
                sum[v] = mono.idx[a][v] + mono.idx[b][v];
            e2 += c[a] * c[b] * moments[flatIndex< M >( sum )];
        }
    }
    return e2 - mean * mean;
}

}  // namespace detail

// -----------------------------------------------------------------------------
// expectation — E[f] for a scalar TE, or element-wise mean of a TE matrix
// -----------------------------------------------------------------------------

/**
 * @brief Expectation `E[g]` of a scalar `TaylorExpansion` under `dist`.
 *
 * Works for any polynomial `g` (it is a plain linear contraction of the
 * coefficients with the input moments), so it doubles as the low-level
 * primitive behind the mean and central-moment routines.
 */
template < typename T, int N, int M, typename S, MomentProvider P >
[[nodiscard]] T expectation( const TaylorExpansion< T, N, M, S >& g, const P& dist )
{
    static_assert( P::vars == M, "expectation(): distribution variable count must match M" );
    return detail::contract( g, dist.momentTable( N ) );
}

/**
 * @brief Element-wise mean of an Eigen matrix/vector of `TaylorExpansion`s.
 * @return Eigen matrix/vector of scalars, same shape as `F`.
 */
template < typename Derived, MomentProvider P >
    requires( la::detail::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto expectation( const Eigen::MatrixBase< Derived >& F, const P& dist )
{
    using TE = typename Derived::Scalar;
    using tr = la::detail::te_traits< TE >;
    using T = typename tr::scalar_type;
    constexpr int N = tr::order_v;
    constexpr int M = tr::vars_v;
    static_assert( P::vars == M, "expectation(): distribution variable count must match M" );

    const std::vector< T > moments = dist.momentTable( N );
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime > out( F.rows(),
                                                                                    F.cols() );
    for ( Eigen::Index e = 0; e < F.size(); ++e )
        out( e ) = detail::contract( F.derived().coeff( e ), moments );
    return out;
}

// -----------------------------------------------------------------------------
// covariance / variance — exact second-moment contraction (moments up to 2N)
// -----------------------------------------------------------------------------

/**
 * @brief Covariance matrix of a vector-valued `TaylorExpansion` map under `dist`.
 *
 * `Cov(i, j) = E[F_i F_j] - E[F_i] E[F_j]`, evaluated exactly via the order-`2N`
 * double contraction of the coefficient arrays — no truncation of the implied
 * polynomial product.
 *
 * @param F  Eigen column vector of `K` `TaylorExpansion` components.
 * @return   `Eigen::Matrix<T, K, K>` symmetric covariance matrix.
 */
template < typename Derived, MomentProvider P >
    requires( la::detail::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto covariance( const Eigen::MatrixBase< Derived >& F, const P& dist )
{
    using TE = typename Derived::Scalar;
    using tr = la::detail::te_traits< TE >;
    using T = typename tr::scalar_type;
    constexpr int N = tr::order_v;
    constexpr int M = tr::vars_v;
    constexpr int K = Derived::SizeAtCompileTime;
    static_assert( P::vars == M, "covariance(): distribution variable count must match M" );

    constexpr std::size_t nc = numMonomials( N, M );
    const std::vector< T > mom = dist.momentTable( 2 * N );
    const auto& mono = detail::monomialTable< N, M >;

    // Gather coefficient rows and per-component means once.
    const Eigen::Index k_dim = F.size();
    std::vector< std::array< T, nc > > coeffs( static_cast< std::size_t >( k_dim ) );
    Eigen::Matrix< T, K, 1 > mean( k_dim );
    for ( Eigen::Index r = 0; r < k_dim; ++r )
    {
        const TE& f = F.derived().coeff( r );
        T m = T( 0 );
        for ( std::size_t k = 0; k < nc; ++k )
        {
            const T c = f.coeff( mono.idx[k] );
            coeffs[std::size_t( r )][k] = c;
            m += c * mom[k];
        }
        mean( r ) = m;
    }

    Eigen::Matrix< T, K, K > cov( k_dim, k_dim );
    for ( Eigen::Index i = 0; i < k_dim; ++i )
    {
        const auto& ci = coeffs[std::size_t( i )];
        for ( Eigen::Index j = i; j < k_dim; ++j )
        {
            const auto& cj = coeffs[std::size_t( j )];
            T e2 = T( 0 );
            for ( std::size_t a = 0; a < nc; ++a )
            {
                if ( ci[a] == T( 0 ) ) continue;
                for ( std::size_t b = 0; b < nc; ++b )
                {
                    if ( cj[b] == T( 0 ) ) continue;
                    MultiIndex< M > sum{};
                    for ( std::size_t v = 0; v < std::size_t( M ); ++v )
                        sum[v] = mono.idx[a][v] + mono.idx[b][v];
                    e2 += ci[a] * cj[b] * mom[flatIndex< M >( sum )];
                }
            }
            const T c = e2 - mean( i ) * mean( j );
            cov( i, j ) = c;
            cov( j, i ) = c;
        }
    }
    return cov;
}

/**
 * @brief Variance `Var[f]` of a scalar `TaylorExpansion` (exact, order `2N`).
 */
template < typename T, int N, int M, typename S, MomentProvider P >
[[nodiscard]] T variance( const TaylorExpansion< T, N, M, S >& f, const P& dist )
{
    static_assert( P::vars == M, "variance(): distribution variable count must match M" );
    return detail::varianceFromTable( f, dist.momentTable( 2 * N ) );
}

// -----------------------------------------------------------------------------
// central moments / standardized moments
// -----------------------------------------------------------------------------

/**
 * @brief Central moment `E[(f - E[f])^p]` of a scalar `TaylorExpansion`.
 *
 * The deviation `f - E[f]` is raised to the `p`-th power in Taylor arithmetic
 * (truncated at order `N`) and then contracted with the input moments. The
 * result is therefore exact when the deviation polynomial has degree `<= N/p`
 * and an order-`N` approximation otherwise. For `p == 2`, prefer `variance`,
 * which is exact to order `2N`.
 *
 * @param p Moment order (`p == 0` returns 1, `p == 1` returns ~0).
 */
template < typename T, int N, int M, typename S, MomentProvider P >
[[nodiscard]] T centralMoment( const TaylorExpansion< T, N, M, S >& f, const P& dist, int p )
{
    static_assert( P::vars == M, "centralMoment(): distribution variable count must match M" );
    if ( p <= 0 ) return T( 1 );
    const T mu = expectation( f, dist );
    const TaylorExpansion< T, N, M, S > dev = f - TaylorExpansion< T, N, M, S >( mu );
    TaylorExpansion< T, N, M, S > power = dev;
    for ( int e = 1; e < p; ++e ) power = power * dev;
    return expectation( power, dist );
}

/**
 * @brief Skewness `E[(f - E[f])^3] / Var[f]^{3/2}` of a scalar `TaylorExpansion`.
 */
template < typename T, int N, int M, typename S, MomentProvider P >
[[nodiscard]] T skewness( const TaylorExpansion< T, N, M, S >& f, const P& dist )
{
    const T var = variance( f, dist );
    return centralMoment( f, dist, 3 ) / std::pow( var, T( 1.5 ) );
}

/**
 * @brief Kurtosis `E[(f - E[f])^4] / Var[f]^2` of a scalar `TaylorExpansion`.
 * @param excess  When true, returns the excess kurtosis (subtracting 3).
 */
template < typename T, int N, int M, typename S, MomentProvider P >
[[nodiscard]] T kurtosis( const TaylorExpansion< T, N, M, S >& f, const P& dist,
                          bool excess = false )
{
    const T var = variance( f, dist );
    const T k = centralMoment( f, dist, 4 ) / ( var * var );
    return excess ? k - T( 3 ) : k;
}

// -----------------------------------------------------------------------------
// order-≤4 convenience: mean alias + one-shot moment summary
// -----------------------------------------------------------------------------

/// @brief Mean `E[f]` of a scalar `TaylorExpansion` (alias of `expectation`).
template < typename T, int N, int M, typename S, MomentProvider P >
[[nodiscard]] T mean( const TaylorExpansion< T, N, M, S >& f, const P& dist )
{
    return expectation( f, dist );
}

/// @brief Mean vector of an Eigen matrix/vector of `TaylorExpansion`s
///        (alias of `expectation`).
template < typename Derived, MomentProvider P >
    requires( la::detail::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto mean( const Eigen::MatrixBase< Derived >& F, const P& dist )
{
    return expectation( F, dist );
}

/**
 * @brief The first four moments of a scalar `TaylorExpansion`.
 *
 * `mean` and `variance` are exact (orders `N` and `2N`); `skewness` and
 * `kurtosis` use the order-`N`-truncated deviation powers (see `centralMoment`).
 * `kurtosis` is the non-excess form (3 for a Gaussian).
 */
template < typename T >
struct Moments
{
    T mean;      ///< E[f]
    T variance;  ///< E[(f - E[f])^2]
    T skewness;  ///< E[(f - E[f])^3] / variance^{3/2}
    T kurtosis;  ///< E[(f - E[f])^4] / variance^2  (Gaussian -> 3)

    /// @brief Standard deviation, `sqrt(variance)`.
    [[nodiscard]] T standardDeviation() const { return std::sqrt( variance ); }
    /// @brief Excess kurtosis, `kurtosis - 3` (Gaussian -> 0).
    [[nodiscard]] T excessKurtosis() const { return kurtosis - T( 3 ); }
};

/**
 * @brief Compute mean, variance, skewness and kurtosis of a scalar
 *        `TaylorExpansion` in a single pass (shared moment table).
 *
 * Equivalent to calling `mean`, `variance`, `skewness` and `kurtosis`
 * separately, but builds the order-`2N` moment table only once.
 */
template < typename T, int N, int M, typename S, MomentProvider P >
[[nodiscard]] Moments< T > moments( const TaylorExpansion< T, N, M, S >& f, const P& dist )
{
    static_assert( P::vars == M, "moments(): distribution variable count must match M" );
    using TE = TaylorExpansion< T, N, M, S >;

    const std::vector< T > mom = dist.momentTable( 2 * N );
    const T mu = detail::contract( f, mom );  // uses moments up to order N
    const T var = detail::varianceFromTable( f, mom );

    // Third/fourth central moments from the (truncated) deviation powers.
    const TE dev = f - TE( mu );
    const TE dev2 = dev * dev;
    const TE dev3 = dev2 * dev;
    const TE dev4 = dev3 * dev;
    const T mu3 = detail::contract( dev3, mom );
    const T mu4 = detail::contract( dev4, mom );

    const T sd = std::sqrt( var );
    return Moments< T >{ mu, var, mu3 / ( var * sd ), mu4 / ( var * var ) };
}

}  // namespace tax::stats
