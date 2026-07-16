// Statistical moments (mean, covariance, per-component standardized
// skewness/kurtosis, and the full skewness/kurtosis central-moment tensors) of a
// polynomial map `F : R^M -> R^D` represented as an Eigen vector of dense,
// isotropic `TaylorExpansion`s, under the assumption that the expansion's `M`
// formal variables are i.i.d. standard normal (`x ~ N(0, I_M)`) — the usual
// differential-algebra convention where any input covariance/correlation is
// already folded into how the caller built `x0` and the variables (e.g. via a
// Cholesky-whitened `tax::la::variables` call).
//
// Every moment reduces to `E[prod x_i^{p_i}]` over the (independent) formal
// variables, which for a standard normal factors axis-by-axis into
// `E[X^p] = (p-1)!!` for even `p`, `0` for odd `p` — the closed form behind
// Isserlis'/Wick's theorem for a single Gaussian variable. See
// `docs/internals/moments.md` for the full derivation and references.
//
// This header is NOT pulled in by the `<tax/tax.hpp>` / `<tax/la.hpp>` umbrellas
// (it depends on Eigen's heavy `unsupported/Eigen/CXX11/Tensor` module), so
// include it explicitly: `#include <tax/la/moments.hpp>`.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/la/num_traits.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace tax::la::detail
{

/// Constrains `mean`/`covariance`/`skewnessTensor`/`kurtosisTensor` to dense,
/// isotropic expansions: the raw-moment machinery below indexes coefficients
/// via `Coeffs<T, N, M> == numMonomials(N, M)` slots, which only matches the
/// `IsotropicScheme<N, M>` layout.
template < typename TE >
concept MomentsCompatibleTE =
    is_te_v< TE > && is_isotropic_scheme_v< typename te_traits< TE >::scheme_t > &&
    std::is_same_v< typename te_traits< TE >::storage_t, tax::storage::Dense >;

/// `E[X^p]` for `X ~ N(0,1)`: the double factorial `(p-1)!!` for even `p` (with
/// `(-1)!! := 1`), `0` for odd `p`.
template < typename T >
constexpr T gaussianRawMoment( int p ) noexcept
{
    if ( p % 2 != 0 ) return T{};
    T r{ 1 };
    for ( int k = p - 1; k > 0; k -= 2 ) r *= T( k );
    return r;
}

/// `E[F(x)]` for `x ~ N(0, I_M)` i.i.d., `F` given by its monomial coefficients.
template < int N, int M, typename T >
[[nodiscard]] T jointRawMoment1( const Coeffs< T, N, M >& a ) noexcept
{
    T acc{};
    forEachMonomial< M, N >( [&]( const MultiIndex< M >& alpha ) {
        const T ca = a[flatIndex< M >( alpha )];
        if ( ca == T{} ) return;
        T w{ 1 };
        for ( int ax = 0; ax < M; ++ax ) w *= gaussianRawMoment< T >( alpha[std::size_t( ax )] );
        acc += ca * w;
    } );
    return acc;
}

/// `E[F_a(x) F_b(x)]` for `x ~ N(0, I_M)` i.i.d.
template < int N, int M, typename T >
[[nodiscard]] T jointRawMoment2( const Coeffs< T, N, M >& a, const Coeffs< T, N, M >& b ) noexcept
{
    T acc{};
    forEachMonomial< M, N >( [&]( const MultiIndex< M >& alpha ) {
        const T ca = a[flatIndex< M >( alpha )];
        if ( ca == T{} ) return;
        forEachMonomial< M, N >( [&]( const MultiIndex< M >& beta ) {
            const T cb = b[flatIndex< M >( beta )];
            if ( cb == T{} ) return;
            T w{ 1 };
            for ( int ax = 0; ax < M; ++ax )
                w *= gaussianRawMoment< T >( alpha[std::size_t( ax )] + beta[std::size_t( ax )] );
            acc += ca * cb * w;
        } );
    } );
    return acc;
}

/// `E[F_a(x) F_b(x) F_c(x)]` for `x ~ N(0, I_M)` i.i.d.
template < int N, int M, typename T >
[[nodiscard]] T jointRawMoment3( const Coeffs< T, N, M >& a, const Coeffs< T, N, M >& b,
                                 const Coeffs< T, N, M >& c ) noexcept
{
    T acc{};
    forEachMonomial< M, N >( [&]( const MultiIndex< M >& alpha ) {
        const T ca = a[flatIndex< M >( alpha )];
        if ( ca == T{} ) return;
        forEachMonomial< M, N >( [&]( const MultiIndex< M >& beta ) {
            const T cb = b[flatIndex< M >( beta )];
            if ( cb == T{} ) return;
            forEachMonomial< M, N >( [&]( const MultiIndex< M >& gamma ) {
                const T cc = c[flatIndex< M >( gamma )];
                if ( cc == T{} ) return;
                T w{ 1 };
                for ( int ax = 0; ax < M; ++ax )
                    w *=
                        gaussianRawMoment< T >( alpha[std::size_t( ax )] + beta[std::size_t( ax )] +
                                                gamma[std::size_t( ax )] );
                acc += ca * cb * cc * w;
            } );
        } );
    } );
    return acc;
}

/// `E[F_a(x) F_b(x) F_c(x) F_d(x)]` for `x ~ N(0, I_M)` i.i.d.
template < int N, int M, typename T >
[[nodiscard]] T jointRawMoment4( const Coeffs< T, N, M >& a, const Coeffs< T, N, M >& b,
                                 const Coeffs< T, N, M >& c, const Coeffs< T, N, M >& d ) noexcept
{
    T acc{};
    forEachMonomial< M, N >( [&]( const MultiIndex< M >& alpha ) {
        const T ca = a[flatIndex< M >( alpha )];
        if ( ca == T{} ) return;
        forEachMonomial< M, N >( [&]( const MultiIndex< M >& beta ) {
            const T cb = b[flatIndex< M >( beta )];
            if ( cb == T{} ) return;
            forEachMonomial< M, N >( [&]( const MultiIndex< M >& gamma ) {
                const T cc = c[flatIndex< M >( gamma )];
                if ( cc == T{} ) return;
                forEachMonomial< M, N >( [&]( const MultiIndex< M >& delta ) {
                    const T cd = d[flatIndex< M >( delta )];
                    if ( cd == T{} ) return;
                    T w{ 1 };
                    for ( int ax = 0; ax < M; ++ax )
                        w *= gaussianRawMoment< T >(
                            alpha[std::size_t( ax )] + beta[std::size_t( ax )] +
                            gamma[std::size_t( ax )] + delta[std::size_t( ax )] );
                    acc += ca * cb * cc * cd * w;
                } );
            } );
        } );
    } );
    return acc;
}

/// Monomial coefficients of `f - mu` for a scalar constant `mu`: since
/// subtracting a constant only ever touches the degree-0 slot, this is exact
/// regardless of `f`'s degree (`mu` need not be `f`'s own constant term — it's
/// the *statistical* mean, which generally differs from it).
template < int N, int M, typename T >
[[nodiscard]] Coeffs< T, N, M > centeredCoeffs( const Coeffs< T, N, M >& f, T mu ) noexcept
{
    Coeffs< T, N, M > c = f;
    c[0] -= mu;
    return c;
}

}  // namespace tax::la::detail

namespace tax::la
{

/// Mean `E[F]` of a vector map `F(x)`, `x ~ N(0, I)` i.i.d. standard normal.
template < typename Derived >
    requires( detail::MomentsCompatibleTE< typename Derived::Scalar > )
[[nodiscard]] auto mean( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T = typename detail::te_traits< TE >::scalar_type;
    constexpr int N = detail::te_traits< TE >::order_v;
    constexpr int M = detail::te_traits< TE >::vars_v;

    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime > out( F.rows(),
                                                                                    F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i )
        out( i ) = detail::jointRawMoment1< N, M, T >( F.derived().coeff( i ).coefficients() );
    return out;
}

/// Covariance matrix `Cov(F_i, F_j)` of a vector map, `x ~ N(0, I)` i.i.d.,
/// returned as a fixed-size `D x D` Eigen matrix.
template < typename Derived >
    requires( detail::MomentsCompatibleTE< typename Derived::Scalar > )
[[nodiscard]] auto covariance( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T = typename detail::te_traits< TE >::scalar_type;
    constexpr int N = detail::te_traits< TE >::order_v;
    constexpr int M = detail::te_traits< TE >::vars_v;
    static_assert( Derived::SizeAtCompileTime != Eigen::Dynamic,
                   "covariance: the map's dimension D must be known at compile time" );
    constexpr int D = Derived::SizeAtCompileTime;

    const auto mu = mean( F );
    std::vector< Coeffs< T, N, M > > centered( static_cast< std::size_t >( D ) );
    for ( int i = 0; i < D; ++i )
        centered[std::size_t( i )] =
            detail::centeredCoeffs< N, M, T >( F.derived().coeff( i ).coefficients(), mu( i ) );

    Eigen::Matrix< T, D, D > out;
    for ( int i = 0; i < D; ++i )
        for ( int j = i; j < D; ++j )
        {
            const T v = detail::jointRawMoment2< N, M, T >( centered[std::size_t( i )],
                                                            centered[std::size_t( j )] );
            out( i, j ) = v;
            out( j, i ) = v;
        }
    return out;
}

/// Third joint central-moment tensor `S_ijk = E[(F_i-mu_i)(F_j-mu_j)(F_k-mu_k)]`
/// of a vector map, `x ~ N(0, I)` i.i.d., returned as a fully-symmetric
/// fixed-size `Eigen::TensorFixedSize<T, Sizes<D, D, D>>`: `skewnessTensor(F)(i, j, k) == S_ijk`.
template < typename Derived >
    requires( detail::MomentsCompatibleTE< typename Derived::Scalar > )
[[nodiscard]] auto skewnessTensor( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T = typename detail::te_traits< TE >::scalar_type;
    constexpr int N = detail::te_traits< TE >::order_v;
    constexpr int M = detail::te_traits< TE >::vars_v;
    static_assert( Derived::SizeAtCompileTime != Eigen::Dynamic,
                   "skewnessTensor: the map's dimension D must be known at compile time" );
    constexpr int D = Derived::SizeAtCompileTime;

    const auto mu = mean( F );
    std::vector< Coeffs< T, N, M > > centered( static_cast< std::size_t >( D ) );
    for ( int i = 0; i < D; ++i )
        centered[std::size_t( i )] =
            detail::centeredCoeffs< N, M, T >( F.derived().coeff( i ).coefficients(), mu( i ) );

    // S is fully symmetric in (i,j,k): compute each unique sorted triple once and
    // scatter its value to every distinct permutation (next_permutation handles
    // repeated indices without double-emitting). Every tuple is a permutation of
    // exactly one sorted triple, so this writes every entry once — no setZero needed.
    Eigen::TensorFixedSize< T, Eigen::Sizes< D, D, D > > S;
    for ( int i = 0; i < D; ++i )
        for ( int j = i; j < D; ++j )
            for ( int k = j; k < D; ++k )
            {
                const T v = detail::jointRawMoment3< N, M, T >( centered[std::size_t( i )],
                                                                centered[std::size_t( j )],
                                                                centered[std::size_t( k )] );
                std::array< int, 3 > p{ i, j, k };
                do
                {
                    S( p[0], p[1], p[2] ) = v;
                } while ( std::next_permutation( p.begin(), p.end() ) );
            }
    return S;
}

/// Fourth joint central-moment tensor `K_ijkl = E[(F_i-mu_i)(F_j-mu_j)(F_k-mu_k)(F_l-mu_l)]`
/// of a vector map, `x ~ N(0, I)` i.i.d., returned as a fully-symmetric fixed-size
/// `Eigen::TensorFixedSize<T, Sizes<D, D, D, D>>`: `kurtosisTensor(F)(i, j, k, l) == K_ijkl`.
template < typename Derived >
    requires( detail::MomentsCompatibleTE< typename Derived::Scalar > )
[[nodiscard]] auto kurtosisTensor( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T = typename detail::te_traits< TE >::scalar_type;
    constexpr int N = detail::te_traits< TE >::order_v;
    constexpr int M = detail::te_traits< TE >::vars_v;
    static_assert( Derived::SizeAtCompileTime != Eigen::Dynamic,
                   "kurtosisTensor: the map's dimension D must be known at compile time" );
    constexpr int D = Derived::SizeAtCompileTime;

    const auto mu = mean( F );
    std::vector< Coeffs< T, N, M > > centered( static_cast< std::size_t >( D ) );
    for ( int i = 0; i < D; ++i )
        centered[std::size_t( i )] =
            detail::centeredCoeffs< N, M, T >( F.derived().coeff( i ).coefficients(), mu( i ) );

    // K is fully symmetric in (i,j,k,l): compute each unique sorted quadruple once
    // and scatter its value to every distinct permutation. Every tuple is a
    // permutation of exactly one sorted quadruple, so this writes every entry once.
    Eigen::TensorFixedSize< T, Eigen::Sizes< D, D, D, D > > K;
    for ( int i = 0; i < D; ++i )
        for ( int j = i; j < D; ++j )
            for ( int k = j; k < D; ++k )
                for ( int l = k; l < D; ++l )
                {
                    const T v = detail::jointRawMoment4< N, M, T >(
                        centered[std::size_t( i )], centered[std::size_t( j )],
                        centered[std::size_t( k )], centered[std::size_t( l )] );
                    std::array< int, 4 > p{ i, j, k, l };
                    do
                    {
                        K( p[0], p[1], p[2], p[3] ) = v;
                    } while ( std::next_permutation( p.begin(), p.end() ) );
                }
    return K;
}

/// Excess-kurtosis tensor: `kurtosisTensor(F)` minus the jointly-Gaussian
/// reference baseline `Cov_ij Cov_kl + Cov_ik Cov_jl + Cov_il Cov_jk`
/// (Isserlis'/Wick's theorem for four jointly Gaussian variables) — a standard
/// measure of how far `F`'s output distribution departs from Gaussian.
template < typename Derived >
    requires( detail::MomentsCompatibleTE< typename Derived::Scalar > )
[[nodiscard]] auto excessKurtosisTensor( const Eigen::MatrixBase< Derived >& F )
{
    constexpr int D = Derived::SizeAtCompileTime;
    auto K = kurtosisTensor( F );
    const auto C = covariance( F );
    for ( int i = 0; i < D; ++i )
        for ( int j = 0; j < D; ++j )
            for ( int k = 0; k < D; ++k )
                for ( int l = 0; l < D; ++l )
                    K( i, j, k, l ) -=
                        C( i, j ) * C( k, l ) + C( i, k ) * C( j, l ) + C( i, l ) * C( j, k );
    return K;
}

/// Per-component (marginal) standardized skewness coefficient
/// `gamma1_i = E[(F_i-mu_i)^3] / Var(F_i)^{3/2}` (Fisher's skewness), one scalar
/// per output dimension, returned as a fixed-size `D x 1` Eigen vector. This is
/// the diagonal of `skewnessTensor(F)` normalized by `sigma_i^3` — cheaper to
/// obtain directly, as here. Zero for any symmetric marginal; a zero-variance
/// (constant) component yields a non-finite value (skewness is undefined there).
template < typename Derived >
    requires( detail::MomentsCompatibleTE< typename Derived::Scalar > )
[[nodiscard]] auto skewness( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T = typename detail::te_traits< TE >::scalar_type;
    constexpr int N = detail::te_traits< TE >::order_v;
    constexpr int M = detail::te_traits< TE >::vars_v;
    static_assert( Derived::SizeAtCompileTime != Eigen::Dynamic,
                   "skewness: the map's dimension D must be known at compile time" );
    constexpr int D = Derived::SizeAtCompileTime;

    const auto mu = mean( F );
    Eigen::Matrix< T, D, 1 > out;
    for ( int i = 0; i < D; ++i )
    {
        const Coeffs< T, N, M > c =
            detail::centeredCoeffs< N, M, T >( F.derived().coeff( i ).coefficients(), mu( i ) );
        const T var = detail::jointRawMoment2< N, M, T >( c, c );
        const T m3 = detail::jointRawMoment3< N, M, T >( c, c, c );
        using std::sqrt;
        out( i ) = m3 / ( var * sqrt( var ) );
    }
    return out;
}

/// Per-component (marginal) standardized kurtosis `E[(F_i-mu_i)^4] / Var(F_i)^2`
/// (Pearson kurtosis, `== 3` for a Gaussian marginal), one scalar per output
/// dimension, returned as a fixed-size `D x 1` Eigen vector. For the departure
/// from the Gaussian value use `excessKurtosis` below.
template < typename Derived >
    requires( detail::MomentsCompatibleTE< typename Derived::Scalar > )
[[nodiscard]] auto kurtosis( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T = typename detail::te_traits< TE >::scalar_type;
    constexpr int N = detail::te_traits< TE >::order_v;
    constexpr int M = detail::te_traits< TE >::vars_v;
    static_assert( Derived::SizeAtCompileTime != Eigen::Dynamic,
                   "kurtosis: the map's dimension D must be known at compile time" );
    constexpr int D = Derived::SizeAtCompileTime;

    const auto mu = mean( F );
    Eigen::Matrix< T, D, 1 > out;
    for ( int i = 0; i < D; ++i )
    {
        const Coeffs< T, N, M > c =
            detail::centeredCoeffs< N, M, T >( F.derived().coeff( i ).coefficients(), mu( i ) );
        const T var = detail::jointRawMoment2< N, M, T >( c, c );
        const T m4 = detail::jointRawMoment4< N, M, T >( c, c, c, c );
        out( i ) = m4 / ( var * var );
    }
    return out;
}

/// Per-component (marginal) excess kurtosis `kurtosis(F) - 3` (Fisher's excess
/// kurtosis, `== 0` for a Gaussian marginal), one scalar per output dimension,
/// returned as a fixed-size `D x 1` Eigen vector — a compact non-Gaussianity
/// diagnostic per output coordinate.
template < typename Derived >
    requires( detail::MomentsCompatibleTE< typename Derived::Scalar > )
[[nodiscard]] auto excessKurtosis( const Eigen::MatrixBase< Derived >& F )
{
    using T = typename detail::te_traits< typename Derived::Scalar >::scalar_type;
    auto out = kurtosis( F );
    for ( Eigen::Index i = 0; i < out.size(); ++i ) out( i ) -= T( 3 );
    return out;
}

}  // namespace tax::la
