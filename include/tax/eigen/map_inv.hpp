#pragma once

#include <Eigen/Dense>
#include <tax/da.hpp>
#include <tax/eigen/tensor_function.hpp>
#include <tax/eigen/types.hpp>
#include <tax/kernels.hpp>

namespace tax
{

namespace detail
{

/**
 * @brief Compose a single polynomial with a polynomial map (substitution).
 * @details Given `p(x_1, ..., x_M)` and a map `g = (g_1, ..., g_M)`, computes
 *          `p(g_1(y), ..., g_M(y))` truncated to order `N`.
 * @param p   Polynomial to evaluate.
 * @param g   Map whose components replace the variables.
 * @return    Composed polynomial `p ∘ g`.
 */
template < typename T, int N, int M, int K >
[[nodiscard]] TDA< T, N, K >
composePoly( const TDA< T, N, M >& p,
             const Eigen::Matrix< TDA< T, N, K >, M, 1 >& g ) noexcept
{
    using coeff_array = typename TDA< T, N, K >::coeff_array;
    constexpr auto S = TDA< T, N, K >::ncoef;

    // Precompute powers: gpow[j][k] = g_j^k as coefficient arrays.
    std::array< std::array< coeff_array, N + 1 >, M > gpow{};
    for ( int j = 0; j < M; ++j )
    {
        gpow[j][0]    = {};
        gpow[j][0][0] = T{ 1 };
        gpow[j][1]    = g( j ).coeffs();
        for ( int k = 2; k <= N; ++k )
            cauchyProduct< T, N, K >( gpow[j][k], gpow[j][k - 1], gpow[j][1] );
    }

    // Accumulate: out = Σ_α  p_α · Π_j gpow[j][α_j]
    coeff_array out{};
    const auto& pc = p.coeffs();

    for ( int d = 0; d <= N; ++d )
    {
        forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
            const T coeff = pc[ai];
            if ( coeff == T{} ) return;

            // Build monomial product Π_j gpow[j][α_j], skipping trivial g^0 = 1.
            coeff_array mono{};
            mono[0]       = T{ 1 };
            bool first_nz = true;
            for ( int j = 0; j < M; ++j )
            {
                if ( alpha[j] == 0 ) continue;
                if ( first_nz )
                {
                    mono     = gpow[j][alpha[j]];
                    first_nz = false;
                } else
                {
                    coeff_array tmp{};
                    cauchyProduct< T, N, K >( tmp, mono, gpow[j][alpha[j]] );
                    mono = tmp;
                }
            }

            // Accumulate coeff * mono
            for ( std::size_t k = 0; k < S; ++k ) out[k] += coeff * mono[k];
        } );
    }

    return TDA< T, N, K >( out );
}

/**
 * @brief Compose a polynomial map with another polynomial map.
 * @details Computes `(f ∘ g)(y) = (f_1(g(y)), ..., f_M(g(y)))`.
 */
template < typename T, int N, int M, int K, int Size >
[[nodiscard]] Eigen::Matrix< TDA< T, N, K >, Size, 1 >
composeMap( const Eigen::Matrix< TDA< T, N, M >, Size, 1 >& f,
            const Eigen::Matrix< TDA< T, N, K >, M, 1 >& g ) noexcept
{
    Eigen::Matrix< TDA< T, N, K >, Size, 1 > result;
    for ( int i = 0; i < Size; ++i ) result( i ) = composePoly< T, N, M, K >( f( i ), g );
    return result;
}

}  // namespace detail

/**
 * @brief Multivariate polynomial map inversion via Picard iterations.
 * @details For a map `f : R^M → R^Size` with `f(0)=0` and full-rank Jacobian `J` at
 *          the origin, returns `g : R^Size → R^M` as a formal inverse to order `N`.
 *          If `Size == M`, this is the standard two-sided local inverse.
 *          If `Size > M` (tall map), it is a left inverse (`g ∘ f = id` locally).
 *          If `Size < M` (wide map), it is a right inverse (`f ∘ g = id` locally).
 *
 *          The algorithm decomposes `f = J·x + N(x)` where `N` contains degree ≥ 2
 *          terms. It uses a generalized inverse `G` of `J`:
 *            - `G = J^{-1}` if square,
 *            - `G = (J^T J)^{-1} J^T` for tall maps,
 *            - `G = J^T (J J^T)^{-1}` for wide maps.
 *          Then solves `g(y) = G·(y − N(g(y)))` by Picard iteration.
 *
 * @tparam T Scalar type.
 * @tparam N Maximum polynomial order.
 * @tparam M    Number of source variables.
 * @tparam Size Number of map outputs.
 * @param f     Polynomial map with `f(0)=0` and full-rank Jacobian.
 * @return      Formal inverse map `g : R^Size → R^M`.
 */
template < typename T, int N, int M, int Size >
    requires( M > 1 )
[[nodiscard]] Eigen::Matrix< TDA< T, N, Size >, M, 1 >
inv( const Eigen::Matrix< TDA< T, N, M >, Size, 1 >& f ) noexcept
{
    using DAIn       = TDA< T, N, M >;
    using DAOut      = TDA< T, N, Size >;
    using InVec      = Eigen::Matrix< DAIn, Size, 1 >;
    using MidVec     = Eigen::Matrix< DAOut, Size, 1 >;
    using OutVec     = Eigen::Matrix< DAOut, M, 1 >;
    using coeff_array = typename DAOut::coeff_array;
    constexpr auto S = DAOut::ncoef;

    // 1. Extract Jacobian J_{ij} = ∂f_i/∂x_j at origin.
    const Eigen::Matrix< T, Size, M > J = jacobian( f );

    // 2. Compute a generalized inverse of J via linear solves
    Eigen::Matrix< T, M, Size > G;
    if constexpr ( Size == M )
    {
        G = J.partialPivLu().solve( Eigen::Matrix< T, M, M >::Identity() );
    } else if constexpr ( Size > M )
    {
        const Eigen::Matrix< T, M, M > JtJ = J.transpose() * J;
        G = JtJ.ldlt().solve( J.transpose() );
    } else
    {
        const Eigen::Matrix< T, Size, Size > JJt = J * J.transpose();
        const Eigen::Matrix< T, Size, M > Gt   = JJt.ldlt().solve( J );
        G = Gt.transpose();
    }

    // 3. Extract nonlinear part: N_i(x) = f_i(x) − Σ_j J_{ij} x_j
    InVec N_nl;
    for ( int i = 0; i < Size; ++i )
    {
        auto nc = f( i ).coeffs();
        nc[0]          = T{ 0 };  // clear constant term
        for ( int j = 0; j < M; ++j )
        {
            MultiIndex< M > ej{};
            ej[j] = 1;
            nc[detail::flatIndex< M >( ej )] -= J( i, j );
        }
        N_nl( i ) = DAIn( nc );
    }

    // 4. Initial guess: g_0 = G · y.
    OutVec g;
    for ( int i = 0; i < M; ++i )
    {
        coeff_array c{};
        for ( int j = 0; j < Size; ++j )
        {
            MultiIndex< Size > ej{};
            ej[j] = 1;
            c[detail::flatIndex< Size >( ej )] = G( i, j );
        }
        g( i ) = DAOut( c );
    }

    if constexpr ( N <= 1 ) return g;

    // 5. Picard iterations: g_{k+1} = G · (y − N(g_k)).
    //    Each iteration adds one correct degree; N−1 iterations needed for degree N.
    for ( int iter = 0; iter < N - 1; ++iter )
    {
        MidVec Ng = detail::composeMap< T, N, M, Size, Size >( N_nl, g );

        // rhs_i = y_i − Ng_i  (y_i = i-th coordinate variable)
        MidVec rhs;
        for ( int i = 0; i < Size; ++i )
        {
            coeff_array c{};
            MultiIndex< Size > ei{};
            ei[i] = 1;
            c[detail::flatIndex< Size >( ei )] = T{ 1 };
            const auto& ngc = Ng( i ).coeffs();
            for ( std::size_t k = 0; k < S; ++k ) c[k] -= ngc[k];
            rhs( i ) = DAOut( c );
        }

        // g = G · rhs  (matrix-vector multiply in coefficient space)
        for ( int i = 0; i < M; ++i )
        {
            coeff_array c{};
            for ( int j = 0; j < Size; ++j )
            {
                const auto& rc = rhs( j ).coeffs();
                for ( std::size_t k = 0; k < S; ++k ) c[k] += G( i, j ) * rc[k];
            }
            g( i ) = DAOut( c );
        }
    }

    return g;
}

/**
 * @brief Component-wise univariate series inversion on a DA vector.
 * @details Applies `inv(f_i)` independently to each component.
 *          Each `f_i` must satisfy `f_i(0) = 0` and `f_i'(0) ≠ 0`.
 *
 * @tparam N    Taylor order.
 * @tparam Size Vector dimension (compile-time).
 * @param f     Vector of univariate DA polynomials.
 * @return      Vector where each component is the compositional inverse of the input.
 */
template < int N, int Size >
[[nodiscard]] Eigen::Matrix< DA< N >, Size, 1 >
inv( const Eigen::Matrix< DA< N >, Size, 1 >& f ) noexcept
{
    using coeff_array = typename DA< N >::coeff_array;
    Eigen::Matrix< DA< N >, Size, 1 > result;
    for ( Eigen::Index i = 0; i < f.size(); ++i )
    {
        coeff_array out{};
        detail::seriesInv< double, N, 1 >( out, f( i ).coeffs() );
        result( i ) = DA< N >( out );
    }
    return result;
}

}  // namespace tax
