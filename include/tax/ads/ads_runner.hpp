#pragma once

#include <tax/ads/ads_tree.hpp>
#include <tax/tte.hpp>
#include <tax/utils/combinatorics.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace tax
{

/**
 * @brief Automatic Domain Splitting (ADS) runner.
 *
 * Implements the ADS algorithm from Wittig et al. (2015) as shown in Fig. 4:
 *
 *   1. Insert initial domain in work queue.
 *   2. Pop a domain, evaluate its TTE, estimate the truncation error.
 *   3. If error < tolerance OR depth >= max_depth  →  add to result list (done).
 *   4. Otherwise  →  pick the dimension that contributes most to the
 *      highest-degree coefficients, halve the box along that dimension,
 *      evaluate the two child TTEs, and re-enqueue them.
 *   5. Repeat until the work queue is empty.
 *
 * The truncation error is estimated as the infinity norm of the degree-N
 * coefficients in the normalised variable δ ∈ [−1, 1]^M.
 *
 * @tparam N         DA order.
 * @tparam M         Number of variables.
 * @tparam F         Callable: (const TEn<N,M>& x_0, …, const TEn<N,M>& x_{M-1}) -> Expr
 *                   **Arguments must be taken by const reference.**  Expression
 *                   nodes store leaf TTEs by reference; a by-value copy of an
 *                   argument would dangle once the function returns its lazy
 *                   expression, causing undefined behaviour.
 */
template < int N, int M, typename F >
class AdsRunner
{
public:
    using TTE  = TEn< N, M >;
    using Tree = AdsTree< TTE >;

    /**
     * @param func      Function to approximate (takes M DA variables).
     * @param tolerance Maximum allowed truncation-error norm per subdomain.
     * @param max_depth Maximum number of bisections from root to any leaf.
     */
    AdsRunner( F func, double tolerance, int max_depth = 30 )
        : func_( std::move( func ) ), tol_( tolerance ), max_depth_( max_depth )
    {
    }

    Tree run( Box< double, M > initial_box )
    {
        Tree tree;
        tree.add_leaf( evaluate( initial_box ), initial_box );

        // depth[node_idx] = number of splits from the root to that node.
        std::vector< int > depth( 1, 0 );

        while ( !tree.empty() )
        {
            const int    idx = tree.pop();
            const double err = truncation_error( tree.node( idx ).leaf().tte );
            const int    d   = depth[idx];

            if ( err < tol_ || d >= max_depth_ )
            {
                tree.mark_done( idx );
            }
            else
            {
                const auto& box = tree.node( idx ).leaf().box;
                const int   dim = best_split_dim( tree.node( idx ).leaf().tte );

                auto [lb, rb] = box.split( dim );
                auto lt       = evaluate( lb );
                auto rt       = evaluate( rb );

                auto [li, ri] = tree.split( idx, dim, std::move( lt ), std::move( rt ) );

                if ( static_cast< int >( depth.size() ) <= ri )
                    depth.resize( ri + 1, 0 );
                depth[li] = d + 1;
                depth[ri] = d + 1;
            }
        }
        return tree;
    }

private:
    F      func_;
    double tol_;
    int    max_depth_;

    /**
     * Evaluate func_ on @p box by creating M scaled DA variables:
     *   x_k = center_k + half_width_k * δ_k,  δ_k ∈ [−1, 1].
     *
     * Each variable is built directly from its coefficient array to avoid any
     * implicit scalar conversion in the expression-template layer.
     */
    TTE evaluate( const Box< double, M >& box )
    {
        // Build x_k = center[k] + half_width[k]*δ_k for each k.
        auto make_var = [&]( std::size_t k ) -> TTE {
            typename TTE::Data c{};
            c[0] = box.center[k];
            if constexpr ( N >= 1 )
            {
                MultiIndex< M > ek{};
                ek[k] = 1;
                c[detail::flatIndex< M >( ek )] = box.half_width[k];
            }
            return TTE{ c };
        };

        auto xs = [&]< std::size_t... I >( std::index_sequence< I... > ) {
            return std::tuple< decltype( make_var( I ) )... >{ make_var( I )... };
        }( std::make_index_sequence< M >{} );

        // Materialise while xs (and the TTEs it holds) are still alive.
        // Expression nodes store leaf TTEs by const-ref, so we must call
        // evalTo *before* xs goes out of scope.
        typename TTE::Data c{};
        std::apply( func_, xs ).evalTo( c );
        return TTE{ c };
    }

    /**
     * Truncation-error estimate: infinity norm of the degree-N coefficients.
     * These are the highest-order terms retained in the polynomial; their
     * magnitude indicates how quickly the series is (or is not) converging.
     */
    static double truncation_error( const TTE& tte ) noexcept
    {
        double err = 0.0;
        for ( std::size_t i = 0; i < TTE::nCoefficients; ++i )
        {
            const auto alpha = detail::unflatIndex< M >( i );
            if ( detail::totalDegree< M >( alpha ) == N )
                err = std::max( err, std::abs( tte[i] ) );
        }
        return err;
    }

    /**
     * Choose the split dimension: the variable whose degree-N coefficients
     * have the largest combined absolute value, i.e. the variable that
     * "most affects" the truncation error (Wittig et al. 2015).
     */
    static int best_split_dim( const TTE& tte ) noexcept
    {
        std::array< double, M > scores{};
        for ( std::size_t i = 0; i < TTE::nCoefficients; ++i )
        {
            const auto alpha = detail::unflatIndex< M >( i );
            if ( detail::totalDegree< M >( alpha ) == N )
                for ( int k = 0; k < M; ++k )
                    if ( alpha[k] > 0 )
                        scores[k] += std::abs( tte[i] );
        }
        return static_cast< int >(
            std::max_element( scores.begin(), scores.end() ) - scores.begin() );
    }
};

/// Convenience factory — deduces N, M, F from arguments.
template < int N, int M, typename F >
AdsRunner< N, M, F > make_ads_runner( F func, double tol, int max_depth = 30 )
{
    return AdsRunner< N, M, F >( std::move( func ), tol, max_depth );
}

}  // namespace tax
