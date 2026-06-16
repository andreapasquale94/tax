// include/tax/stats/gaussian.hpp
//
// GaussianMoments<T, M>: raw moments of a zero-mean multivariate normal
// distribution N(0, Sigma) over `M` variables.
//
// The raw moment of a monomial, E[ prod_i x_i^{alpha_i} ], is given by
// Isserlis' (Wick's) theorem: it vanishes for odd total degree, and for even
// degree equals the sum over all perfect pairings of the monomial's factors of
// the products of the corresponding covariance entries. We evaluate it with the
// equivalent linear recurrence
//
//     E[x^alpha] = sum_j (alpha'_j) * Sigma(i, j) * E[x^{alpha' - e_j}],
//
// where i is the first variable with alpha_i > 0 and alpha' = alpha - e_i. The
// recurrence is applied bottom-up over graded-lex flat order so every
// sub-moment is already available, giving an O(numMonomials * M) table build
// instead of enumerating the (2m-1)!! pairings explicitly.
//
// The expansion variables of a `TaylorExpansion` represent *deviations* from
// the nominal point (captured by the constant term), so the zero-mean model is
// the natural one for uncertainty propagation; a non-zero nominal is carried by
// the function value itself, not by this distribution.

#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <tax/core/multi_index.hpp>
#include <vector>

namespace tax::stats
{

/**
 * @brief Raw moments of a zero-mean Gaussian `N(0, Sigma)` over `M` variables.
 * @tparam T Floating-point scalar type.
 * @tparam M Number of variables (must match the TaylorExpansion's variable count).
 */
template < typename T, int M >
class GaussianMoments
{
   public:
    using scalar_type = T;
    static constexpr int vars = M;
    using Covariance = Eigen::Matrix< T, M, M >;

    /// @brief Build from a full `M x M` covariance matrix `Sigma`.
    explicit GaussianMoments( const Covariance& sigma ) noexcept : sigma_( sigma ) {}

    /// @brief Build from a diagonal covariance given the per-variable variances.
    [[nodiscard]] static GaussianMoments diagonal(
        const Eigen::Matrix< T, M, 1 >& variances ) noexcept
    {
        return GaussianMoments( Covariance( variances.asDiagonal() ) );
    }

    /// @brief The covariance matrix backing this distribution.
    [[nodiscard]] const Covariance& covariance() const noexcept { return sigma_; }

    /**
     * @brief Raw moments `E[x^alpha]` for all `|alpha| <= maxDegree`, in
     *        graded-lex flat order (`numMonomials(maxDegree, M)` entries).
     */
    [[nodiscard]] std::vector< T > momentTable( int maxDegree ) const
    {
        const std::size_t n = numMonomials( maxDegree, M );
        std::vector< T > table( n, T( 0 ) );
        if ( n == 0 ) return table;
        table[0] = T( 1 );  // E[1] = 1
        for ( std::size_t k = 1; k < n; ++k )
        {
            MultiIndex< M > alpha = unflatIndex< M >( k );
            if ( ( totalDegree( alpha ) & 1 ) != 0 ) continue;  // odd order -> 0

            // First variable still present; remove one of its factors.
            std::size_t i = 0;
            while ( alpha[i] == 0 ) ++i;
            MultiIndex< M > ap = alpha;
            ap[i] -= 1;

            // Pair that factor with one factor of each remaining variable j.
            T acc = T( 0 );
            for ( std::size_t j = 0; j < std::size_t( M ); ++j )
            {
                if ( ap[j] == 0 ) continue;
                MultiIndex< M > ar = ap;
                ar[j] -= 1;
                acc += T( ap[j] ) * sigma_( int( i ), int( j ) ) * table[flatIndex< M >( ar )];
            }
            table[k] = acc;
        }
        return table;
    }

    /// @brief Single raw moment `E[x^alpha]` (convenience; prefer `momentTable`).
    [[nodiscard]] T rawMoment( const MultiIndex< M >& alpha ) const
    {
        const std::vector< T > table = momentTable( totalDegree( alpha ) );
        return table[flatIndex< M >( alpha )];
    }

   private:
    Covariance sigma_;
};

}  // namespace tax::stats
