// include/tax/stats/concepts.hpp
//
// Concept describing a moment provider: an object that supplies the raw
// (non-central) moments E[ x^alpha ] of an input distribution over `M`
// variables, indexed in the same graded-lexicographic flat order used by
// `TaylorExpansion`. The statistics free functions (expectation, covariance,
// ...) are generic over any type satisfying this concept; `GaussianMoments`
// is the concrete implementation shipped here.

#pragma once

#include <concepts>
#include <type_traits>
#include <vector>

namespace tax::stats
{

/**
 * @brief A source of raw statistical moments for an `M`-variate distribution.
 *
 * A model `P` must expose:
 *   - `typename P::scalar_type` — the floating-point scalar type;
 *   - `static constexpr int P::vars` — the number of variables `M`;
 *   - `std::vector<scalar_type> P::momentTable(int maxDegree) const` —
 *     the raw moments `E[ x^alpha ]` for every multi-index `alpha` with total
 *     degree `<= maxDegree`, laid out by `flatIndex<M>(alpha)` (so the result
 *     has `numMonomials(maxDegree, M)` entries).
 */
template < typename P >
concept MomentProvider = requires {
    typename P::scalar_type;
    requires std::is_convertible_v< decltype( P::vars ), int >;
} && requires( const P& p, int d ) {
    { p.momentTable( d ) } -> std::convertible_to< std::vector< typename P::scalar_type > >;
};

}  // namespace tax::stats
