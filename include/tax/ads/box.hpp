#pragma once

#include <array>
#include <cmath>
#include <utility>

namespace tax
{

/**
 * @brief Axis-aligned hyperrectangle in M dimensions.
 *
 * Represented as the Cartesian product of closed intervals
 *   [center[i] - half_width[i],  center[i] + half_width[i]]
 * for each dimension i.
 *
 * @tparam T  Scalar type.
 * @tparam M  Number of dimensions.
 */
template < typename T, int M >
struct Box
{
    static_assert( M >= 1, "Box dimension must be at least 1" );

    std::array< T, M > center{};
    std::array< T, M > half_width{};

    /// Returns true if @p pt lies inside (or on the boundary of) this box.
    [[nodiscard]] bool contains( const std::array< T, M >& pt ) const noexcept
    {
        using std::abs;
        for ( int i = 0; i < M; ++i )
            if ( abs( pt[i] - center[i] ) > half_width[i] )
                return false;
        return true;
    }

    /**
     * @brief Split the box at its midpoint along dimension @p dim.
     * @return {left_child, right_child}
     *   left_child  has smaller center[dim]  (lower half)
     *   right_child has larger  center[dim]  (upper half)
     */
    [[nodiscard]] std::pair< Box, Box > split( int dim ) const noexcept
    {
        Box left  = *this;
        Box right = *this;
        const T hw        = half_width[dim] * T{ 0.5 };
        left.half_width[dim]  = hw;
        right.half_width[dim] = hw;
        left.center[dim]  -= hw;
        right.center[dim] += hw;
        return { left, right };
    }

    /// Returns the split boundary value for @p dim (i.e. center[dim]).
    [[nodiscard]] T split_value( int dim ) const noexcept { return center[dim]; }
};

}  // namespace tax
