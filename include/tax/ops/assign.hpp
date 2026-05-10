// SPDX-License-Identifier: BSD-3-Clause
//
// Streaming assignment driver.  Both storage types delegate their
// `operator<<=` here so the same loop services static and dynamic.

#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <type_traits>

#include "tax/util/multi_index.hpp"

namespace tax::detail
{

template < class Dst, class Expr >
inline void streamingAssign( Dst& dst, const Expr& expr )
{
    static_assert( Dst::IsStatic == std::remove_cvref_t< Expr >::IsStatic,
                   "Cannot mix static and dynamic Taylor expansions in one assignment" );
    if constexpr ( Dst::IsStatic )
    {
        static_assert( Dst::OrderAtCompileTime == std::remove_cvref_t< Expr >::OrderAtCompileTime,
                       "Static Taylor order mismatch on <<=" );
        static_assert( Dst::VarsAtCompileTime == std::remove_cvref_t< Expr >::VarsAtCompileTime,
                       "Static Taylor variable count mismatch on <<=" );
    }

    const std::size_t order = dst.order();
    const std::size_t nvars = dst.nvars();
    for ( std::size_t d = 0; d <= order; ++d )
    {
        expr.advanceTo( d );
        auto out_d = dst.slice( d );
        auto in_d = expr.slice( d );
        const Eigen::Index n = static_cast< Eigen::Index >( util::degreeSize( d, nvars ) );
        for ( Eigen::Index i = 0; i < n; ++i )
        {
            out_d.coeffRef( i ) = in_d.coeff( i );
        }
    }
}

}  // namespace tax::detail
