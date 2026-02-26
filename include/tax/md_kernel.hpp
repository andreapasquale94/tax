#pragma once

#include "tax/index.hpp"
#include "tax/utils.hpp"

namespace tax::detail
{

template < typename T, std::size_t N, std::size_t D, std::size_t M = countMonomials< N, D >() >
constexpr void seriesProductND( std::array< T, M >& out, const std::array< T, M >& a,
                                const std::array< T, M >& b ) noexcept
{
    using IdxSet = IndexSet< N, D >;

    // Iterate by degree blocks to enforce truncation and improve locality
    for ( std::size_t da = 0; da <= N; ++da )
    {
        const std::size_t ia0 = IdxSet::offset[da];
        const std::size_t ia1 = IdxSet::offset[da + 1];

        for ( std::size_t db = 0; db + da <= N; ++db )
        {
            const std::size_t ib0 = IdxSet::offset[db];
            const std::size_t ib1 = IdxSet::offset[db + 1];

            for ( std::size_t i = ia0; i < ia1; ++i )
            {
                const T ai = a[i];
                // Sparsity shortcut: skip if coefficient is exactly zero
                if ( ai == T{} ) continue;
                const auto& alpha = IdxSet::data[i];

                for ( std::size_t j = ib0; j < ib1; ++j )
                {
                    const T bj = b[j];
                    if ( bj == T{} ) continue;
                    const auto& beta = IdxSet::data[j];

                    std::array< std::size_t, D > gamma;
                    for ( std::size_t k = 0; k < D; ++k ) gamma[k] = alpha[k] + beta[k];
                    out[IdxSet::indexOf( gamma )] += ai * bj;
                }
            }
        }
    }
}

template < typename T, std::size_t N, std::size_t D, std::size_t M = countMonomials< N, D >() >
constexpr void seriesDivisionND( std::array< T, M >& out, const std::array< T, M >& num,
                                 const std::array< T, M >& den ) noexcept
{
    using IdxSet = IndexSet< N, D >;

    // Preconditions: den[0] != 0
    out = {};
    out[0] = num[0] / den[0];

    for ( std::size_t g = 1; g < M; ++g )
    {
        const auto& gamma = IdxSet::data[g];
        const std::size_t dg = IdxSet::degreeOf( gamma );

        T s{};

        // sum over degrees of beta (exclude beta = 0 by starting at 1)
        for ( std::size_t db = 1; db <= dg; ++db )
        {
            const std::size_t ib0 = IdxSet::offset[db];
            const std::size_t ib1 = IdxSet::offset[db + 1];

            for ( std::size_t j = ib0; j < ib1; ++j )
            {
                const T bj = den[j];
                if ( bj == T{} ) continue;

                const auto& beta = IdxSet::data[j];
                if ( !leq< D >( beta, gamma ) ) continue;

                std::array< std::size_t, D > diff;
                for ( std::size_t k = 0; k < D; ++k ) diff[k] = gamma[k] - beta[k];

                s += bj * out[IdxSet::indexOf( diff )];
            }
        }

        out[g] = ( num[g] - s ) / den[0];
    }
}

template < typename T, std::size_t N, std::size_t D, std::size_t M = countMonomials< N, D >() >
constexpr void seriesReciprocalND( std::array< T, M >& out, const std::array< T, M >& a ) noexcept
{
    using IdxSet = IndexSet< N, D >;

    out = {};
    out[0] = T{ 1 } / a[0];

    for ( std::size_t g = 1; g < M; ++g )
    {
        const auto& gamma = IdxSet::data[g];
        const std::size_t dg = IdxSet::degreeOf( gamma );
        T s{};

        for ( std::size_t db = 1; db <= dg; ++db )
        {
            const std::size_t ib0 = IdxSet::offset[db];
            const std::size_t ib1 = IdxSet::offset[db + 1];

            for ( std::size_t j = ib0; j < ib1; ++j )
            {
                const T aj = a[j];
                if ( aj == T{} ) continue;
                const auto& beta = IdxSet::data[j];
                if ( !leq< D >( beta, gamma ) ) continue;

                std::array< std::size_t, D > diff;
                for ( std::size_t k = 0; k < D; ++k ) diff[k] = gamma[k] - beta[k];
                s += aj * out[IdxSet::indexOf( diff )];
            }
        }
        out[g] = -s / a[0];
    }
}

}  // namespace tax::detail
