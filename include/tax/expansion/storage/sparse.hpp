#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tax/expansion/multi_index.hpp>
#include <utility>
#include <vector>

namespace tax::storage
{

/// Flat-index type used by sparse containers (32-bit is sufficient for any practical N, M).
using flat_index_t = std::uint32_t;

/// Tag type selecting the sparse (sorted-index-pair) storage policy.
struct Sparse
{
};

/// Sparse coefficient container for a TaylorExpansion.
template < typename T, int N, int M >
class SparseContainer
{
   public:
    using value_type = T;
    static constexpr std::size_t nCoefficientsMax = numMonomials( N, M );

    constexpr SparseContainer() = default;

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Number of currently stored nonzero monomials.
    [[nodiscard]] std::size_t nnz() const noexcept { return idx_.size(); }

    /// Constant (zeroth) coefficient; returns 0 if the constant slot is absent.
    [[nodiscard]] T value() const noexcept
    {
        return ( !idx_.empty() && idx_.front() == 0 ) ? val_.front() : T{ 0 };
    }

    /// Read-only view of the sorted flat indices of all nonzero slots.
    [[nodiscard]] std::span< const flat_index_t > support() const noexcept
    {
        return { idx_.data(), idx_.size() };
    }

    /// Read-only view of the values aligned with `support()`.
    [[nodiscard]] std::span< const T > values() const noexcept
    {
        return { val_.data(), val_.size() };
    }

    /// Coefficient at flat index `k`; returns `T{0}` if the slot is absent. O(log nnz) binary search — intended for tests and inspection, not hot loops.
    [[nodiscard]] T coeffAtFlat( std::size_t k ) const noexcept
    {
        auto it = std::lower_bound( idx_.begin(), idx_.end(), flat_index_t( k ) );
        if ( it == idx_.end() || *it != flat_index_t( k ) ) return T{ 0 };
        return val_[std::size_t( it - idx_.begin() )];
    }

    // -----------------------------------------------------------------------
    // Mutation primitives
    // -----------------------------------------------------------------------

    /// Set the coefficient at flat index `k` to `v`, preserving sorted order.
    void set( std::size_t k, T v ) noexcept
    {
        auto it = std::lower_bound( idx_.begin(), idx_.end(), flat_index_t( k ) );
        if ( it != idx_.end() && *it == flat_index_t( k ) )
        {
            // Slot exists.
            const std::size_t pos = std::size_t( it - idx_.begin() );
            if ( v == T{ 0 } )
            {
                idx_.erase( idx_.begin() + std::ptrdiff_t( pos ) );
                val_.erase( val_.begin() + std::ptrdiff_t( pos ) );
            } else
            {
                val_[pos] = v;
            }
        } else if ( v != T{ 0 } )
        {
            // Slot absent — insert in sorted position.
            const std::size_t pos = std::size_t( it - idx_.begin() );
            idx_.insert( idx_.begin() + std::ptrdiff_t( pos ), flat_index_t( k ) );
            val_.insert( val_.begin() + std::ptrdiff_t( pos ), v );
        }
    }

    /// Add `v` to the coefficient at flat index `k`.
    void accumulate( std::size_t k, T v ) noexcept
    {
        if ( v == T{ 0 } ) return;

        auto it = std::lower_bound( idx_.begin(), idx_.end(), flat_index_t( k ) );
        if ( it != idx_.end() && *it == flat_index_t( k ) )
        {
            const std::size_t pos = std::size_t( it - idx_.begin() );
            val_[pos] += v;
            if ( val_[pos] == T{ 0 } )
            {
                idx_.erase( idx_.begin() + std::ptrdiff_t( pos ) );
                val_.erase( val_.begin() + std::ptrdiff_t( pos ) );
            }
        } else
        {
            // Not present — insert at sorted position.
            const std::size_t pos = std::size_t( it - idx_.begin() );
            idx_.insert( idx_.begin() + std::ptrdiff_t( pos ), flat_index_t( k ) );
            val_.insert( val_.begin() + std::ptrdiff_t( pos ), v );
        }
    }

    // -----------------------------------------------------------------------
    // Traversal
    // -----------------------------------------------------------------------

    /// Visit every nonzero in flat-index order: `fn(k, val)`. `Fn` must not mutate the container.
    template < typename Fn >
    void forEachNonzero( Fn&& fn ) const
        noexcept( noexcept( fn( std::size_t{ 0 }, std::declval< T >() ) ) )
    {
        for ( std::size_t i = 0; i < idx_.size(); ++i ) fn( std::size_t( idx_[i] ), val_[i] );
    }

    /// Merged walk over the union of `support(*this)` and `support(other)`.
    template < typename Fn >
    void forEachPair( const SparseContainer& other, Fn&& fn ) const
        noexcept( noexcept( fn( std::size_t{ 0 }, std::declval< T >(), std::declval< T >() ) ) )
    {
        std::size_t i = 0;
        std::size_t j = 0;
        while ( i < idx_.size() && j < other.idx_.size() )
        {
            const auto ia = idx_[i];
            const auto ib = other.idx_[j];
            if ( ia < ib )
            {
                fn( std::size_t( ia ), val_[i], T{ 0 } );
                ++i;
            } else if ( ib < ia )
            {
                fn( std::size_t( ib ), T{ 0 }, other.val_[j] );
                ++j;
            } else
            {
                fn( std::size_t( ia ), val_[i], other.val_[j] );
                ++i;
                ++j;
            }
        }
        for ( ; i < idx_.size(); ++i ) fn( std::size_t( idx_[i] ), val_[i], T{ 0 } );
        for ( ; j < other.idx_.size(); ++j )
            fn( std::size_t( other.idx_[j] ), T{ 0 }, other.val_[j] );
    }

    // -----------------------------------------------------------------------
    // Raw access (for kernels that build content incrementally)
    // -----------------------------------------------------------------------

    [[nodiscard]] std::vector< flat_index_t >& rawIndices() noexcept { return idx_; }
    [[nodiscard]] std::vector< T >& rawValues() noexcept { return val_; }
    [[nodiscard]] const std::vector< flat_index_t >& rawIndices() const noexcept { return idx_; }
    [[nodiscard]] const std::vector< T >& rawValues() const noexcept { return val_; }

   private:
    std::vector< flat_index_t > idx_;
    std::vector< T > val_;
};

}  // namespace tax::storage
