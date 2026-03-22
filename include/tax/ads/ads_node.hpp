#pragma once

#include <tax/ads/box.hpp>
#include <tuple>
#include <variant>

namespace tax
{

/**
 * @brief A node in the ADS arena tree.
 *
 * Each node is either a **Leaf** (holds a TTE and its subdomain box) or an
 * **Internal** node (produced when a leaf is split, holds split metadata and
 * the indices of the two children in the arena).
 *
 * @tparam TTE  Truncated Taylor expansion type (e.g. tax::TEn<N, M>).
 */
template < typename TTE >
struct AdsNode
{
    using T                = typename TTE::Input::value_type;
    static constexpr int M = int( std::tuple_size_v< typename TTE::Input > );

    // -------------------------------------------------------------------------
    // Leaf: the TTE polynomial and the subdomain it is defined over.
    // -------------------------------------------------------------------------
    struct Leaf
    {
        TTE        tte;
        Box< T, M > box;
        bool       done       = false;  ///< true once this leaf has been processed
        int        leaves_pos = -1;     ///< index of this node in AdsTree::leaf_list_
    };

    // -------------------------------------------------------------------------
    // Internal: produced when a leaf is split.
    // -------------------------------------------------------------------------
    struct Internal
    {
        int split_dim;    ///< dimension that was split
        T   split_value;  ///< boundary value (= center[dim] of the former leaf)
        int left_idx;     ///< arena index of the left  (lower) child
        int right_idx;    ///< arena index of the right (upper) child
    };

    // -------------------------------------------------------------------------
    int                         parent_idx = -1;  ///< -1 for root nodes
    std::variant< Leaf, Internal > data;

    // -- Type queries ---------------------------------------------------------

    [[nodiscard]] bool is_leaf()     const noexcept
    {
        return std::holds_alternative< Leaf >( data );
    }
    [[nodiscard]] bool is_internal() const noexcept
    {
        return std::holds_alternative< Internal >( data );
    }

    // -- Accessors ------------------------------------------------------------

    [[nodiscard]] Leaf&          leaf()           noexcept { return std::get< Leaf >( data ); }
    [[nodiscard]] const Leaf&    leaf()     const noexcept { return std::get< Leaf >( data ); }
    [[nodiscard]] Internal&      internal()       noexcept { return std::get< Internal >( data ); }
    [[nodiscard]] const Internal& internal() const noexcept
    {
        return std::get< Internal >( data );
    }
};

}  // namespace tax
