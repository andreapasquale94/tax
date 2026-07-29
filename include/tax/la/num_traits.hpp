// Eigen::NumTraits<TaylorExpansion> (makes a TE a first-class Eigen scalar) and
// the tax::la::detail traits used across la/ (te_traits, is_te, rebind_matrix_t).

#pragma once

#include <Eigen/Core>
#include <tax/core/taylor_expansion.hpp>
#include <type_traits>

namespace Eigen
{

template < typename T, typename Scheme >
struct NumTraits< tax::TaylorExpansion< T, Scheme > > : NumTraits< T >
{
    using Self = tax::TaylorExpansion< T, Scheme >;
    using Real = Self;
    using NonInteger = Self;
    using Nested = Self;

    static constexpr int kNc = int( Scheme::nCoeff );

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = kNc,
        AddCost = kNc,
        // kNc * kNc overflows int for kNc > ~46340; clamp to HugeCost (Eigen's
        // "very expensive scalar" sentinel) so the cost stays a valid int.
        MulCost = kNc < 46341 ? kNc * kNc : HugeCost
    };

    // NumTraits<T> (the base) declares these returning the scalar T, but our
    // Real is Self (the expansion). Re-expose them as constant expansions so
    // Eigen's normwise/fuzzy comparison paths see a Real-typed threshold rather
    // than relying on the implicit scalar -> constant-TE conversion.
    static inline Self epsilon() { return Self( NumTraits< T >::epsilon() ); }
    static inline Self dummy_precision() { return Self( NumTraits< T >::dummy_precision() ); }
    static inline Self highest() { return Self( NumTraits< T >::highest() ); }
    static inline Self lowest() { return Self( NumTraits< T >::lowest() ); }
    static inline Self infinity() { return Self( NumTraits< T >::infinity() ); }
    static inline Self quiet_NaN() { return Self( NumTraits< T >::quiet_NaN() ); }
};

}  // namespace Eigen

namespace tax::la::detail
{

template < typename >
struct te_traits;

template < typename T, typename Scheme >
struct te_traits< TaylorExpansion< T, Scheme > >
{
    using scalar_type = T;
    static constexpr int order_v = Scheme::order;
    static constexpr int vars_v = Scheme::vars;
    using scheme_t = Scheme;
};

template < typename T >
struct is_te : std::false_type
{
};

template < typename T, typename Scheme >
struct is_te< TaylorExpansion< T, Scheme > > : std::true_type
{
};

template < typename T >
inline constexpr bool is_te_v = is_te< T >::value;

/// Rebind the scalar type of an Eigen matrix expression.
template < typename Derived, typename Scalar >
using rebind_matrix_t =
    Eigen::Matrix< Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime, Derived::Options,
                   Derived::MaxRowsAtCompileTime, Derived::MaxColsAtCompileTime >;

}  // namespace tax::la::detail
