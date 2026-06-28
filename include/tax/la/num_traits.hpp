// Eigen::NumTraits<TaylorExpansion> (makes a TE a first-class Eigen scalar) and
// the tax::la::detail traits used across la/ (te_traits, is_te, rebind_matrix_t).

#pragma once

#include <Eigen/Core>
#include <tax/core/expansion.hpp>
#include <type_traits>

// -----------------------------------------------------------------------------
// NumTraits specialization — namespace Eigen
// -----------------------------------------------------------------------------

namespace Eigen
{

template < typename T, typename Basis, typename Scheme, typename Storage >
struct NumTraits< tax::Expansion< T, Basis, Scheme, Storage > > : NumTraits< T >
{
    using Self = tax::Expansion< T, Basis, Scheme, Storage >;
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

// -----------------------------------------------------------------------------
// Internal traits — namespace tax::la::detail
// -----------------------------------------------------------------------------

namespace tax::la::detail
{

template < typename >
struct te_traits;

template < typename T, typename Basis, typename Scheme, typename S >
struct te_traits< Expansion< T, Basis, Scheme, S > >
{
    using scalar_type = T;
    static constexpr int order_v = Scheme::order;
    static constexpr int vars_v = Scheme::vars;
    using scheme_t = Scheme;
    using storage_t = S;
    using basis_t = Basis;
};

template < typename T >
struct is_te : std::false_type
{
};

template < typename T, typename Basis, typename Scheme, typename S >
struct is_te< Expansion< T, Basis, Scheme, S > > : std::true_type
{
};

/// Any tax `Expansion` (used by basis-generic la helpers: variables/value/eval).
template < typename T >
inline constexpr bool is_te_v = is_te< T >::value;

/// A `TaylorExpansion` specifically — gate for the Taylor-only value semantics
/// (k!-scaled derivative *values*, expansion-point gradient/Hessian/Jacobian).
template < typename E >
inline constexpr bool is_taylor_te_v =
    is_te_v< E > && std::is_same_v< typename te_traits< E >::basis_t, TaylorBasis >;

/// Rebind the scalar type of an Eigen matrix expression.
template < typename Derived, typename Scalar >
using rebind_matrix_t =
    Eigen::Matrix< Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime, Derived::Options,
                   Derived::MaxRowsAtCompileTime, Derived::MaxColsAtCompileTime >;

}  // namespace tax::la::detail
