#pragma once

// Eigen compatibility layer for Taylor models: makes `TaylorModel` a
// first-class Eigen scalar so state vectors and the state-transition matrix
// can be `Eigen::Matrix<TM, ...>` rather than hand-rolled `std::array`s.
//
// The one wrinkle a domain-carrying scalar creates is that Eigen synthesises
// `Scalar(0)` / `Scalar(1)` literals (in `setZero`, `Identity`, reductions,
// products) with no domain. These become *domain-agnostic constants* (see the
// `TaylorModel(T)` constructor), which adopt their partner's domain in every
// binary operation, so they compose correctly with real domain-carrying
// models. Ordering-based Eigen paths (fuzzy `isApprox`, pivoted
// decompositions) are unsupported — Taylor models are not ordered.

#include <Eigen/Core>
#include <cstddef>
#include <tax/la/types.hpp>
#include <tax/model/arithmetic.hpp>
#include <tax/model/io.hpp>
#include <tax/model/math.hpp>
#include <tax/model/taylor_model.hpp>

// -----------------------------------------------------------------------------
// NumTraits specialization — namespace Eigen
// -----------------------------------------------------------------------------

namespace Eigen
{

template < std::floating_point T, int N, int M >
struct NumTraits< tax::model::TaylorModel< T, N, M > > : NumTraits< T >
{
    using Self = tax::model::TaylorModel< T, N, M >;
    using Real = Self;
    using NonInteger = Self;
    using Nested = Self;
    using Literal = Self;

    static constexpr int kNc = int( Self::nCoefficients );

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = kNc,
        AddCost = kNc,
        // kNc * kNc overflows int for large kNc; clamp to HugeCost.
        MulCost = kNc < 46341 ? kNc * kNc : HugeCost
    };

    // These build domain-agnostic constants (the TaylorModel(T) ctor), so they
    // are usable against any real model without a domain clash.
    static inline Self epsilon() { return Self( NumTraits< T >::epsilon() ); }
    static inline Self dummy_precision() { return Self( NumTraits< T >::dummy_precision() ); }
    static inline Self highest() { return Self( NumTraits< T >::highest() ); }
    static inline Self lowest() { return Self( NumTraits< T >::lowest() ); }
    static inline Self infinity() { return Self( NumTraits< T >::infinity() ); }
    static inline Self quiet_NaN() { return Self( NumTraits< T >::quiet_NaN() ); }
};

}  // namespace Eigen

namespace tax::model
{

// -----------------------------------------------------------------------------
// Convenience aliases
// -----------------------------------------------------------------------------

/// Eigen column vector of `D` Taylor models (an ODE state, a flow map, …).
template < int D, int N, int M >
using TMVec = tax::la::VecNT< D, TaylorModel< double, N, M > >;

/// Eigen `R x C` matrix of Taylor models.
template < int R, int C, int N, int M >
using TMMat = tax::la::MatNMT< R, C, TaylorModel< double, N, M > >;

// -----------------------------------------------------------------------------
// Factories / extractors on Eigen vectors of Taylor models
// -----------------------------------------------------------------------------

/// Eigen column vector of the `M` coordinate variables over the box
/// `[x0 - .., x0 + ..] = dom`, matching `TaylorModel::variables` but returning
/// an Eigen vector so it plugs straight into matrix expressions.
template < std::floating_point T, int N, int M >
[[nodiscard]] tax::la::VecNT< M, TaylorModel< T, N, M > > variables(
    const std::array< T, std::size_t( M ) >& x0,
    const std::array< Interval< T >, std::size_t( M ) >& dom )
    requires( N >= 1 )
{
    tax::la::VecNT< M, TaylorModel< T, N, M > > v;
    const auto arr = TaylorModel< T, N, M >::variables( x0, dom );
    for ( int i = 0; i < M; ++i ) v( i ) = arr[std::size_t( i )];
    return v;
}

/// Constant parts of an Eigen vector of models (the state value at x0).
template < typename Derived >
[[nodiscard]] auto value( const Eigen::MatrixBase< Derived >& v )
{
    using TM = typename Derived::Scalar;
    using T = typename TM::scalar_type;
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime > out;
    out.resize( v.rows(), v.cols() );
    for ( Eigen::Index j = 0; j < v.cols(); ++j )
        for ( Eigen::Index i = 0; i < v.rows(); ++i ) out( i, j ) = v( i, j ).value();
    return out;
}

/// Rigorous enclosure of each entry of an Eigen vector/matrix of models.
template < typename Derived >
[[nodiscard]] auto bound( const Eigen::MatrixBase< Derived >& v,
                          Bounder which = Bounder::Quadratic )
{
    using TM = typename Derived::Scalar;
    using T = typename TM::scalar_type;
    Eigen::Matrix< Interval< T >, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime > out;
    out.resize( v.rows(), v.cols() );
    for ( Eigen::Index j = 0; j < v.cols(); ++j )
        for ( Eigen::Index i = 0; i < v.rows(); ++i ) out( i, j ) = v( i, j ).bound( which );
    return out;
}

/// State-transition matrix J(i, j) = d(state_i)/d(x_j) of a flow vector, read
/// from the polynomial parts.
template < typename Derived >
[[nodiscard]] auto jacobian( const Eigen::MatrixBase< Derived >& state )
{
    using TM = typename Derived::Scalar;
    using T = typename TM::scalar_type;
    constexpr int M = TM::vars_v;
    const int D = int( state.rows() );
    Eigen::Matrix< T, Derived::RowsAtCompileTime, M > J;
    J.resize( D, M );
    for ( int i = 0; i < D; ++i )
    {
        MultiIndex< M > alpha{};
        for ( int j = 0; j < M; ++j )
        {
            alpha[std::size_t( j )] = 1;
            J( i, j ) = state( i ).polynomial().derivative( alpha );
            alpha[std::size_t( j )] = 0;
        }
    }
    return J;
}

}  // namespace tax::model
