#pragma once

#include <tax/eigen/adapters.hpp>
#include <tax/tax.hpp>

#include <cassert>
#include <type_traits>
#include <utility>

namespace tax::ode
{

namespace detail::taylor
{

template < typename Vec >
inline void assertColumnVector( const Vec& y ) noexcept
{
    if constexpr ( Vec::ColsAtCompileTime != Eigen::Dynamic )
        static_assert( Vec::ColsAtCompileTime == 1, "Vec must be an Eigen column vector." );
    assert( y.cols() == 1 && "Vec must be an Eigen column vector." );
}

}  // namespace detail::taylor

/**
 * @brief Builder/evaluator of local Taylor expansions for ODE states.
 *
 * @tparam N   Taylor order.
 * @tparam RHS Callable with signature compatible with
 *             `VecT<TE<N>, Rows> rhs(TE<N> t, VecT<TE<N>, Rows> y)`.
 */
template < int N, typename RHS >
class TaylorStepper
{
  public:
    static_assert( N >= 1, "Taylor order N must be at least 1." );

    using da_type = TE< N >;
    using expansion_traits = ::tax::detail::expansion_traits< da_type >;
    using scalar_type = typename expansion_traits::scalar_type;

    using rhs_type = RHS;
    static constexpr int order = expansion_traits::order;
    static constexpr int vars  = expansion_traits::vars;
    static_assert( vars == 1, "TaylorStepper currently requires univariate DA scalars." );

    explicit TaylorStepper( RHS rhs ) noexcept : rhs_( std::move( rhs ) ) {}

    [[nodiscard]] RHS& rhs() noexcept { return rhs_; }
    [[nodiscard]] const RHS& rhs() const noexcept { return rhs_; }

    /**
     * @brief Build the order-N local Taylor state polynomial coefficients.
     *
     * @tparam Vec Eigen column-vector state type.
     * @param t0 Current time.
     * @param y0 Current state.
     * @return DA-vector series of y(t0 + dt).
     */
    template < typename Vec >
    [[nodiscard]] ::tax::detail::rebind_matrix_t< Vec, da_type >
    series( double t0, const Vec& y0 )
    {
        detail::taylor::assertColumnVector( y0 );

        using DVec = ::tax::detail::rebind_matrix_t< Vec, da_type >;

        da_type t_da = da_type::variable( static_cast< scalar_type >( t0 ) );

        // Initialise each component as a constant DA polynomial at the expansion point.
        DVec y_da = y0.unaryExpr( []( const typename Vec::Scalar v ) -> da_type {
            return da_type( static_cast< scalar_type >( v ) );
        } );

        // ODE recurrence: c_{k+1}(y_i) = c_k(f_i) / (k+1)
        // coeffRow gathers the k-th coefficient of every f_da component into an Eigen
        // vector; the Eigen division is vectorised; setCoeffRow scatters the result back.
        for ( int k = 0; k < order; ++k )
        {
            DVec f_da = rhs_( t_da, y_da );
            ::tax::setCoeffRow< scalar_type, order, DVec::RowsAtCompileTime >(
                y_da, k + 1,
                ::tax::coeffRow< scalar_type, order, DVec::RowsAtCompileTime >( f_da, k ) /
                    scalar_type( k + 1 ) );
        }

        return y_da;
    }

    /**
     * @brief Perform one step y(t0+h) from (t0, y0).
     *
     * The polynomial evaluation is vectorised: all component polynomials are
     * evaluated simultaneously via a single Eigen matrix–vector product
     * `C * [1, h, …, hᴺ]ᵀ` (see `tax::evalSeries`).  Scalar type, Taylor
     * order, and state dimension are passed as explicit template arguments so
     * Eigen can select fully-specialised, SIMD-accelerated kernels.
     */
    template < typename Vec >
    [[nodiscard]] Vec step( double t0, const Vec& y0, double h )
    {
        using DVec = ::tax::detail::rebind_matrix_t< Vec, da_type >;
        auto y_da  = series< Vec >( t0, y0 );

        // Vectorised evaluation with compile-time T, N, and Dim.
        // evalSeries returns Eigen::Matrix<T, Dim, 1>; assigning to Vec is a
        // zero-loop Eigen SIMD copy when the scalar and size types match.
        return Vec( ::tax::evalSeries< scalar_type, order, DVec::RowsAtCompileTime >(
            y_da, scalar_type( h ) ) );
    }

  private:
    RHS rhs_;
};

/**
 * @brief Build a TaylorStepper with decayed RHS type.
 */
template < int N, typename F >
[[nodiscard]] auto makeTaylorStepper( F&& rhs )
{
    return TaylorStepper< N, std::decay_t< F > >( std::forward< F >( rhs ) );
}

}  // namespace tax::ode
