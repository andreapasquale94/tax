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
 *             `VecT<DA<N>, Rows> rhs(DA<N> t, VecT<DA<N>, Rows> y)`.
 */
template < int N, typename RHS >
class TaylorStepper
{
  public:
    static_assert( N >= 1, "Taylor order N must be at least 1." );

    using da_type = DA< N >;
    using da_traits = ::tax::detail::eigen::da_traits< da_type >;
    using scalar_type = typename da_traits::scalar_type;

    using rhs_type = RHS;
    static constexpr int order = da_traits::order;
    static constexpr int vars  = da_traits::vars;
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
    [[nodiscard]] ::tax::detail::eigen::rebind_matrix_t< Vec, da_type >
    series( double t0, const Vec& y0 )
    {
        detail::taylor::assertColumnVector( y0 );

        using DVec = ::tax::detail::eigen::rebind_matrix_t< Vec, da_type >;

        const Eigen::Index dim = y0.size();
        da_type t_da = da_type::variable( static_cast< scalar_type >( t0 ) );

        DVec y_da( dim );
        for ( Eigen::Index i = 0; i < dim; ++i )
            y_da( i ) = da_type( static_cast< scalar_type >( y0( i ) ) );

        // ODE recurrence: c_{k+1}(y_i) = c_k(f_i) / (k+1)
        for ( int k = 0; k < order; ++k )
        {
            DVec f_da = rhs_( t_da, y_da );
            for ( Eigen::Index i = 0; i < dim; ++i )
                y_da( i )[k + 1] = f_da( i )[k] / double( k + 1 );
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
        using DVec = ::tax::detail::eigen::rebind_matrix_t< Vec, da_type >;
        auto y_da  = series< Vec >( t0, y0 );

        // Vectorised evaluation with compile-time T, N, and Dim.
        auto y_result =
            ::tax::evalSeries< scalar_type, order, DVec::RowsAtCompileTime >( y_da,
                                                                               scalar_type( h ) );

        const Eigen::Index dim = y0.size();
        Vec y_new( dim );
        for ( Eigen::Index i = 0; i < dim; ++i )
            y_new( i ) = y_result( i );
        return y_new;
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
