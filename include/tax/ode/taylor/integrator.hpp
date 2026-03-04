#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <tax/ode/taylor/controller.hpp>
#include <tax/ode/taylor/stepper.hpp>
#include <tax/ode/taylor/types.hpp>
#include <type_traits>
#include <utility>

namespace tax::ode
{

/**
 * @brief N-th order adaptive Taylor integrator for vector ODEs.
 *
 * @tparam N   Taylor order.
 * @tparam RHS Callable with signature compatible with
 *             `VecT<TE<N>, Rows> rhs(TE<N> t, VecT<TE<N>, Rows> y)`.
 * @tparam StepSizeController Step-size controller type.
 */
template < int N, typename RHS, typename StepSizeController = JorbaZouStepSizeController< N > >
class TaylorIntegrator
{
   public:
    static_assert( N >= 1, "Taylor order N must be at least 1." );

    using stepper_type = TaylorStepper< N, RHS >;
    using da_type = typename stepper_type::da_type;
    using scalar_type = typename stepper_type::scalar_type;

    using rhs_type = RHS;
    using controller_type = StepSizeController;
    static constexpr int order = stepper_type::order;

    explicit TaylorIntegrator( RHS rhs, StepSizeController controller = {} ) noexcept
        : stepper_( std::move( rhs ) ), controller_( std::move( controller ) )
    {
    }

    explicit TaylorIntegrator( RHS rhs, TaylorIntegratorOptions options ) noexcept
        requires( std::same_as< StepSizeController, JorbaZouStepSizeController< N > > )
        : stepper_( std::move( rhs ) ), controller_( options )
    {
    }

    [[nodiscard]] RHS& rhs() noexcept { return stepper_.rhs(); }
    [[nodiscard]] const RHS& rhs() const noexcept { return stepper_.rhs(); }
    [[nodiscard]] StepSizeController& stepSizeController() noexcept { return controller_; }
    [[nodiscard]] const StepSizeController& stepSizeController() const noexcept
    {
        return controller_;
    }

    template < typename C = StepSizeController >
    [[nodiscard]] const auto& options() const noexcept
        requires( requires( const C& c ) { c.options(); } )
    {
        return controller_.options();
    }

    template < typename C = StepSizeController >
    void setOptions( const TaylorIntegratorOptions& options ) noexcept
        requires( requires( C& c ) { c.setOptions( options ); } )
    {
        controller_.setOptions( options );
    }

    /**
     * @brief Build local Taylor state series around (t0, y0).
     */
    template < typename Vec >
    [[nodiscard]] auto series( double t0, const Vec& y0 )
    {
        return stepper_.template series< Vec >( t0, y0 );
    }

    /**
     * @brief Perform one step y(t0+h) from (t0, y0).
     */
    template < typename Vec >
    [[nodiscard]] Vec step( double t0, const Vec& y0, double h )
    {
        return stepper_.template step< Vec >( t0, y0, h );
    }

    /**
     * @brief Integrate dy/dt = rhs(t, y) from @p t0 to @p tf.
     */
    template < typename Vec >
    [[nodiscard]] Solution< Vec > integrate( const double t0, const double tf, const Vec& y0,
                                             double h0 )
    {
        return integrateInternal( t0, tf, y0, h0 );
    }

   private:
    template < typename Vec >
    [[nodiscard]] Solution< Vec > integrateInternal( const double t0, const double tf,
                                                     const Vec& y0, double h0 )
    {
        static_assert( order >= 2, "Taylor order N must be at least 2 for adaptive control." );

        detail::taylor::assertColumnVector( y0 );
        assert( h0 > 0.0 );
        const std::size_t max_steps = detail::taylor::maxSteps( controller_ );
        const double final_time_rel_eps = detail::taylor::finalTimeRelEps( controller_ );
        assert( final_time_rel_eps >= 0.0 );

        Solution< Vec > result;
        result.t.push_back( t0 );
        result.y.push_back( y0 );

        using DVec = ::tax::detail::rebind_matrix_t< Vec, da_type >;

        Vec y = y0;
        double t = t0;
        double h = std::min( h0, tf - t0 );
        std::size_t accepted_steps = 0;

        while ( accepted_steps < max_steps && ( t < tf - final_time_rel_eps * std::abs( tf ) ) )
        {
            h = std::min( h, tf - t );
            if ( h <= 0.0 ) break;

            auto y_da     = stepper_.template series< Vec >( t, y );
            const double h_opt = detail::taylor::proposeNextStep( controller_, h, tf, y, y_da );
            assert( h_opt > 0.0 && "Step-size controller must return a positive h_opt." );

            Vec y_new( ::tax::evalSeries< scalar_type, order, DVec::RowsAtCompileTime >(
                y_da, scalar_type( h ) ) );

            t += h;
            y = y_new;
            result.t.push_back( t );
            result.y.push_back( y_new );
            ++accepted_steps;

            h = ( tf - t > 0.0 ) ? std::min( h_opt, tf - t ) : h_opt;
        }

        return result;
    }

    stepper_type stepper_;
    StepSizeController controller_;
};

/**
 * @brief Build a TaylorIntegrator with decayed RHS type.
 */
template < int N, typename F >
[[nodiscard]] auto makeTaylorIntegrator( F&& rhs, TaylorIntegratorOptions options = {} )
{
    return TaylorIntegrator< N, std::decay_t< F > >( std::forward< F >( rhs ), options );
}

template < int N, typename F, typename Controller >
[[nodiscard]] auto makeTaylorIntegrator( F&& rhs, Controller&& controller )
    requires( !std::same_as< std::remove_cvref_t< Controller >, TaylorIntegratorOptions > )
{
    return TaylorIntegrator< N, std::decay_t< F >, std::remove_cvref_t< Controller > >(
        std::forward< F >( rhs ), std::forward< Controller >( controller ) );
}

}  // namespace tax::ode
