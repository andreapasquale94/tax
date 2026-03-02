#pragma once

#include <tax/ode/taylor/types.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <type_traits>

namespace tax::ode
{

/**
 * @brief Default Jorba-Zou adaptive step-size controller.
 *
 * @tparam Order Taylor order.
 */
template < int Order >
class JorbaZouStepSizeController
{
   public:
    static_assert( Order >= 2, "JorbaZouStepSizeController requires Order >= 2." );

    using options_type = TaylorIntegratorOptions;
    static constexpr int order = Order;

    explicit JorbaZouStepSizeController( options_type options = {} ) noexcept : options_( options )
    {
    }

    [[nodiscard]] const options_type& options() const noexcept { return options_; }
    
    void setOptions( const options_type& options ) noexcept { options_ = options; }

    /**
     * @brief Propose the next step-size using current series coefficients.
     *
     * @param h Current accepted step-size.
     * @param tf Final integration time.
     * @param y Current state (at current time).
     * @param yDA Current Taylor series of the state.
     * @return Proposed next step-size (before end-interval clamping).
     */
    template < typename Vec, typename Series >
    [[nodiscard]] double nextStep( double h, double tf, const Vec& y, const Series& yDA ) const
    {
        (void)tf;
        assert( h > 0.0 );
        assert( options_.atol >= 0.0 );
        assert( options_.rtol >= 0.0 );
        assert( options_.safetyFactor > 0.0 );
        assert( options_.maxGrowth > 0.0 );

        const Eigen::Index dim = y.size();

        double y_norm = 0.0;
        for ( Eigen::Index i = 0; i < dim; ++i )
            y_norm = std::max( y_norm, std::abs( double( y( i ) ) ) );
        const double tol = options_.atol + options_.rtol * y_norm;

        double h_opt = h * options_.maxGrowth;
        constexpr int k_min = ( order / 2 > 1 ) ? ( order / 2 ) : 2;
        for ( int k = k_min; k <= order; ++k )
        {
            double ck = 0.0;
            for ( Eigen::Index i = 0; i < dim; ++i )
                ck = std::max( ck, std::abs( yDA( i )[k] ) );
            if ( ck > 0.0 ) h_opt = std::min( h_opt, std::pow( tol / ck, 1.0 / k ) );
        }
        h_opt *= options_.safetyFactor;
        return h_opt;
    }

   private:
    options_type options_;
};

namespace detail::taylor
{

template < typename... >
inline constexpr bool always_false_v = false;

template < typename Controller >
[[nodiscard]] double finalTimeRelEps( const Controller& controller )
{
    if constexpr ( requires { controller.options().finalTimeRelEps; } )
    {
        return controller.options().finalTimeRelEps;
    } else
    {
        return 1e-14;
    }
}

template < typename Controller >
[[nodiscard]] std::size_t maxSteps( const Controller& controller )
{
    if constexpr ( requires { controller.options().maxSteps; } )
    {
        return controller.options().maxSteps;
    } else
    {
        return std::numeric_limits< std::size_t >::max();
    }
}

template < typename Controller, typename Vec, typename Series >
[[nodiscard]] double proposeNextStep( Controller& controller, double h, double tf, const Vec& y,
                                      const Series& yDA )
{
    if constexpr ( requires { controller.nextStep( h, tf, y, yDA ); } )
    {
        return static_cast< double >( controller.nextStep( h, tf, y, yDA ) );
    } else if constexpr ( requires { controller( h, tf, y, yDA ); } )
    {
        return static_cast< double >( controller( h, tf, y, yDA ) );
    } else
    {
        static_assert( always_false_v< Controller >,
                       "Step controller must provide nextStep(h, tf, y, yDA) "
                       "or operator()(h, tf, y, yDA)." );
    }
}

}  // namespace detail::taylor

}  // namespace tax::ode
