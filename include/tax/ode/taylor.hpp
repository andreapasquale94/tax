#pragma once

/**
 * @file tax/ode/taylor.hpp
 * @brief Taylor ODE integration module.
 *
 * Requires Eigen. Include this header directly; it is not part of the
 * default tax.hpp umbrella.
 *
 * Usage example:
 * @code
 *   #include <tax/ode/taylor.hpp>
 *   using namespace tax;
 *   using namespace tax::ode;
 *
 *   auto rhs = [](auto t, auto y) { return decltype(y){ y(1), -y(0) }; };
 *
 *   Eigen::Vector2d y0{1.0, 0.0};
 *   auto integrator = makeTaylorIntegrator<10>(rhs);
 *   auto result = integrator.integrate(0.0, 1.0, y0, 0.1);
 * @endcode
 */

#include <tax/ode/taylor/controller.hpp>
#include <tax/ode/taylor/integrator.hpp>
#include <tax/ode/taylor/stepper.hpp>
#include <tax/ode/taylor/types.hpp>
