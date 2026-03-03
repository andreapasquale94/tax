#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace tax::ode
{

/**
 * @brief Runtime options for Taylor integration and default adaptive control.
 */
struct TaylorIntegratorOptions
{
    double atol = 1e-8;              ///< Absolute tolerance.
    double rtol = 1e-8;              ///< Relative tolerance.
    double safetyFactor = 0.9;       ///< Safety factor applied to h_opt.
    double maxGrowth = 4.0;          ///< Max multiplicative growth per step.
    double finalTimeRelEps = 1e-14;  ///< Relative epsilon for loop termination.
    std::size_t maxSteps = 10000;    ///< Max accepted integration steps.
};

/**
 * @brief Time/state samples returned by Taylor integration.
 *
 * @tparam Vec State vector type.
 */
template < typename Vec >
struct Solution
{
    std::vector< double > t;  ///< Time values.
    std::vector< Vec > y;     ///< State snapshots.
};

}  // namespace tax::ode
