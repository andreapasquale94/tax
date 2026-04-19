#pragma once

#include <cmath>

namespace cam
{

struct EarthPhysics
{
    static constexpr double mu = 398600.4418;      // [km^3/s^2]
    static constexpr double rE = 6378.137;         // [km]
    static constexpr double J2 = 1.08262668e-3;    // [-]
    static constexpr double J3 = -2.53648e-6;      // [-]
    static constexpr double J4 = -1.6233e-6;       // [-]
};

struct Scaling
{
    double Lsc;
    double Vsc;
    double Tsc;
    double muSc;
    double rESc;

    static Scaling make( double mu = EarthPhysics::mu, double rE = EarthPhysics::rE ) noexcept
    {
        Scaling s{};
        s.Lsc = rE;
        s.Vsc = std::sqrt( mu / rE );
        s.Tsc = s.Lsc / s.Vsc;
        s.muSc = mu / s.Lsc / s.Lsc / s.Lsc * s.Tsc * s.Tsc;
        s.rESc = rE / s.Lsc;
        return s;
    }
};

}  // namespace cam
