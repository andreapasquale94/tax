#pragma once

/**
 * @file cam.hpp
 * @brief Umbrella include for the CAM downstream library.
 *
 * CAM = Collision Avoidance Maneuver. This library is a C++23 port of the C++ core
 * of https://github.com/arma1978/multi-impulsive-cam, replacing DACE with tax for
 * differential-algebra polynomial propagation.
 */

#include <cam/anomaly.hpp>
#include <cam/collision.hpp>
#include <cam/constants.hpp>
#include <cam/elements.hpp>
#include <cam/frames.hpp>
#include <cam/linalg.hpp>
#include <cam/multi_impulse.hpp>
#include <cam/optim/optim.hpp>
#include <cam/propagator.hpp>
#include <cam/tca.hpp>
