#pragma once
// Facade: Taylor models (Makino ch. 4) — interval arithmetic plus
// remainder-verified truncated Taylor expansions (namespace tax::model).

#include <tax/model/arithmetic.hpp>
#include <tax/model/interval.hpp>
#include <tax/model/math.hpp>
#include <tax/model/taylor_model.hpp>

namespace tax
{

// Re-export the public types; the operator/function surface is found via ADL.
using model::Bounder;
using model::Interval;
using model::TaylorModel;
using model::TM;

}  // namespace tax
