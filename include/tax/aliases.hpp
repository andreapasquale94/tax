#pragma once

#include <tax/da.hpp>

namespace tax {

/// @brief Univariate DA alias (`double`, order `N`, one variable).
template <int N>        using DA  = TDA<double, N, 1>;
/// @brief Multivariate DA alias (`double`, order `N`, `M` variables).
template <int N, int M> using DAn = TDA<double, N, M>;

} // namespace tax
