#pragma once

#include <tax/da.hpp>

namespace da {

template <int N>        using DAd  = DA<double, N, 1>;
template <int N>        using DAf  = DA<float,  N, 1>;
template <int N, int M> using DAMd = DA<double, N, M>;
template <int N, int M> using DAMf = DA<float,  N, M>;

} // namespace da
