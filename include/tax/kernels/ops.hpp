#pragma once

#include <tax/utils/fwd.hpp>

namespace tax::detail {

template <typename T, std::size_t S>
/// @brief In-place element-wise addition: `o += r`.
constexpr void addInPlace(std::array<T, S>& o, const std::array<T, S>& r) noexcept
{ for (std::size_t i = 0; i < S; ++i) o[i] += r[i]; }

template <typename T, std::size_t S>
/// @brief In-place element-wise subtraction: `o -= r`.
constexpr void subInPlace(std::array<T, S>& o, const std::array<T, S>& r) noexcept
{ for (std::size_t i = 0; i < S; ++i) o[i] -= r[i]; }

template <typename T, std::size_t S>
/// @brief In-place sign flip.
constexpr void negateInPlace(std::array<T, S>& o) noexcept
{ for (auto& v : o) v = -v; }

template <typename T, std::size_t S>
/// @brief In-place scalar multiply.
constexpr void scaleInPlace(std::array<T, S>& o, T s) noexcept
{ for (auto& v : o) v *= s; }

template <typename T, std::size_t S>
/// @brief Absolute value: `out = |a|`. Requires `a[0] != 0`.
constexpr void seriesAbs(std::array<T, S>& out, const std::array<T, S>& a) noexcept
{
    out = a;
    if (a[0] < T{0}) negateInPlace<T, S>(out);
}

} // namespace tax::detail
