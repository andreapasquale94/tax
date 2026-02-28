#pragma once

#include <utility>

#include <tax/operators/common.hpp>

namespace tax {

/**
 * @brief Compute `sin(e)` and `cos(e)` together, returning `{sin, cos}`.
 * @details This evaluates the operand once and runs a coupled recurrence.
 */
template <typename E>
[[nodiscard]] constexpr auto sincos(const ExprBase<E>& e) noexcept {
    using T = typename E::scalar_type;
    constexpr int N = E::order, M = E::nvars;
    using DAType = TDA<T, N, M>;
    using coeff_array = std::array<T, detail::numMonomials(N, M)>;

    coeff_array sa{}, ca{};
    if constexpr (detail::is_leaf_v<E>) {
        detail::seriesSinCos<T, N, M>(sa, ca, e.self().coeffs());
    } else {
        coeff_array a{};
        e.self().evalTo(a);
        detail::seriesSinCos<T, N, M>(sa, ca, a);
    }
    return std::pair<DAType, DAType>{DAType{sa}, DAType{ca}};
}

template <typename E>
[[nodiscard]] constexpr auto sinhcosh(const ExprBase<E>& e) noexcept {
    using T = typename E::scalar_type;
    constexpr int N = E::order, M = E::nvars;
    using DAType = TDA<T, N, M>;
    using coeff_array = std::array<T, detail::numMonomials(N, M)>;

    coeff_array sha{}, cha{};
    if constexpr (detail::is_leaf_v<E>) {
        detail::seriesSinhCosh<T, N, M>(sha, cha, e.self().coeffs());
    } else {
        coeff_array a{};
        e.self().evalTo(a);
        detail::seriesSinhCosh<T, N, M>(sha, cha, a);
    }
    return std::pair<DAType, DAType>{DAType{sha}, DAType{cha}};
}

} // namespace tax
