#pragma once

#include <tax/expr/base.hpp>
#include <tax/kernels.hpp>

namespace da {

// =============================================================================
// DA<T, N, M> — leaf / materialised type
// =============================================================================

template <typename T, int N, int M = 1>
class DA
    : public DAExpr<DA<T, N, M>, T, N, M>
    , public DALeaf
{
public:
    static_assert(N >= 0, "DA order must be non-negative");
    static_assert(M >= 1, "Number of variables must be at least 1");

    static constexpr std::size_t ncoef = detail::numMonomials(N, M);
    using coeff_array = std::array<T, ncoef>;
    using point_type  = std::array<T, M>;

    // -- Constructors ---------------------------------------------------------

    constexpr DA() noexcept : c_{} {}
    explicit constexpr DA(coeff_array c) noexcept : c_(std::move(c)) {}
    /*implicit*/ constexpr DA(T val) noexcept : c_{} { c_[0] = val; }

    /// Materialise any expression: a single evalTo into c_.  Zero copies.
    template <typename Derived>
    /*implicit*/ constexpr DA(const DAExpr<Derived, T, N, M>& expr) noexcept
        : c_{} { expr.self().evalTo(c_); }

    // -- Variable factories ---------------------------------------------------

    [[nodiscard]] static constexpr DA variable(T x0) noexcept requires (M == 1)
    {
        coeff_array c{};
        c[0] = x0;
        if constexpr (N >= 1) c[1] = T{1};
        return DA{c};
    }

    template <int I>
    [[nodiscard]] static constexpr DA variable(const point_type& x0) noexcept {
        static_assert(I >= 0 && I < M, "Variable index out of range");
        coeff_array c{};
        c[0] = x0[I];
        if constexpr (N >= 1) {
            MultiIndex<M> ei{};
            ei[I] = 1;
            c[detail::flatIndex<M>(ei)] = T{1};
        }
        return DA{c};
    }

    [[nodiscard]] static constexpr auto variables(const point_type& x0) noexcept {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::tuple{ variable<int(I)>(x0)... };
        }(std::make_index_sequence<std::size_t(M)>{});
    }

    [[nodiscard]] static constexpr DA constant(T v) noexcept { return DA{v}; }

    // -- evalTo / addTo / subTo -----------------------------------------------

    constexpr void evalTo(coeff_array& out) const noexcept { out = c_; }

    constexpr void addTo(coeff_array& out) const noexcept
    { detail::addInPlace<T, ncoef>(out, c_); }

    constexpr void subTo(coeff_array& out) const noexcept
    { detail::subInPlace<T, ncoef>(out, c_); }

    // -- Element access -------------------------------------------------------

    [[nodiscard]] constexpr T  operator[](std::size_t i) const noexcept { return c_[i]; }
    [[nodiscard]] constexpr T& operator[](std::size_t i)       noexcept { return c_[i]; }
    [[nodiscard]] constexpr const coeff_array& coeffs() const noexcept  { return c_; }
    [[nodiscard]] constexpr T value() const noexcept { return c_[0]; }

    [[nodiscard]] constexpr T coeff(const MultiIndex<M>& alpha) const noexcept
    { return c_[detail::flatIndex<M>(alpha)]; }

    [[nodiscard]] constexpr T derivative(const MultiIndex<M>& alpha) const noexcept {
        const auto id = detail::flatIndex<M>(alpha);
        const auto factor = derivative_factors_[id];
        return c_[id] * T(factor);
    }

    [[nodiscard]] constexpr coeff_array derivatives() const noexcept {
        coeff_array out{};
        for (std::size_t i = 0; i < ncoef; ++i) out[i] = c_[i] * derivative_factors_[i];
        return out;
    }

    // -- In-place operators ---------------------------------------------------

    constexpr DA& operator+=(const DA& o) noexcept
    { detail::addInPlace<T, ncoef>(c_, o.c_); return *this; }
    constexpr DA& operator-=(const DA& o) noexcept
    { detail::subInPlace<T, ncoef>(c_, o.c_); return *this; }
    template <typename Derived>
    constexpr DA& operator+=(const DAExpr<Derived, T, N, M>& e) noexcept
    { coeff_array t{}; e.self().evalTo(t); detail::addInPlace<T, ncoef>(c_, t); return *this; }
    template <typename Derived>
    constexpr DA& operator-=(const DAExpr<Derived, T, N, M>& e) noexcept
    { coeff_array t{}; e.self().evalTo(t); detail::subInPlace<T, ncoef>(c_, t); return *this; }
    constexpr DA& operator*=(T s) noexcept
    { detail::scaleInPlace<T, ncoef>(c_, s); return *this; }
    constexpr DA& operator/=(T s) noexcept
    { detail::scaleInPlace<T, ncoef>(c_, T{1} / s); return *this; }

private:
    [[nodiscard]] static constexpr coeff_array makeDerivativeFactors() noexcept {
        coeff_array factors{};
        MultiIndex<M> alpha{};

        auto fillAlpha = [&](auto& self, int var, int rem) constexpr -> void {
            if (var == M - 1) {
                alpha[var] = rem;
                std::size_t fac = 1;
                for (int i = 0; i < M; ++i)
                    for (int j = 1; j <= alpha[i]; ++j)
                        fac *= std::size_t(j);
                factors[detail::flatIndex<M>(alpha)] = T(fac);
                return;
            }
            for (int k = rem; k >= 0; --k) {
                alpha[var] = k;
                self(self, var + 1, rem - k);
            }
        };

        for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d);
        return factors;
    }

    inline static constexpr coeff_array derivative_factors_ = makeDerivativeFactors();
    coeff_array c_;
};

} // namespace da
