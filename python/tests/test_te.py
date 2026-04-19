"""Tests for the runtime Python bindings of tax."""

from __future__ import annotations

import math

import pytest

import tax


def _close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_module_metadata():
    assert isinstance(tax.__version__, str)
    assert tax.TAX_PY_MAX_N >= 1
    assert tax.TAX_PY_MAX_M >= 1
    assert tax.TAX_PY_MAX_COEFFS >= 10


def test_variable_univariate():
    x = tax.variable(order=3, nvars=1, index=0, x0=2.0)
    assert x.order == 3
    assert x.nvars == 1
    assert x.value() == 2.0
    assert x.coeff([0]) == 2.0
    assert x.coeff([1]) == 1.0
    assert x.coeff([2]) == 0.0


def test_variables_multivariate():
    x, y = tax.variables(order=4, nvars=2, x0=[1.0, -3.0])
    assert x.value() == 1.0
    assert y.value() == -3.0
    assert x.coeff([1, 0]) == 1.0
    assert x.coeff([0, 1]) == 0.0
    assert y.coeff([0, 1]) == 1.0
    assert y.coeff([1, 0]) == 0.0


def test_arithmetic_univariate():
    x = tax.variable(order=5, nvars=1, index=0, x0=0.0)
    f = 2.0 + 3.0 * x - x * x
    # f = 2 + 3x - x^2 → eval(0.1) = 2 + 0.3 - 0.01 = 2.29
    assert _close(f.eval([0.1]), 2.29)
    assert f.value() == 2.0


def test_sin_taylor_coefficients():
    x = tax.variable(order=5, nvars=1, index=0, x0=0.0)
    f = tax.sin(x)
    assert _close(f.coeff([0]), 0.0)
    assert _close(f.coeff([1]), 1.0)
    assert _close(f.coeff([2]), 0.0)
    assert _close(f.coeff([3]), -1.0 / 6.0)
    assert _close(f.coeff([4]), 0.0)
    assert _close(f.coeff([5]), 1.0 / 120.0)


def test_exp_sin_composition():
    x = tax.variable(order=6, nvars=1, index=0, x0=0.0)
    f = tax.exp(tax.sin(x))
    # f(0.05) ≈ exp(sin(0.05))
    assert _close(f.eval([0.05]), math.exp(math.sin(0.05)), tol=1e-6)


def test_multivariate_product():
    x, y = tax.variables(order=4, nvars=2, x0=[0.0, 0.0])
    f = tax.sin(x) * tax.cos(y)
    # coefficient of x^1 y^0 = cos(0) * 1 = 1
    assert _close(f.coeff([1, 0]), 1.0)
    # coefficient of x^0 y^2 = sin(0) * (-1/2) = 0
    assert _close(f.coeff([0, 2]), 0.0)
    # coefficient of x^1 y^2 = 1 * (-1/2) = -0.5
    assert _close(f.coeff([1, 2]), -0.5)


def test_derivative_and_integral():
    x = tax.variable(order=5, nvars=1, index=0, x0=0.0)
    f = tax.sin(x)
    df = f.deriv(0)
    # d/dx sin(x) at 0 = cos(0) = 1
    assert _close(df.eval([0.0]), 1.0)
    F = f.integ(0)
    # ∫ sin = -cos + C, at 0.1: -cos(0.1) + 1 ≈ 0.00499
    assert _close(F.eval([0.1]), 1 - math.cos(0.1), tol=1e-6)


def test_derivative_at_point_factorial_scaling():
    x = tax.variable(order=5, nvars=1, index=0, x0=0.0)
    f = tax.exp(x)
    # k-th derivative of exp at 0 is 1 for all k
    for k in range(6):
        assert _close(f.derivative([k]), 1.0, tol=1e-10)


def test_division():
    x = tax.variable(order=4, nvars=1, index=0, x0=0.0)
    f = 1.0 / (1.0 - x)  # geometric series: 1 + x + x^2 + x^3 + x^4
    for k in range(5):
        assert _close(f.coeff([k]), 1.0)


def test_pow_int_real_and_te():
    x = tax.variable(order=5, nvars=1, index=0, x0=1.0)
    # x^3 at x0=1 → value = 1
    assert _close((x ** 3).value(), 1.0)
    # x^0.5 at 1 → value = 1, derivative = 0.5
    f = x ** 0.5
    assert _close(f.value(), 1.0)
    assert _close(f.derivative([1]), 0.5)
    # x^x at x0=1 → value = 1
    g = tax.pow(x, x)
    assert _close(g.value(), 1.0)


def test_scalar_radd_rsub_rmul_rtruediv():
    x = tax.variable(order=3, nvars=1, index=0, x0=1.0)
    assert _close((2.0 + x).value(), 3.0)
    assert _close((2.0 - x).value(), 1.0)
    assert _close((2.0 * x).value(), 2.0)
    assert _close((2.0 / x).value(), 2.0)


def test_unsupported_pair_raises():
    # N=20, M=10 vastly exceeds the 10k coefficient cap
    with pytest.raises(ValueError):
        tax.variable(order=20, nvars=10, index=0, x0=0.0)


def test_incompatible_operands_raise():
    x = tax.variable(order=3, nvars=1, index=0, x0=0.0)
    y = tax.variable(order=4, nvars=1, index=0, x0=0.0)
    with pytest.raises(ValueError):
        _ = x + y


def test_repr():
    x = tax.variable(order=3, nvars=2, index=0, x0=1.5)
    r = repr(x)
    assert "TE(" in r
    assert "N=3" in r
    assert "M=2" in r
