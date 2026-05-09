"""Smoke tests for the tax Python bindings."""

import math

import pytest

import tax


def test_class_is_not_directly_constructible():
    # The DynTE type exists but has no public Python constructor;
    # construction goes through the module-level factories.
    with pytest.raises(TypeError):
        tax.DynTE()


def test_zero_constant_one():
    z = tax.zero(3, 2)
    assert isinstance(z, tax.DynTE)
    assert z.order == 3
    assert z.nvars == 2
    assert z.value() == 0.0

    k = tax.constant(2.5, 3, 2)
    assert k.value() == 2.5

    o = tax.one(3, 2)
    assert o.value() == 1.0


def test_variable_seed():
    x = tax.variable(1.5, 3, 1, 0)
    assert x.value() == 1.5
    assert x.coeff([1]) == 1.0


def test_variables_unpacks_to_list():
    vars_ = tax.variables([1.0, 2.0], 2)
    x, y = vars_
    assert x.value() == 1.0
    assert y.value() == 2.0
    assert x.coeff([1, 0]) == 1.0
    assert y.coeff([0, 1]) == 1.0


def test_arithmetic_with_dynte_and_scalar():
    x = tax.variable(1.0, 3, 1, 0)
    y = tax.variable(2.0, 3, 1, 0)

    r_add = x + y
    assert r_add.value() == pytest.approx(3.0)
    assert r_add.coeff([1]) == pytest.approx(2.0)

    r_sub = x - y
    assert r_sub.value() == pytest.approx(-1.0)

    r_scal = 3.0 * x + 7.0
    assert r_scal.value() == pytest.approx(10.0)
    assert r_scal.coeff([1]) == pytest.approx(3.0)

    r_neg = -x
    assert r_neg.value() == pytest.approx(-1.0)
    assert r_neg.coeff([1]) == pytest.approx(-1.0)


def test_multiply_multivariate():
    x, y = tax.variables([1.0, 2.0], 2)
    r = x * y
    assert r.value() == pytest.approx(2.0)
    assert r.coeff([1, 0]) == pytest.approx(2.0)
    assert r.coeff([0, 1]) == pytest.approx(1.0)
    assert r.coeff([1, 1]) == pytest.approx(1.0)


def test_division_reciprocal():
    # 1 / (1 + x) at x=0 = 1 - x + x^2 - x^3 + x^4
    x = tax.variable(0.0, 4, 1, 0)
    one = tax.one(4, 1)
    r = one / (one + x)
    expected = [1.0, -1.0, 1.0, -1.0, 1.0]
    for i, e in enumerate(expected):
        assert r.coeff([i]) == pytest.approx(e)


def test_math_exp_at_zero():
    x = tax.variable(0.0, 4, 1, 0)
    r = tax.exp(x)
    assert r.coeff([0]) == pytest.approx(1.0)
    assert r.coeff([1]) == pytest.approx(1.0)
    assert r.coeff([2]) == pytest.approx(0.5)
    assert r.coeff([3]) == pytest.approx(1.0 / 6.0)


def test_math_log_at_one():
    x = tax.variable(1.0, 4, 1, 0)
    r = tax.log(x)
    assert r.coeff([0]) == pytest.approx(0.0)
    assert r.coeff([1]) == pytest.approx(1.0)
    assert r.coeff([2]) == pytest.approx(-0.5)
    assert r.coeff([3]) == pytest.approx(1.0 / 3.0)


def test_math_sin_cos_pythagorean():
    x = tax.variable(0.7, 5, 1, 0)
    s = tax.sin(x)
    c = tax.cos(x)
    r = tax.square(s) + tax.square(c)
    assert r.coeff([0]) == pytest.approx(1.0)
    for i in range(1, 6):
        assert r.coeff([i]) == pytest.approx(0.0, abs=1e-11)


def test_math_sqrt_around_one():
    x = tax.variable(1.0, 4, 1, 0)
    r = tax.sqrt(x)
    assert r.coeff([0]) == pytest.approx(1.0)
    assert r.coeff([1]) == pytest.approx(0.5)
    assert r.coeff([2]) == pytest.approx(-1.0 / 8.0)
    assert r.coeff([3]) == pytest.approx(1.0 / 16.0)


def test_eval_matches_function_value():
    x = tax.variable(0.0, 8, 1, 0)
    r = tax.exp(x)
    assert r.eval([0.3]) == pytest.approx(math.exp(0.3), abs=1e-9)


def test_brief_example_end_to_end():
    u, v = tax.variables([1.0, 2.0], 3)
    f = u * tax.sin(v) + u * v
    assert f.value() == pytest.approx(math.sin(2.0) + 2.0)
    # df/du at (1, 2) = sin(2) + 2
    assert f.derivative([1, 0]) == pytest.approx(math.sin(2.0) + 2.0)
    # df/dv at (1, 2) = u*cos(2) + u = cos(2) + 1
    assert f.derivative([0, 1]) == pytest.approx(math.cos(2.0) + 1.0)


def test_repr_smoke():
    z = tax.constant(1.5, 2, 1)
    assert "DynTE" in repr(z)
    assert "order=2" in repr(z)
    assert "nvars=1" in repr(z)
