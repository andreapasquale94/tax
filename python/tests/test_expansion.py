import math, numpy as np, pytest
from tax._frontend.types import Expansion
from tax._frontend.scheme import Isotropic

def test_value_and_numpy():
    e = Expansion([2.0, 1.0, 0.0], Isotropic(2, 1))
    assert e.value() == 2.0
    assert np.array_equal(e.numpy(), np.array([2.0, 1.0, 0.0]))

def test_coeff_and_derivative_univariate():
    # exp(x) at 0, order 3: coeffs [1, 1, 1/2, 1/6]; derivatives all 1
    e = Expansion([1.0, 1.0, 0.5, 1.0/6.0], Isotropic(3, 1))
    assert e.coeff(2) == 0.5
    assert math.isclose(e.derivative(2), 1.0)   # 2! * 1/2
    assert math.isclose(e.derivative(3), 1.0)   # 3! * 1/6

def test_coeff_out_of_range():
    e = Expansion([1.0, 1.0], Isotropic(1, 1))
    with pytest.raises(IndexError):
        e.coeff(5)
