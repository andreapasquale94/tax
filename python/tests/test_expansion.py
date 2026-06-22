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

def test_coeff_out_of_box_returns_zero_and_validates_arity():
    e = Expansion([1.0, 1.0], Isotropic(1, 1))
    assert e.coeff(5) == 0.0            # degree 5 > order 1 -> not in box -> 0 (C++ kNotInBox)
    with pytest.raises(ValueError):
        e.coeff(1, 2)                   # wrong arity (vars == 1)
    with pytest.raises(ValueError):
        e.coeff(-1)                     # negative exponent


def test_multivariate_coeff_and_derivative():
    # f = x0*x1 expanded at (1,2), order 2, M=2:
    #   flat layout [ (0,0),(1,0),(0,1),(2,0),(1,1),(0,2) ] = [2, 2, 1, 0, 1, 0]
    f = Expansion([2.0, 2.0, 1.0, 0.0, 1.0, 0.0], Isotropic(2, 2))
    assert f.coeff(0, 0) == 2.0
    assert f.coeff(1, 0) == 2.0
    assert f.coeff(0, 1) == 1.0
    assert f.coeff(1, 1) == 1.0
    assert f.coeff(2, 0) == 0.0
    assert f.derivative(1, 1) == 1.0    # mixed partial of x0*x1 = 1
    assert f.derivative(2, 0) == 0.0    # coeff(2,0)=0 -> 0


def test_gradient_and_hessian():
    f = Expansion([2.0, 2.0, 1.0, 0.0, 1.0, 0.0], Isotropic(2, 2))   # x0*x1 at (1,2)
    assert np.array_equal(f.gradient(), np.array([2.0, 1.0]))        # [x1, x0] = [2, 1]
    assert np.array_equal(f.hessian(), np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_eval_multivariate():
    f = Expansion([2.0, 2.0, 1.0, 0.0, 1.0, 0.0], Isotropic(2, 2))   # (1+dx0)(2+dx1)
    # exact at order 2: (1.1)*(2.2) = 2.42
    assert abs(f.eval([0.1, 0.2]) - 2.42) < 1e-12
