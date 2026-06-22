import numpy as np
import tax
from tax._frontend.array import Array
from tax._frontend.scheme import Isotropic
from tests._helpers import needs_toolchain

def test_variable_seeds_linear_term():
    x = tax.variable(2.5, order=4)
    assert x.scheme == Isotropic(4, 1)
    assert np.array_equal(x.numpy(), np.array([2.5, 1.0, 0.0, 0.0, 0.0]))

def test_variable_order_zero_has_no_linear_slot():
    x = tax.variable(2.5, order=0)
    assert np.array_equal(x.numpy(), np.array([2.5]))

def test_variables_seeds_coordinate_rows():
    X = tax.variables([1.0, 2.0], order=2)
    assert isinstance(X, Array)
    assert X.scheme == Isotropic(2, 2)
    # row 0: x0 = 1 + dx0  -> [1, 1, 0, 0, 0, 0]; row 1: x1 = 2 + dx1 -> [2, 0, 1, 0, 0, 0]
    assert np.array_equal(X[0].coeffs, np.array([1.0, 1.0, 0, 0, 0, 0]))
    assert np.array_equal(X[1].coeffs, np.array([2.0, 0.0, 1.0, 0, 0, 0]))

@needs_toolchain
def test_multivariate_eager_product():
    X = tax.variables([1.0, 2.0], order=2)
    f = X[0] * X[1]                       # eager mul over IsotropicScheme<2,2>
    # (1+dx0)(2+dx1) = 2 + 2 dx0 + 1 dx1 + dx0 dx1
    np.testing.assert_allclose(
        f.numpy(), np.array([2.0, 2.0, 1.0, 0.0, 1.0, 0.0]), atol=1e-12
    )
    assert np.array_equal(f.gradient(), np.array([2.0, 1.0]))
