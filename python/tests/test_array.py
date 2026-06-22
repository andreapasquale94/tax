import numpy as np
import tax
from tax._frontend.array import Array
from tax._frontend.types import Expansion
from tax._frontend.scheme import Isotropic
from tests._helpers import needs_toolchain

def test_array_construction_and_indexing():
    s = Isotropic(2, 2)
    data = np.array([[1.0, 2.0, 3.0, 0, 0, 0], [4.0, 5.0, 6.0, 0, 0, 0]])
    a = Array(data, s)
    assert len(a) == 2
    assert isinstance(a[0], Expansion)
    assert np.array_equal(a[0].coeffs, data[0])
    assert np.array_equal(a.value(), np.array([1.0, 4.0]))
    sub = a[0:1]
    assert isinstance(sub, Array) and len(sub) == 1

def test_array_numpy_is_copy():
    s = Isotropic(1, 1)
    a = Array(np.array([[1.0, 2.0]]), s)
    out = a.numpy()
    out[0, 0] = 99.0
    assert a.value()[0] == 1.0

def test_array_shape_validation():
    import pytest
    with pytest.raises(ValueError):
        Array(np.zeros((2, 5)), Isotropic(2, 2))   # nCoeff(2,2)=6, not 5

@needs_toolchain
def test_array_elementwise_math():
    X = tax.variables([0.0, 0.0], order=3)
    S = tax.sin(X)                         # elementwise sin over the 2-vector
    # each row depends only on its own variable: sin(dx_i)
    # row 0 = sin(dx0): coeff(1,0)=1, coeff(3,0)=-1/6
    np.testing.assert_allclose(S[0].coeff(1, 0), 1.0, atol=1e-12)
    np.testing.assert_allclose(S[0].coeff(3, 0), -1.0 / 6.0, atol=1e-12)
    np.testing.assert_allclose(S[1].coeff(0, 1), 1.0, atol=1e-12)

@needs_toolchain
def test_array_arithmetic_and_broadcast():
    X = tax.variables([1.0, 2.0], order=2)
    Y = 2.0 * X + X                        # scalar broadcast + elementwise add -> 3*X
    np.testing.assert_allclose(Y.value(), np.array([3.0, 6.0]), atol=1e-12)
    Z = X + X[0]                            # broadcast an Expansion over the Array
    np.testing.assert_allclose(Z.value(), np.array([2.0, 3.0]), atol=1e-12)
