import numpy as np
from tax._frontend.array import Array
from tax._frontend.types import Expansion
from tax._frontend.scheme import Isotropic

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
