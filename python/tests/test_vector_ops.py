import numpy as np
import tax
from tax._frontend.array import Array
from tests._helpers import needs_toolchain


@needs_toolchain
def test_concatenate_scalars_into_vector():
    X = tax.variables([1.0, 2.0], order=2)
    Y = tax.concatenate([X[0] * X[1], X[0] + X[1]])
    assert isinstance(Y, Array) and len(Y) == 2
    np.testing.assert_allclose(Y.value(), np.array([2.0, 3.0]), atol=1e-12)


@needs_toolchain
def test_concatenate_flattens_arrays():
    X = tax.variables([1.0, 2.0], order=2)
    Y = tax.concatenate([X, X[0] * X[1]])     # 2 rows from X + 1 row -> length 3
    assert len(Y) == 3
    np.testing.assert_allclose(Y.value(), np.array([1.0, 2.0, 2.0]), atol=1e-12)
