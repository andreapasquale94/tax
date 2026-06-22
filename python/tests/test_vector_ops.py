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


@needs_toolchain
def test_dot_and_norm():
    X = tax.variables([3.0, 4.0], order=2)
    d = tax.dot(X, X)                      # x0^2 + x1^2, value 9+16=25
    np.testing.assert_allclose(d.value(), 25.0, atol=1e-12)
    n = tax.norm(X)                        # sqrt(25) = 5
    np.testing.assert_allclose(n.value(), 5.0, atol=1e-12)
    # d/dx0 (x0^2+x1^2) = 2 x0 = 6 ; d/dx1 = 2 x1 = 8
    np.testing.assert_allclose(d.gradient(), np.array([6.0, 8.0]), atol=1e-12)

@needs_toolchain
def test_cross_3d_matches_numpy():
    X = tax.variables([1.0, 2.0, 3.0], order=1)
    C = tax.concatenate([X[0] * 0.0 + 0.0, X[0] * 0.0 + 1.0, X[0] * 0.0 + 0.0])  # const [0,1,0]
    R = tax.cross(X, C)
    # cross([1,2,3],[0,1,0]) = [2*0-3*1, 3*0-1*0, 1*1-2*0] = [-3, 0, 1]
    np.testing.assert_allclose(R.value(), np.array([-3.0, 0.0, 1.0]), atol=1e-12)

@needs_toolchain
def test_cross_2d_is_scalar():
    X = tax.variables([1.0, 2.0], order=1)
    C = tax.concatenate([X[0] * 0.0 + 3.0, X[0] * 0.0 + 4.0])   # const [3,4]
    z = tax.cross(X, C)                    # x0*4 - x1*3, value 1*4 - 2*3 = -2
    np.testing.assert_allclose(z.value(), -2.0, atol=1e-12)
