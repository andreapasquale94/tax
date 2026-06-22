import numpy as np
import tax
from tax._frontend.array import Array
from tests._helpers import needs_toolchain


@needs_toolchain
def test_vector_map_value_jacobian_numpy():
    X = tax.variables([1.0, 2.0], order=4)
    Y = tax.concatenate([X[0] * X[1], X[0] / X[1]])     # [x0*x1, x0/x1] at (1,2)
    assert isinstance(Y, Array)
    np.testing.assert_allclose(Y.value(), np.array([2.0, 0.5]), atol=1e-12)
    # ∂(x0*x1) = [x1, x0] = [2, 1]; ∂(x0/x1) = [1/x1, -x0/x1^2] = [0.5, -0.25]
    np.testing.assert_allclose(
        Y.jacobian(), np.array([[2.0, 1.0], [0.5, -0.25]]), atol=1e-12
    )
    assert Y.numpy().shape == (2, tax._frontend.scheme.num_monomials(4, 2))


@needs_toolchain
def test_norm_of_vector_map():
    X = tax.variables([3.0, 4.0], order=3)
    r = tax.norm(X)                                      # sqrt(x0^2 + x1^2), value 5
    np.testing.assert_allclose(r.value(), 5.0, atol=1e-12)
    # d/dx0 sqrt(x0^2+x1^2) = x0/r = 3/5 ; d/dx1 = 4/5
    np.testing.assert_allclose(r.gradient(), np.array([0.6, 0.8]), atol=1e-12)
