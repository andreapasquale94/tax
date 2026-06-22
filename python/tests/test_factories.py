import numpy as np
import tax
from tax._frontend.scheme import Isotropic

def test_variable_seeds_linear_term():
    x = tax.variable(2.5, order=4)
    assert x.scheme == Isotropic(4, 1)
    assert np.array_equal(x.numpy(), np.array([2.5, 1.0, 0.0, 0.0, 0.0]))

def test_variable_order_zero_has_no_linear_slot():
    x = tax.variable(2.5, order=0)
    assert np.array_equal(x.numpy(), np.array([2.5]))
