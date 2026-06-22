import pytest
from tax._frontend.scheme import Isotropic, num_monomials

def test_num_monomials():
    assert num_monomials(5, 1) == 6          # univariate order 5 -> 6 coeffs
    assert num_monomials(4, 4) == 70

def test_isotropic_properties():
    s = Isotropic(5, 1)
    assert s.n_coeff == 6
    assert s.cpp_type_string() == "tax::IsotropicScheme<5, 1>"
    assert s.descriptor_hash() == "iso:5:1"

def test_isotropic_validation():
    with pytest.raises(ValueError):
        Isotropic(-1, 1)
    with pytest.raises(ValueError):
        Isotropic(3, 0)

def test_isotropic_union_promotes_order():
    assert Isotropic(3, 1).union(Isotropic(5, 1)) == Isotropic(5, 1)
    with pytest.raises(ValueError):
        Isotropic(3, 1).union(Isotropic(3, 2))   # differing vars
