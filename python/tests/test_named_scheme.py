import pytest
from tax._frontend.scheme import Axis, Named, Isotropic

def test_axis_validation():
    with pytest.raises(ValueError):
        Axis("x", 0)

def test_named_requires_canonical():
    Named(4, (Axis("mu", 1), Axis("x", 4)))          # ok: sorted
    with pytest.raises(ValueError):
        Named(4, (Axis("x", 4), Axis("mu", 1)))      # not sorted
    with pytest.raises(ValueError):
        Named(4, (Axis("x", 4), Axis("x", 4)))       # duplicate name

def test_named_of_sorts_and_checks_dims():
    n = Named.of(4, [Axis("x", 4), Axis("mu", 1)])
    assert n.axes == (Axis("mu", 1), Axis("x", 4))   # sorted mu < x
    with pytest.raises(ValueError):
        Named.of(4, [Axis("x", 4), Axis("x", 2)])    # same name, conflicting dim

def test_vars_ncoeff_and_isotropic_delegation():
    n = Named.of(4, [Axis("mu", 1), Axis("x", 4)])
    assert n.vars == 5
    assert n.n_coeff == Isotropic(4, 5).n_coeff      # 126
    assert n.isotropic() == Isotropic(4, 5)
    assert n.cpp_type_string() == "tax::IsotropicScheme<4, 5>"
    assert n.descriptor_hash() == Isotropic(4, 5).descriptor_hash()

def test_var_offset_and_dim():
    n = Named.of(4, [Axis("mu", 1), Axis("x", 4)])
    assert n.var_offset("mu") == 0
    assert n.var_offset("x") == 1                     # mu(dim1) precedes x
    assert n.dim_of("x") == 4
    with pytest.raises(KeyError):
        n.var_offset("nope")

def test_union_and_var_map():
    x = Named.of(4, [Axis("x", 4)])
    mu = Named.of(2, [Axis("mu", 1)])
    u = x.union(mu)
    assert u.axes == (Axis("mu", 1), Axis("x", 4))    # union, sorted
    assert u.order == 4                               # max order
    assert x.axis_var_map(u) == [1, 2, 3, 4]          # x's vars -> union vars 1..4
    assert mu.axis_var_map(u) == [0]
    with pytest.raises(ValueError):
        Named.of(3, [Axis("x", 4)]).union(Named.of(3, [Axis("x", 2)]))  # dim conflict
