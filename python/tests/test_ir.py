from tax._frontend.ir import Var, Const, Op, Graph, single_op_graph
from tax._frontend.scheme import Isotropic

def test_single_op_graph_shape():
    s = Isotropic(5, 1)
    g = single_op_graph("mul", [s, s], s)
    assert g.n_inputs == 2
    assert isinstance(g.nodes[0], Var) and isinstance(g.nodes[1], Var)
    assert isinstance(g.nodes[2], Op) and g.nodes[2].operands == (0, 1)
    assert g.outputs == [2]

def test_canonical_is_structural():
    s = Isotropic(5, 1)
    a = single_op_graph("sin", [s], s).canonical()
    b = single_op_graph("sin", [s], s).canonical()
    c = single_op_graph("cos", [s], s).canonical()
    assert a == b and a != c
