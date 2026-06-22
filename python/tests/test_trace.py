from tax._frontend.trace import TraceBuilder, Tracer
from tax._frontend.ir import Var, Const, Op
from tax._frontend.scheme import Isotropic
import tax

def test_tracer_records_ops_not_eager():
    b = TraceBuilder()
    s = Isotropic(5, 1)
    x = Tracer(b, b.add(Var(0, s)), s)
    y = tax.sin(x) * tax.exp(x)                 # 2 unary ops + 1 mul
    assert isinstance(y, Tracer)
    kinds = [type(n).__name__ for n in b.nodes]
    assert kinds == ["Var", "Op", "Op", "Op"]
    assert b.nodes[1].opcode == "sin" and b.nodes[1].operands == (0,)
    assert b.nodes[2].opcode == "exp" and b.nodes[2].operands == (0,)
    assert b.nodes[3].opcode == "mul" and b.nodes[3].operands == (1, 2)
    assert y.node == 3

def test_tracer_scalar_becomes_const():
    b = TraceBuilder()
    s = Isotropic(3, 1)
    x = Tracer(b, b.add(Var(0, s)), s)
    y = 2.0 * x                                 # scalar -> Const node, then mul
    assert isinstance(b.nodes[1], Const) and b.nodes[1].value == 2.0
    assert b.nodes[2].opcode == "mul"
