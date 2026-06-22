from tax._frontend.trace import TraceBuilder, Tracer, ArrayTracer
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

def test_array_tracer_concatenate_and_dot():
    b = TraceBuilder()
    s = Isotropic(2, 2)
    X = ArrayTracer([Tracer(b, b.add(Var(i, s)), s) for i in range(2)], s)
    assert isinstance(X[0], Tracer) and len(X) == 2
    Y = tax.concatenate([X[0] * X[1], X[0] + X[1]])
    assert isinstance(Y, ArrayTracer) and len(Y) == 2
    d = tax.dot(X, X)                                  # x0*x0 + x1*x1 -> scalar Tracer
    assert isinstance(d, Tracer)

def test_array_tracer_elementwise_and_math():
    b = TraceBuilder()
    s = Isotropic(3, 2)
    X = ArrayTracer([Tracer(b, b.add(Var(i, s)), s) for i in range(2)], s)
    S = tax.sin(X)                                     # elementwise -> ArrayTracer
    assert isinstance(S, ArrayTracer) and len(S) == 2
    Z = 2.0 * X + X                                    # broadcast scalar + elementwise add
    assert isinstance(Z, ArrayTracer) and len(Z) == 2
