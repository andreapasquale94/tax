from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Var:
    slot: int
    scheme: object

@dataclass(frozen=True)
class Const:
    value: float
    scheme: object

@dataclass(frozen=True)
class Op:
    opcode: str
    operands: tuple
    scheme: object

@dataclass
class Graph:
    nodes: list
    outputs: list
    n_inputs: int

    def canonical(self) -> str:
        parts = []
        for i, n in enumerate(self.nodes):
            if isinstance(n, Var):
                parts.append(f"{i}=var({n.slot},{n.scheme.descriptor_hash()})")
            elif isinstance(n, Const):
                parts.append(f"{i}=const({float(n.value).hex()},{n.scheme.descriptor_hash()})")
            elif isinstance(n, Op):
                ops = ",".join(str(o) for o in n.operands)
                parts.append(f"{i}=op({n.opcode},{ops},{n.scheme.descriptor_hash()})")
            else:
                raise TypeError(f"unknown node type {type(n)!r}")
        parts.append("out:" + ",".join(str(o) for o in self.outputs))
        return ";".join(parts)

def single_op_graph(opcode: str, operand_schemes: list, result_scheme) -> Graph:
    nodes = [Var(slot=i, scheme=s) for i, s in enumerate(operand_schemes)]
    op_index = len(nodes)
    nodes.append(Op(opcode=opcode, operands=tuple(range(len(operand_schemes))),
                    scheme=result_scheme))
    return Graph(nodes=nodes, outputs=[op_index], n_inputs=len(operand_schemes))
