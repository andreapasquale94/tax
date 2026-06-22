from __future__ import annotations
from .._frontend.ir import Var, Const, Op

CPP_EXPR = {
    "add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})",
    "div": "({0} / {1})", "neg": "(-{0})",
    "sin": "sin({0})", "cos": "cos({0})", "tan": "tan({0})",
    "asin": "asin({0})", "acos": "acos({0})", "atan": "atan({0})",
    "sinh": "sinh({0})", "cosh": "cosh({0})", "tanh": "tanh({0})",
    "asinh": "asinh({0})", "acosh": "acosh({0})", "atanh": "atanh({0})",
    "exp": "exp({0})", "log": "log({0})", "sqrt": "sqrt({0})", "cbrt": "cbrt({0})",
    "square": "square({0})", "cube": "cube({0})", "erf": "erf({0})",
    "reciprocal": "reciprocal({0})",
    "pow": "pow({0}, {1})", "atan2": "atan2({0}, {1})",
}

def emit(graph) -> str:
    lines = [
        "#include <tax/tax.hpp>",
        "#include <algorithm>",
        "using namespace tax;",
        'extern "C" int tax_kernel(const double* const* ins, '
        "double* const* outs) noexcept {",
    ]
    for i, node in enumerate(graph.nodes):
        cpp_type = node.scheme.cpp_type_string()
        expansion_type = f"tax::TaylorExpansion<double, {cpp_type}>"
        if isinstance(node, Var):
            n = node.scheme.n_coeff
            lines.append(f"    {expansion_type}::Data d{i}; "
                         f"std::copy_n(ins[{node.slot}], {n}, d{i}.data());")
            lines.append(f"    {expansion_type} n{i}{{d{i}}};")
        elif isinstance(node, Const):
            lines.append(f"    auto n{i} = {expansion_type}::constant({node.value!r});")
        elif isinstance(node, Op):
            if node.opcode.startswith("powint:"):
                n = int(node.opcode.split(":", 1)[1])
                expr = f"pow(n{node.operands[0]}, {n})"     # C++ pow(TE, int) = seriesPowInt
            else:
                expr = CPP_EXPR[node.opcode].format(*[f"n{o}" for o in node.operands])
            lines.append(f"    auto n{i} = {expr};")
        else:
            raise TypeError(f"unknown node type {type(node)!r}")
    for j, o in enumerate(graph.outputs):
        n = graph.nodes[o].scheme.n_coeff
        lines.append(f"    std::copy_n(n{o}.coefficients().data(), {n}, outs[{j}]);")
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"
