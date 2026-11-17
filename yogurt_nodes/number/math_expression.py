import ast
import math
import operator
from typing import Any

from ..utils import ANY_TYPE


OUTPUT_COUNT = 6
VARIABLE_NAMES = tuple(f"x{i}" for i in range(1, 33))
CONSTANT_NAMES = ("pi", "e", "tau")
FUNCTION_NAMES = (
    "abs",
    "min",
    "max",
    "round",
    "sum",
    "len",
    "int",
    "float",
    "bool",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "radians",
    "degrees",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "sqrt",
    "cbrt",
    "pow",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log10",
    "log2",
    "log1p",
    "floor",
    "ceil",
    "trunc",
    "fabs",
    "fmod",
    "remainder",
    "copysign",
    "hypot",
    "dist",
    "isfinite",
    "isinf",
    "isnan",
    "isclose",
    "factorial",
    "comb",
    "perm",
    "gcd",
    "lcm",
    "isqrt",
    "gamma",
    "lgamma",
    "erf",
    "erfc",
    "fsum",
    "prod",
)
SYMBOL_HELP = (
    "Variables: x1..x32. Constants: pi, e, tau. "
    "Functions: " + ", ".join(FUNCTION_NAMES) + ". "
    "Operators: + - * / // % **, comparisons == != < <= > >=, boolean and/or/not. "
    "Each non-empty expression line maps to out1..out6; empty or missing lines return None."
)

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_COMPARE_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "bool": bool,
    "pow": pow,
    **{name: getattr(math, name) for name in FUNCTION_NAMES if hasattr(math, name)},
}
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "true": True,
    "false": False,
}


class _ExpressionEvaluator(ast.NodeVisitor):
    def __init__(self, variables: dict[str, Any]):
        self.variables = variables

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")

    def visit_Name(self, node: ast.Name):
        if node.id in self.variables:
            return self.variables[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"unknown name: {node.id}")

    def visit_BinOp(self, node: ast.BinOp):
        operator_fn = _BINARY_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return operator_fn(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not self.visit(node.operand)
        operator_fn = _UNARY_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return operator_fn(self.visit(node.operand))

    def visit_BoolOp(self, node: ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = None
            for value in node.values:
                result = self.visit(value)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = None
            for value in node.values:
                result = self.visit(value)
                if result:
                    return result
            return result
        raise ValueError(f"unsupported boolean operator: {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            operator_fn = _COMPARE_OPERATORS.get(type(op))
            if operator_fn is None:
                raise ValueError(f"unsupported comparison: {type(op).__name__}")
            if not operator_fn(left, right):
                return False
            left = right
        return True

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("unsupported function call")
        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise ValueError(f"unsupported function: {node.func.id}")
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("unsupported function call")
            kwargs[keyword.arg] = self.visit(keyword.value)
        return function(*(self.visit(arg) for arg in node.args), **kwargs)

    def visit_List(self, node: ast.List):
        return [self.visit(item) for item in node.elts]

    def visit_Tuple(self, node: ast.Tuple):
        return tuple(self.visit(item) for item in node.elts)

    def generic_visit(self, node: ast.AST):
        raise ValueError(f"unsupported expression: {type(node).__name__}")


def _normalize_input(value: Any) -> Any:
    if value is None:
        return 0.0
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float, bool)):
        return value
    return float(value)


def _evaluate_expression(expression: str, variables: dict[str, Any]) -> Any:
    tree = ast.parse(expression, mode="eval")
    return _ExpressionEvaluator(variables).visit(tree)


class MathExpression:
    """Math Expression node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"multiline": True, "tooltip": SYMBOL_HELP}),
            },
            "optional": {
                name: (ANY_TYPE, {"tooltip": f"Value for variable {name}"})
                for name in VARIABLE_NAMES
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (ANY_TYPE,) * OUTPUT_COUNT
    RETURN_NAMES = tuple(f"out{i}" for i in range(1, OUTPUT_COUNT + 1))
    FUNCTION = "execute"
    OUTPUT_NODE = False

    _NODE_NAME = "Math Expression"
    DESCRIPTION = SYMBOL_HELP

    def execute(self, expression: str, **kwargs: Any):
        variables = {
            name: _normalize_input(kwargs.get(name, 0.0))
            for name in VARIABLE_NAMES
        }
        outputs = []
        lines = expression.splitlines()
        for index in range(OUTPUT_COUNT):
            if index >= len(lines) or not lines[index].strip():
                outputs.append(None)
                continue
            outputs.append(_evaluate_expression(lines[index].strip(), variables))
        return tuple(outputs)
