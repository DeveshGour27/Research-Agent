"""Safe arithmetic calculator tool.

Uses Python's ``ast`` module to evaluate expressions without calling
``eval``, so no arbitrary code can be executed.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from app.exceptions import ToolExecutionError
from app.tools.base import BaseTool


class CalculatorTool(BaseTool):
    """Evaluate a safe arithmetic expression and return the numeric result.

    Supported operators: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``.
    Only numeric literals are allowed — no variables, function calls, or
    attribute access.
    """

    name = "calculate"
    description = (
        "Evaluate a safe arithmetic expression and return the numeric result. "
        "Supports +, -, *, /, //, %, ** operators on numeric literals only."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "Arithmetic expression to evaluate, e.g. '2 + 3 * 4' or '10 / 2'."
                ),
            }
        },
        "required": ["expression"],
    }

    # Mapping from AST node types to Python operator functions.
    _BINARY_OPS: dict[type, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPS: dict[type, Any] = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def execute(self, **kwargs: object) -> str:
        expression = str(kwargs.get("expression", "")).strip()
        if not expression:
            raise ToolExecutionError(
                "The 'expression' argument must not be empty.",
                tool_name=self.name,
            )

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval(tree.body)
        except ToolExecutionError:
            raise
        except SyntaxError as error:
            raise ToolExecutionError(
                f"Invalid expression syntax: {error.msg}",
                tool_name=self.name,
                details={"expression": expression},
            )
        except Exception as error:
            raise ToolExecutionError(
                f"Failed to evaluate expression: {error}",
                tool_name=self.name,
                details={"expression": expression},
            )

        # Present whole-number floats without the trailing ".0".
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    def _eval(self, node: ast.expr) -> int | float:
        """Recursively evaluate a single AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ToolExecutionError(
                f"Unsupported constant type: {type(node.value).__name__}.",
                tool_name=self.name,
            )

        if isinstance(node, ast.BinOp):
            op_fn = self._BINARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ToolExecutionError(
                    f"Unsupported binary operator: {type(node.op).__name__}.",
                    tool_name=self.name,
                )
            left = self._eval(node.left)
            right = self._eval(node.right)
            try:
                return op_fn(left, right)
            except ZeroDivisionError:
                raise ToolExecutionError(
                    "Division by zero.",
                    tool_name=self.name,
                    details={"expression": ast.unparse(node)},
                )

        if isinstance(node, ast.UnaryOp):
            op_fn = self._UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ToolExecutionError(
                    f"Unsupported unary operator: {type(node.op).__name__}.",
                    tool_name=self.name,
                )
            return op_fn(self._eval(node.operand))

        raise ToolExecutionError(
            f"Unsupported expression node type: {type(node).__name__}.",
            tool_name=self.name,
        )
