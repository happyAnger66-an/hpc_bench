"""Definition data models for kernel specifications."""

from __future__ import annotations

import ast
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from pydantic import Field, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt

if TYPE_CHECKING:
    import torch


class DType(str, Enum):
    """Supported data types for tensors."""

    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT4_E2M1 = "float4_e2m1"
    FLOAT4_E2M1FN_X2 = "float4_e2m1fn_x2"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    BOOL = "bool"


class AxisConst(BaseModelWithDocstrings):
    """Constant axis with a fixed value."""

    type: Literal["const"] = "const"
    value: NonNegativeInt
    description: Optional[str] = None


class AxisVar(BaseModelWithDocstrings):
    """Variable axis that can be specified at runtime."""

    type: Literal["var"] = "var"
    description: Optional[str] = None


class AxisExpr(BaseModelWithDocstrings):
    """Expression axis that is computed from other axes."""

    type: Literal["expr"] = "expr"
    expression: NonEmptyString
    description: Optional[str] = None


AxisSpec = Union[AxisConst, AxisVar, AxisExpr]


class TensorSpec(BaseModelWithDocstrings):
    """Specification for a tensor including shape and data type."""

    shape: Optional[list[NonEmptyString]]
    dtype: DType
    description: Optional[str] = None


class Definition(BaseModelWithDocstrings):
    """Complete definition of a computational workload."""

    name: NonEmptyString
    op_type: Optional[NonEmptyString] = Field(default=None)
    axes: dict[NonEmptyString, AxisSpec]
    inputs: dict[NonEmptyString, TensorSpec]
    outputs: dict[NonEmptyString, TensorSpec]
    reference: NonEmptyString
    constraints: list[str] = Field(default_factory=list)
    custom_inputs_entrypoint: Optional[NonEmptyString] = Field(default=None)
    description: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _validate_reference_code(self) -> "Definition":
        """Validate that reference contains valid Python code with a 'run' function."""
        try:
            mod = ast.parse(self.reference, mode="exec")
        except SyntaxError as e:
            raise ValueError(f"Reference must be valid Python code: {e}") from e

        has_run_func = any(
            isinstance(node, ast.FunctionDef) and node.name == "run"
            for node in mod.body
        )
        if not has_run_func:
            raise ValueError("Reference must define a top-level function named 'run'")
        return self

    @model_validator(mode="after")
    def _validate_reference_inputs_match(self) -> "Definition":
        """Validate that run() parameter names match the inputs keys in order."""
        try:
            tree = ast.parse(self.reference, mode="exec")
        except SyntaxError:
            return self

        run_func = next(
            (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"),
            None,
        )
        if run_func is None:
            return self

        args = run_func.args
        param_names: list[str] = [a.arg for a in (args.posonlyargs or [])] + [
            a.arg for a in args.args
        ]
        input_names = list(self.inputs.keys())

        if len(param_names) != len(input_names):
            raise ValueError(
                f"run() has {len(param_names)} parameter(s) {param_names} but "
                f"definition 'inputs' has {len(input_names)} entries "
                f"{input_names}. They must match exactly."
            )

        mismatched = [(p, i) for p, i in zip(param_names, input_names) if p != i]
        if mismatched:
            raise ValueError(
                f"run() parameter names don't match definition 'inputs' keys. "
                f"Mismatches (run_param -> input_key): {mismatched}. "
                f"run() params: {param_names}, inputs: {input_names}."
            )

        return self

    @model_validator(mode="after")
    def _verify_custom_inputs_entrypoint(self) -> "Definition":
        """Verify that custom inputs entrypoint is valid if specified."""
        if self.custom_inputs_entrypoint is None:
            return self

        if not self.custom_inputs_entrypoint.isidentifier():
            raise ValueError(
                f"custom_inputs_entrypoint must be a valid Python identifier, "
                f"got: {self.custom_inputs_entrypoint!r}"
            )

        try:
            tree = ast.parse(self.reference, mode="exec")
        except SyntaxError:
            return self

        has_entrypoint = any(
            isinstance(node, ast.FunctionDef)
            and node.name == self.custom_inputs_entrypoint
            for node in tree.body
        )
        if not has_entrypoint:
            raise ValueError(
                f"custom_inputs_entrypoint '{self.custom_inputs_entrypoint}' "
                f"is not defined as a top-level function in the reference code"
            )

        return self

    @model_validator(mode="after")
    def _validate_input_output_names(self) -> "Definition":
        """Validate that input and output names are unique."""
        if set(self.inputs.keys()) & set(self.outputs.keys()):
            raise ValueError("Input and output names must not overlap")
        return self

    @model_validator(mode="after")
    def _validate_tensor_axis_references(self) -> "Definition":
        """Validate that tensor shapes reference defined axes."""
        all_tensors = {**self.inputs, **self.outputs}

        for tensor_name, tensor_spec in all_tensors.items():
            if tensor_spec.shape is None:
                continue
            for axis_name in tensor_spec.shape:
                if axis_name.isdigit():
                    continue
                if axis_name not in self.axes:
                    tensor_type = (
                        "input" if tensor_name in self.inputs else "output"
                    )
                    raise ValueError(
                        f'{tensor_type.capitalize()} "{tensor_name}" references undefined '
                        f'axis "{axis_name}".'
                    )
        return self

    @cached_property
    def const_axes(self) -> dict[str, int]:
        """Get all constant axes and their values."""
        return {
            name: axis.value
            for name, axis in self.axes.items()
            if isinstance(axis, AxisConst)
        }

    @cached_property
    def var_axes(self) -> list[str]:
        """Get all variable axis names."""
        return [name for name, axis in self.axes.items() if isinstance(axis, AxisVar)]

    @cached_property
    def expr_axes(self) -> dict[str, AxisExpr]:
        """Get all expression axis names."""
        return {name: axis for name, axis in self.axes.items() if isinstance(axis, AxisExpr)}

    def _resolve_expression(self, expression: str, resolved: dict[str, int]) -> int:
        """Resolve a mathematical expression to an integer value."""
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {expression}") from e

        def eval_node(node: ast.AST) -> int:
            if isinstance(node, ast.Constant):
                return int(node.value)
            elif isinstance(node, ast.Name):
                if node.id in resolved:
                    return resolved[node.id]
                raise ValueError(f"Undefined variable: {node.id}")
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left // right
                elif isinstance(node.op, ast.FloorDiv):
                    return left // right
                elif isinstance(node.op, ast.Mod):
                    return left % right
                elif isinstance(node.op, ast.Pow):
                    return left ** right
                else:
                    raise ValueError(f"Unsupported operator: {type(node.op)}")
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +operand
                elif isinstance(node.op, ast.USub):
                    return -operand
                else:
                    raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            elif isinstance(node, ast.Call):
                raise ValueError("Function calls not supported in expressions")
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")

        return eval_node(tree.body)

    def get_resolved_axes_values(self, var_axes_values: dict[str, int]) -> dict[str, int]:
        """Get concrete axis values from variable axis values, resolving expressions."""
        resolved_axes_values: dict[str, int] = self.const_axes.copy()
        resolved_axes_values.update(var_axes_values)

        for name, axis in self.expr_axes.items():
            resolved_axes_values[name] = self._resolve_expression(
                axis.expression, resolved_axes_values
            )

        return resolved_axes_values

    def get_input_shapes(
        self, var_axes_values: Optional[dict[str, int]] = None
    ) -> dict[str, Optional[tuple[int, ...]]]:
        """Get concrete input shapes given variable axis values."""
        resolved = self.get_resolved_axes_values(var_axes_values or {})
        shapes = {}
        for name, spec in self.inputs.items():
            if spec.shape is None:
                shapes[name] = None
            else:
                shape = []
                for axis_name in spec.shape:
                    if axis_name.isdigit():
                        shape.append(int(axis_name))
                    else:
                        shape.append(resolved[axis_name])
                shapes[name] = tuple(shape)
        return shapes

    def get_output_shapes(
        self, var_axes_values: Optional[dict[str, int]] = None
    ) -> dict[str, Optional[tuple[int, ...]]]:
        """Get concrete output shapes given variable axis values."""
        resolved = self.get_resolved_axes_values(var_axes_values or {})
        shapes = {}
        for name, spec in self.outputs.items():
            if spec.shape is None:
                shapes[name] = None
            else:
                shape = []
                for axis_name in spec.shape:
                    if axis_name.isdigit():
                        shape.append(int(axis_name))
                    else:
                        shape.append(resolved[axis_name])
                shapes[name] = tuple(shape)
        return shapes

    @cached_property
    def torch_input_dtypes(self) -> list["torch.dtype"]:
        """Get the torch data types of the input tensors."""
        from .dtypes import dtype_str_to_torch_dtype
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.inputs.values()]

    @cached_property
    def torch_output_dtypes(self) -> list["torch.dtype"]:
        """Get the torch data types of the output tensors."""
        from .dtypes import dtype_str_to_torch_dtype
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.outputs.values()]
