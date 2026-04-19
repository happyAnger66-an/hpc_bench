"""Data layer with strongly-typed models for hpc_bench."""

from .definition import (
    AxisConst,
    AxisExpr,
    AxisSpec,
    AxisVar,
    Definition,
    TensorSpec,
)
from .solution import (
    BuildSpec,
    CompileOptions,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedHardware,
    SupportedLanguages,
)
from .trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from .workload import (
    CustomInput,
    InputSpec,
    RandomInput,
    SafetensorsInput,
    ScalarInput,
    ToleranceSpec,
    Workload,
)
from .json_utils import (
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_file,
    append_jsonl_file,
)

__all__ = [
    # Definition types
    "AxisConst",
    "AxisExpr",
    "AxisSpec",
    "AxisVar",
    "TensorSpec",
    "Definition",
    # Solution types
    "SourceFile",
    "BuildSpec",
    "CompileOptions",
    "SupportedBindings",
    "SupportedHardware",
    "SupportedLanguages",
    "Solution",
    # Workload types
    "ToleranceSpec",
    "CustomInput",
    "RandomInput",
    "ScalarInput",
    "SafetensorsInput",
    "InputSpec",
    "Workload",
    # Trace types
    "Correctness",
    "Performance",
    "Environment",
    "Evaluation",
    "EvaluationStatus",
    "Trace",
    # JSON utilities
    "save_json_file",
    "load_json_file",
    "save_jsonl_file",
    "load_jsonl_file",
    "append_jsonl_file",
]
