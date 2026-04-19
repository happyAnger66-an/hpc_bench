"""Solution data models for kernel implementations."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString


class SupportedLanguages(str, Enum):
    """Supported programming languages for solution implementations."""

    PYTORCH = "pytorch"
    TRITON = "triton"
    CUTE_DSL = "cute_dsl"
    CUTILE = "cutile"
    CUDNN_FRONTEND = "cudnn_frontend"
    CUTLASS = "cutlass"
    CUDNN = "cudnn"
    CUBLAS = "cublas"
    CUDA_CPP = "cuda_cpp"


class SupportedHardware(str, Enum):
    """Supported hardware targets for solution implementations."""

    B200 = "B200"
    LOCAL = "LOCAL"


class SupportedBindings(str, Enum):
    """Supported bindings for C++/CUDA solution implementations."""

    TORCH = "torch"


class SourceFile(BaseModelWithDocstrings):
    """A single source code file in a solution implementation."""

    path: NonEmptyString
    content: NonEmptyString

    @model_validator(mode="after")
    def _validate_source_path(self) -> "SourceFile":
        """Validate source path for security."""
        src_path = Path(self.path)
        if src_path.is_absolute():
            raise ValueError(
                f"Invalid source path (absolute path not allowed): {self.path}"
            )
        if ".." in src_path.parts:
            raise ValueError(
                f"Invalid source path (parent directory traversal not allowed): {self.path}"
            )
        return self


class CompileOptions(BaseModelWithDocstrings):
    """Compiler and linker flags for C++/CUDA solutions."""

    cflags: list[str] = Field(default_factory=list)
    cuda_cflags: list[str] = Field(
        default_factory=lambda: ["-O3", "--use_fast_math"]
    )
    ld_flags: list[str] = Field(default_factory=lambda: ["-lcuda"])


class BuildSpec(BaseModelWithDocstrings):
    """Build specification for a solution implementation."""

    languages: list[SupportedLanguages]
    target_hardware: list[SupportedHardware] = Field(min_length=1)
    entry_point: NonEmptyString
    destination_passing_style: bool = True
    binding: Optional[SupportedBindings] = None
    dependencies: list[NonEmptyString] = Field(default_factory=list)
    compile_options: Optional[CompileOptions] = None

    @model_validator(mode="after")
    def _validate_entry_point(self) -> "BuildSpec":
        """Validate entry_point format."""
        if self.entry_point.count("::") != 1:
            raise ValueError(
                f"Invalid entry point format: {self.entry_point}. Expected "
                '"<file_path>::<function_name>".'
            )
        return self

    @model_validator(mode="after")
    def _validate_languages(self) -> "BuildSpec":
        """Validate languages support matrix."""
        python_languages = [
            SupportedLanguages.PYTORCH,
            SupportedLanguages.TRITON,
            SupportedLanguages.CUTE_DSL,
            SupportedLanguages.CUTILE,
            SupportedLanguages.CUDNN_FRONTEND,
        ]
        cpp_languages = [
            SupportedLanguages.CUTLASS,
            SupportedLanguages.CUDNN,
            SupportedLanguages.CUBLAS,
            SupportedLanguages.CUDA_CPP,
        ]

        included_python_langs = [
            lang for lang in self.languages if lang in python_languages
        ]
        included_cpp_langs = [
            lang for lang in self.languages if lang in cpp_languages
        ]
        if len(included_cpp_langs) and len(included_python_langs):
            raise ValueError(
                f"C++ and Python cannot be mixed, but got {included_cpp_langs} and {included_python_langs}"
            )

        # Validate entry point file suffix matches the language category
        entry_file = self.entry_point.split("::")[0]
        suffix = Path(entry_file).suffix
        if included_cpp_langs and suffix not in (
            ".cu", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cuh"
        ):
            raise ValueError(
                f"C++ languages require a C++/CUDA entry point file, "
                f"but got '{entry_file}' (suffix '{suffix}')"
            )
        if included_python_langs and suffix != ".py":
            raise ValueError(
                f"Python languages require a .py entry point file, "
                f"but got '{entry_file}' (suffix '{suffix}')"
            )
        return self


class Solution(BaseModelWithDocstrings):
    """A concrete implementation for a given Definition."""

    name: NonEmptyString
    definition: NonEmptyString
    author: NonEmptyString
    spec: BuildSpec
    sources: list[SourceFile] = Field(min_length=1)
    description: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _validate_source_path_entry_point(self) -> "Solution":
        """Validate source file paths for uniqueness and entry file existence."""
        seen_paths = set()
        for source in self.sources:
            if source.path in seen_paths:
                raise ValueError(f"Duplicate source path '{source.path}'")
            seen_paths.add(source.path)

        entry_file = self.spec.entry_point.split("::")[0]
        if entry_file not in seen_paths:
            raise ValueError(f"Entry source file '{entry_file}' not found in sources")

        return self

    def get_entry_path(self) -> Path:
        """Extract the file path from the entry point specification."""
        return Path(self.spec.entry_point.split("::")[0])

    def get_entry_symbol(self) -> str:
        """Extract the function/symbol name from the entry point specification."""
        return self.spec.entry_point.split("::")[-1]

    def get_entry_source(self) -> Optional[SourceFile]:
        """Get the entry source file specified in the build spec."""
        entry_path = self.spec.entry_point.split("::")[0]
        for source in self.sources:
            if source.path == entry_path:
                return source
        return None
