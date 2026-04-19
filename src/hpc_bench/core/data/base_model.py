"""Base model classes for hpc_bench data models."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, PlainValidator


def _validate_non_empty_string(v: str) -> str:
    """Validate that the string is not empty."""
    if not v or not v.strip():
        raise ValueError("String must be non-empty")
    return v


def _validate_non_negative_int(v: int) -> int:
    """Validate that the integer is non-negative."""
    if v < 0:
        raise ValueError("Integer must be non-negative")
    return v


NonEmptyString = Annotated[str, PlainValidator(_validate_non_empty_string)]
NonNegativeInt = Annotated[int, PlainValidator(_validate_non_negative_int)]


class BaseModelWithDocstrings(BaseModel):
    """Base model that preserves docstrings as field descriptions."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        populate_by_name=True,
    )

    def model_dump_json(self, **kwargs: Any) -> str:
        """Dump model to JSON string."""
        return super().model_dump_json(**kwargs)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump model to dictionary."""
        return super().model_dump(**kwargs)
