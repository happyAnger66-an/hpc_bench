"""JSON utilities for hpc_bench data models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json_file(path: Path | str, data: Any) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json_file(path: Path | str) -> Any:
    """Load data from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_jsonl_file(path: Path | str, items: list[Any]) -> None:
    """Save list of items to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl_file(path: Path | str) -> list[Any]:
    """Load list of items from JSONL file."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def append_jsonl_file(path: Path | str, item: Any) -> None:
    """Append a single item to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
