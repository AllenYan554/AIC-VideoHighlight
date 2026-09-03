"""Conservative JSONL reader that makes no implicit annotation assumptions."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class AnnotationSchemaError(ValueError):
    pass


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream top-level JSON objects without assigning label semantics."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationSchemaError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise AnnotationSchemaError(f"line {line_number} is not a JSON object")
            yield record


def read_explicit_field(record: dict[str, Any], field_path: str | None) -> Any:
    """Read only a caller-confirmed field path; never guess annotation semantics."""
    if not field_path:
        raise AnnotationSchemaError(
            "annotation field is not configured; confirm annotation semantics before evaluation"
        )
    value: Any = record
    for part in field_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise AnnotationSchemaError(f"configured annotation field does not exist: {field_path}")
        value = value[part]
    return value
