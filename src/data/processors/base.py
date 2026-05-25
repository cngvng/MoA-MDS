from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


Sample = Dict[str, Any]


def normalize_text(text: Any) -> str:
    """Normalize raw dataset text without applying task-specific rewriting."""
    if text is None:
        return ""

    normalized = str(text).replace("\ufeff", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("NEWLINE_CHAR", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def normalize_documents(documents: Iterable[Any]) -> List[str]:
    normalized = [normalize_text(document) for document in documents]
    return [document for document in normalized if document]


def natural_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Sample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Sample]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parse_info_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    metadata: Dict[str, str] = {}
    for line in read_text(path).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower().replace(" ", "_")] = normalize_text(value)
    return metadata


class BaseDataset:
    """Minimal JSONL-backed dataset with the unified MoA sample schema."""

    REQUIRED_FIELDS = {
        "sample_id",
        "dataset_name",
        "split",
        "language",
        "documents",
        "reference_summary",
        "metadata",
    }

    def __init__(self, processed_path: str | Path, split: str = "test") -> None:
        self.processed_path = Path(processed_path)
        self.split = split
        self.data = read_jsonl(self.processed_path)

        for index, sample in enumerate(self.data):
            self.validate_sample(sample, index=index)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Sample:
        return self.data[idx]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.data)

    @classmethod
    def validate_sample(cls, sample: Sample, index: Optional[int] = None) -> None:
        missing = cls.REQUIRED_FIELDS - set(sample)
        if missing:
            location = f" at index {index}" if index is not None else ""
            raise ValueError(f"Sample{location} missing required fields: {sorted(missing)}")

        if not isinstance(sample["documents"], list) or not all(
            isinstance(document, str) for document in sample["documents"]
        ):
            raise TypeError("Sample documents must be list[str]")

        if not sample["documents"]:
            raise ValueError("Sample documents must not be empty")

        if not isinstance(sample["metadata"], dict):
            raise TypeError("Sample metadata must be a dict")

    def collate_fn(self, batch: List[Sample]) -> Dict[str, Any]:
        return {
            "sample_id": [item["sample_id"] for item in batch],
            "dataset_name": [item["dataset_name"] for item in batch],
            "split": [item["split"] for item in batch],
            "language": [item["language"] for item in batch],
            "documents": [item["documents"] for item in batch],
            "reference_summary": [item["reference_summary"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
        }
