from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .base import BaseDataset, Sample, normalize_documents, normalize_text, write_json, write_jsonl


class MultiNewsDataset(BaseDataset):
    DATASET_NAME = "multi_news"
    LANGUAGE = "en"


def iter_multinews_samples(
    raw_root: str | Path,
    split: str = "test",
    limit: Optional[int] = None,
) -> Iterator[Sample]:
    raw_root = Path(raw_root)
    raw_dir = raw_root / "raw"
    source_path = raw_dir / f"{split}.src.cleaned"
    target_path = raw_dir / f"{split}.tgt"

    if not source_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            f"Multi-News raw split files not found: {source_path}, {target_path}"
        )

    emitted = 0
    with source_path.open("r", encoding="utf-8") as source_file, target_path.open(
        "r", encoding="utf-8"
    ) as target_file:
        for index, (source_line, target_line) in enumerate(zip(source_file, target_file)):
            if limit is not None and emitted >= limit:
                break

            documents = normalize_documents(source_line.split("|||||"))
            if not documents:
                continue

            sample_id = f"multi_news_{split}_{index:06d}"
            emitted += 1

            yield {
                "sample_id": sample_id,
                "dataset_name": MultiNewsDataset.DATASET_NAME,
                "split": split,
                "language": MultiNewsDataset.LANGUAGE,
                "documents": documents,
                "reference_summary": normalize_text(target_line),
                "metadata": {
                    "source_dataset": "Multi-News",
                    "raw_index": index,
                    "raw_source_file": str(source_path),
                    "raw_target_file": str(target_path),
                    "num_documents": len(documents),
                },
            }


def prepare_multinews(
    raw_root: str | Path,
    processed_root: str | Path,
    split: str = "test",
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    processed_root = Path(processed_root)
    raw_dir = Path(raw_root) / "raw"
    raw_source_path = raw_dir / f"{split}.src.cleaned"
    raw_target_path = raw_dir / f"{split}.tgt"
    raw_count = sum(1 for _ in raw_source_path.open("r", encoding="utf-8"))
    target_count = sum(1 for _ in raw_target_path.open("r", encoding="utf-8"))
    samples = list(iter_multinews_samples(raw_root=raw_root, split=split))
    output_path = processed_root / f"{split}.jsonl"
    sample_count = write_jsonl(output_path, samples)

    metadata = {
        "dataset_name": MultiNewsDataset.DATASET_NAME,
        "split": split,
        "language": MultiNewsDataset.LANGUAGE,
        "sample_count": sample_count,
        "raw_source_count": raw_count,
        "raw_target_count": target_count,
        "skipped_empty_document_rows": raw_count - sample_count,
        "expected_count": expected_count,
        "processed_file": str(output_path),
        "schema": sorted(BaseDataset.REQUIRED_FIELDS),
        "count_matches_expected": expected_count is None or sample_count == expected_count,
    }
    write_json(processed_root / "metadata.json", metadata)
    return metadata
