from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .base import BaseDataset, Sample, normalize_documents, normalize_text, write_json, write_jsonl


class MultiXScienceDataset(BaseDataset):
    DATASET_NAME = "multi_xscience"
    LANGUAGE = "en"


def clean_citations(text: Any) -> str:
    text = normalize_text(text)
    text = re.sub(r"@cite_\d+", "", text)
    return normalize_text(text)


def iter_multixscience_samples(
    raw_root: str | Path,
    split: str = "test",
    limit: Optional[int] = None,
) -> Iterator[Sample]:
    raw_root = Path(raw_root)
    raw_path = raw_root / "raw" / f"{split}.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"Multi-XScience raw split file not found: {raw_path}")

    with raw_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    for index, item in enumerate(rows):
        if limit is not None and index >= limit:
            break

        ref_abstract = item.get("ref_abstract") or item.get("ref_abstracts") or {}
        ref_documents = []
        ref_ids = []

        if isinstance(ref_abstract, dict):
            for ref_id, ref_payload in ref_abstract.items():
                ref_ids.append(ref_id)
                if isinstance(ref_payload, dict):
                    ref_documents.append(ref_payload.get("abstract", ""))
                else:
                    ref_documents.append(ref_payload)
        elif isinstance(ref_abstract, list):
            ref_documents.extend(ref_abstract)

        documents = normalize_documents([item.get("abstract", ""), *ref_documents])
        sample_id = normalize_text(item.get("aid")) or f"multi_xscience_{split}_{index:06d}"

        yield {
            "sample_id": sample_id,
            "dataset_name": MultiXScienceDataset.DATASET_NAME,
            "split": split,
            "language": MultiXScienceDataset.LANGUAGE,
            "documents": [clean_citations(document) for document in documents],
            "reference_summary": clean_citations(item.get("related_work", "")),
            "metadata": {
                "source_dataset": "Multi-XScience",
                "raw_index": index,
                "raw_source_file": str(raw_path),
                "aid": item.get("aid"),
                "mid": item.get("mid"),
                "ref_ids": ref_ids,
                "num_documents": len(documents),
            },
        }


def prepare_multixscience(
    raw_root: str | Path,
    processed_root: str | Path,
    split: str = "test",
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    processed_root = Path(processed_root)
    samples = list(iter_multixscience_samples(raw_root=raw_root, split=split))
    output_path = processed_root / f"{split}.jsonl"
    sample_count = write_jsonl(output_path, samples)

    metadata = {
        "dataset_name": MultiXScienceDataset.DATASET_NAME,
        "split": split,
        "language": MultiXScienceDataset.LANGUAGE,
        "sample_count": sample_count,
        "expected_count": expected_count,
        "processed_file": str(output_path),
        "schema": sorted(BaseDataset.REQUIRED_FIELDS),
        "count_matches_expected": expected_count is None or sample_count == expected_count,
    }
    write_json(processed_root / "metadata.json", metadata)
    return metadata
