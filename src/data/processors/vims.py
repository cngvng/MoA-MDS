from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .base import (
    BaseDataset,
    Sample,
    natural_key,
    normalize_documents,
    normalize_text,
    read_text,
    write_json,
    write_jsonl,
)


class ViMsDataset(BaseDataset):
    DATASET_NAME = "vims"
    LANGUAGE = "vi"


def _cluster_id(path: Path) -> int:
    suffix = path.name.split("_")[-1]
    return int(suffix) if suffix.isdigit() else 0


def _vims_root(raw_root: Path) -> Path:
    root = raw_root / "raw" / "extracted" / "ViMs"
    if not root.exists():
        raise FileNotFoundError(f"ViMs extracted directory not found: {root}")
    return root


def _iter_cluster_dirs(raw_root: Path) -> list[Path]:
    original_root = _vims_root(raw_root) / "original"
    if not original_root.exists():
        raise FileNotFoundError(f"ViMs original directory not found: {original_root}")
    return sorted(
        [path for path in original_root.iterdir() if path.is_dir() and path.name.startswith("Cluster_")],
        key=natural_key,
    )


def _parse_original_document(path: Path) -> tuple[str, Dict[str, str]]:
    metadata: Dict[str, str] = {}
    content_lines = []
    in_content = False

    for line in read_text(path).splitlines():
        if in_content:
            content_lines.append(line)
            continue

        if line.strip() == "Content:":
            in_content = True
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().lower().replace(" ", "_")] = normalize_text(value)

    content = normalize_text("\n".join(content_lines))
    return content, metadata


def iter_vims_samples(
    raw_root: str | Path,
    split: str = "test",
    limit: Optional[int] = None,
) -> Iterator[Sample]:
    raw_root = Path(raw_root)
    root = _vims_root(raw_root)
    cluster_dirs = _iter_cluster_dirs(raw_root)

    for index, original_cluster_dir in enumerate(cluster_dirs):
        if limit is not None and index >= limit:
            break

        cluster_name = original_cluster_dir.name
        document_files = sorted(
            [
                path
                for path in (original_cluster_dir / "original").glob("*.txt")
                if not path.name.startswith(".")
            ],
            key=natural_key,
        )
        parsed_documents = [_parse_original_document(path) for path in document_files]
        documents = normalize_documents(content for content, _ in parsed_documents)

        summary_dir = root / "summary" / cluster_name
        summary_files = sorted(summary_dir.glob("*.gold.txt"), key=natural_key)
        if not summary_files:
            raise FileNotFoundError(f"ViMs gold summary file not found in: {summary_dir}")

        reference_summaries = [normalize_text(read_text(path)) for path in summary_files]
        reference_summary = reference_summaries[0]

        yield {
            "sample_id": f"vims_{split}_{_cluster_id(original_cluster_dir):03d}",
            "dataset_name": ViMsDataset.DATASET_NAME,
            "split": split,
            "language": ViMsDataset.LANGUAGE,
            "documents": documents,
            "reference_summary": reference_summary,
            "metadata": {
                "source_dataset": "ViMs",
                "raw_index": index,
                "cluster_dir": str(original_cluster_dir),
                "cluster_id": _cluster_id(original_cluster_dir),
                "document_metadata": [metadata for _, metadata in parsed_documents],
                "reference_files": [str(path) for path in summary_files],
                "reference_summaries": reference_summaries,
                "num_references": len(reference_summaries),
                "num_documents": len(documents),
            },
        }


def prepare_vims(
    raw_root: str | Path,
    processed_root: str | Path,
    split: str = "test",
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    raw_count = len(_iter_cluster_dirs(raw_root))
    limit = expected_count if split == "test" and expected_count else None
    samples = list(iter_vims_samples(raw_root=raw_root, split=split, limit=limit))
    output_path = processed_root / f"{split}.jsonl"
    sample_count = write_jsonl(output_path, samples)

    metadata = {
        "dataset_name": ViMsDataset.DATASET_NAME,
        "split": split,
        "language": ViMsDataset.LANGUAGE,
        "sample_count": sample_count,
        "raw_cluster_count": raw_count,
        "expected_count": expected_count,
        "processed_file": str(output_path),
        "schema": sorted(BaseDataset.REQUIRED_FIELDS),
        "selection_policy": "first_n_natural_clusters_for_configured_test_count",
        "count_matches_expected": expected_count is None or sample_count == expected_count,
    }
    write_json(processed_root / "metadata.json", metadata)
    return metadata
