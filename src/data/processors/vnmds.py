from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .base import (
    BaseDataset,
    Sample,
    natural_key,
    normalize_documents,
    normalize_text,
    parse_info_file,
    read_text,
    write_json,
    write_jsonl,
)


class VNMDSDataset(BaseDataset):
    DATASET_NAME = "vn_mds"
    LANGUAGE = "vi"


def _cluster_id(path: Path) -> int:
    suffix = path.name.split("_")[-1]
    return int(suffix) if suffix.isdigit() else 0


def _iter_cluster_dirs(raw_root: Path) -> list[Path]:
    clusters_root = raw_root / "raw" / "extracted" / "clusters"
    if not clusters_root.exists():
        raise FileNotFoundError(f"VN-MDS clusters directory not found: {clusters_root}")
    return sorted(
        [path for path in clusters_root.iterdir() if path.is_dir() and path.name.startswith("cluster_")],
        key=natural_key,
    )


def iter_vnmds_samples(
    raw_root: str | Path,
    split: str = "test",
    limit: Optional[int] = None,
) -> Iterator[Sample]:
    raw_root = Path(raw_root)
    cluster_dirs = _iter_cluster_dirs(raw_root)

    for index, cluster_dir in enumerate(cluster_dirs):
        if limit is not None and index >= limit:
            break

        body_files = sorted(cluster_dir.glob("*.body.txt"), key=natural_key)
        documents = normalize_documents(read_text(path) for path in body_files)

        reference_path = cluster_dir / f"{cluster_dir.name}.ref1.txt"
        if not reference_path.exists():
            reference_path = cluster_dir / f"{cluster_dir.name}.sum.txt"

        cluster_info = parse_info_file(cluster_dir / f"{cluster_dir.name}.info.txt")
        document_metadata = [
            {
                "document_id": path.name.replace(".body.txt", ""),
                **parse_info_file(cluster_dir / f"{path.name.replace('.body.txt', '.info.txt')}"),
            }
            for path in body_files
        ]

        yield {
            "sample_id": f"vn_mds_{split}_{_cluster_id(cluster_dir):03d}",
            "dataset_name": VNMDSDataset.DATASET_NAME,
            "split": split,
            "language": VNMDSDataset.LANGUAGE,
            "documents": documents,
            "reference_summary": normalize_text(read_text(reference_path)),
            "metadata": {
                "source_dataset": "VN-MDS",
                "raw_index": index,
                "cluster_dir": str(cluster_dir),
                "cluster_id": _cluster_id(cluster_dir),
                "cluster_info": cluster_info,
                "document_metadata": document_metadata,
                "reference_file": str(reference_path),
                "num_documents": len(documents),
            },
        }


def prepare_vnmds(
    raw_root: str | Path,
    processed_root: str | Path,
    split: str = "test",
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    raw_count = len(_iter_cluster_dirs(raw_root))
    limit = expected_count if split == "test" and expected_count else None
    samples = list(iter_vnmds_samples(raw_root=raw_root, split=split, limit=limit))
    output_path = processed_root / f"{split}.jsonl"
    sample_count = write_jsonl(output_path, samples)

    metadata = {
        "dataset_name": VNMDSDataset.DATASET_NAME,
        "split": split,
        "language": VNMDSDataset.LANGUAGE,
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
