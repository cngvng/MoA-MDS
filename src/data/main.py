from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml

from src.data.processors import (
    BaseDataset,
    MultiNewsDataset,
    MultiXScienceDataset,
    VNMDSDataset,
    ViMsDataset,
    prepare_multinews,
    prepare_multixscience,
    prepare_vims,
    prepare_vnmds,
)


PrepareFn = Callable[[str | Path, str | Path, str, Optional[int]], Dict[str, Any]]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    raw_config_key: str
    processed_config_key: str
    storage_dataset_key: str
    dataset_class: type[BaseDataset]
    prepare_fn: PrepareFn


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "multi_news": DatasetSpec(
        dataset_name="multi_news",
        raw_config_key="storage.global.multinews",
        processed_config_key="storage.workspace.multinews_processed",
        storage_dataset_key="multi_news",
        dataset_class=MultiNewsDataset,
        prepare_fn=prepare_multinews,
    ),
    "multi_xscience": DatasetSpec(
        dataset_name="multi_xscience",
        raw_config_key="storage.global.multixscience",
        processed_config_key="storage.workspace.multixscience_processed",
        storage_dataset_key="multi_xscience",
        dataset_class=MultiXScienceDataset,
        prepare_fn=prepare_multixscience,
    ),
    "vn_mds": DatasetSpec(
        dataset_name="vn_mds",
        raw_config_key="storage.global.vnmds",
        processed_config_key="storage.workspace.vnmds_processed",
        storage_dataset_key="vn_mds",
        dataset_class=VNMDSDataset,
        prepare_fn=prepare_vnmds,
    ),
    "vims": DatasetSpec(
        dataset_name="vims",
        raw_config_key="storage.global.vims",
        processed_config_key="storage.workspace.vims_processed",
        storage_dataset_key="vims",
        dataset_class=ViMsDataset,
        prepare_fn=prepare_vims,
    ),
}


class DataConfig:
    """Small YAML-backed config reader for Phase 2 data paths only."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload

    @classmethod
    def load(cls, storage_config_path: str | Path = "configs/storage.yaml") -> "DataConfig":
        path = Path(storage_config_path)
        with path.open("r", encoding="utf-8") as file:
            return cls(yaml.safe_load(file) or {})

    def get(self, path: str, default: Any = None) -> Any:
        current: Any = self.payload
        for key in self._strip_storage_prefix(path).split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def require(self, path: str) -> Any:
        value = self.get(path)
        if value is None:
            raise KeyError(f"Config key not found: {path}")
        return value

    @staticmethod
    def _strip_storage_prefix(path: str) -> str:
        return path.removeprefix("storage.")


class UnifiedMDSDataset:
    """Release-facing dataset wrapper for normalized MoA samples."""

    def __init__(
        self,
        samples: Iterable[Dict[str, Any]],
        dataset_name: str,
        split: str,
        processed_path: str | Path,
    ) -> None:
        self.samples = list(samples)
        self.dataset_name = dataset_name
        self.split = split
        self.processed_path = Path(processed_path)

        for index, sample in enumerate(self.samples):
            BaseDataset.validate_sample(sample, index=index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)


def _expected_count(config: Any, spec: DatasetSpec, split: str) -> Optional[int]:
    if split != "test":
        return None
    return config.get(f"storage.datasets.{spec.storage_dataset_key}.test_samples")


def _processed_file(config: Any, spec: DatasetSpec, split: str) -> Path:
    return Path(config.require(spec.processed_config_key)) / f"{split}.jsonl"


def prepare_dataset(
    dataset_name: str,
    split: str = "test",
    force: bool = False,
    config: Any = None,
) -> Dict[str, Any]:
    config = config or DataConfig.load()
    spec = get_dataset_spec(dataset_name)
    processed_file = _processed_file(config, spec, split)

    if processed_file.exists() and not force:
        metadata_path = processed_file.parent / "metadata.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "dataset_name": spec.dataset_name,
            "split": split,
            "processed_file": str(processed_file),
            "sample_count": sum(1 for _ in processed_file.open("r", encoding="utf-8")),
            "prepared": False,
        }

    return spec.prepare_fn(
        Path(config.require(spec.raw_config_key)),
        processed_file.parent,
        split,
        _expected_count(config, spec, split),
    )


def load_dataset(
    dataset_name: str,
    split: str = "test",
    num_samples: Optional[int] = None,
    sample_ids: Optional[Iterable[str]] = None,
    prepare: bool = True,
    force_prepare: bool = False,
    config: Any = None,
) -> UnifiedMDSDataset:
    config = config or DataConfig.load()
    spec = get_dataset_spec(dataset_name)

    if prepare:
        prepare_dataset(dataset_name, split=split, force=force_prepare, config=config)

    processed_file = _processed_file(config, spec, split)
    if not processed_file.exists():
        raise FileNotFoundError(
            f"Processed split file not found: {processed_file}. "
            "Run with prepare=True or use the CLI without --no-prepare."
        )

    try:
        dataset = spec.dataset_class(processed_file, split=split)
    except Exception:
        if not prepare:
            raise
        prepare_dataset(dataset_name, split=split, force=True, config=config)
        dataset = spec.dataset_class(processed_file, split=split)
    samples = list(dataset)

    selected_ids = set(sample_ids or [])
    if selected_ids:
        samples = [sample for sample in samples if sample["sample_id"] in selected_ids]

    if num_samples is not None:
        samples = samples[:num_samples]

    return UnifiedMDSDataset(
        samples=samples,
        dataset_name=spec.dataset_name,
        split=split,
        processed_path=processed_file,
    )


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset_name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}") from exc


def summarize_dataset(dataset: UnifiedMDSDataset) -> Dict[str, Any]:
    first_sample = dataset[0] if len(dataset) else None
    return {
        "dataset_name": dataset.dataset_name,
        "split": dataset.split,
        "sample_count": len(dataset),
        "processed_file": str(dataset.processed_path),
        "schema": sorted(BaseDataset.REQUIRED_FIELDS),
        "first_sample": {
            "sample_id": first_sample["sample_id"] if first_sample else None,
            "language": first_sample["language"] if first_sample else None,
            "document_count": len(first_sample["documents"]) if first_sample else 0,
            "document_types": sorted(
                {type(document).__name__ for document in first_sample["documents"]}
            )
            if first_sample
            else [],
            "reference_summary_type": type(first_sample["reference_summary"]).__name__
            if first_sample
            else None,
            "metadata_keys": sorted(first_sample["metadata"].keys()) if first_sample else [],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and load unified MoA MDS datasets.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print only JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        sample_ids=args.sample_ids,
        prepare=not args.no_prepare,
        force_prepare=args.force_prepare,
    )
    summary = summarize_dataset(dataset)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"dataset_name: {summary['dataset_name']}")
    print(f"split: {summary['split']}")
    print(f"sample_count: {summary['sample_count']}")
    print(f"processed_file: {summary['processed_file']}")
    print(f"schema: {', '.join(summary['schema'])}")
    print("first_sample:")
    print(json.dumps(summary["first_sample"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
