from __future__ import annotations

import argparse
import json
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.agents import MixtureOfAgents
from src.data.main import load_dataset
from src.utils.config import Config, load_config


CASE_SUFFIX = "_moa"


def run_one_sample(dataset: str, split: str, num_samples: int = 1) -> Dict[str, Any]:
    result = run_dataset(
        dataset=dataset,
        split=split,
        num_samples=num_samples,
        resume=False,
        skip_existing=False,
    )
    if not result["samples"]:
        raise RuntimeError("No sample was processed.")
    return result["samples"][0]


def run_dataset(
    *,
    dataset: str,
    split: str,
    num_samples: Optional[int] = None,
    sample_ids: Optional[Iterable[str]] = None,
    resume: bool = True,
    skip_existing: bool = True,
    output_root: Optional[str | Path] = None,
    overlay_configs: Optional[Iterable[str]] = None,
    continue_on_error: bool = True,
) -> Dict[str, Any]:
    config = load_config(overlay_configs=_resolve_overlay_configs(dataset, overlay_configs))
    if output_root is not None:
        config = config.merge({"outputs": {"root_dir": str(output_root)}})

    samples = load_dataset(
        dataset,
        split=split,
        num_samples=num_samples,
        sample_ids=sample_ids,
        config=config,
    )
    moa = MixtureOfAgents(config)
    dataset_dir = Path(config.get("outputs.root_dir", "outputs")) / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    sample_results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    attempted = 0
    completed = 0
    skipped = 0

    for sample in samples:
        attempted += 1
        sample_id = str(sample["sample_id"])
        case_dir = dataset_dir / "case_studies" / f"{sample_id}{CASE_SUFFIX}"

        if resume and skip_existing and _is_completed_case(case_dir):
            skipped += 1
            completed += 1
            sample_results.append(_load_completed_case(sample, case_dir))
            print(f"sample_status=skipped sample_id={json.dumps(sample_id, ensure_ascii=False)}")
            continue

        try:
            result = moa.process_sample(sample, sample_id_suffix=CASE_SUFFIX)
            sample_results.append(_sample_summary(result))
            completed += 1
            print(f"sample_status=completed sample_id={json.dumps(sample_id, ensure_ascii=False)}")
        except Exception as exc:
            failure = {
                "sample_id": sample_id,
                "dataset_name": dataset,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"sample_status=failed sample_id={json.dumps(sample_id, ensure_ascii=False)} error={json.dumps(str(exc))}")
            if not continue_on_error:
                raise

    metrics = _aggregate_metrics(sample_results)
    metadata = _dataset_metadata(
        config=config,
        dataset=dataset,
        split=split,
        attempted=attempted,
        completed=completed,
        skipped=skipped,
        failed=len(failures),
        sample_results=sample_results,
        failures=failures,
    )

    metrics_path = dataset_dir / "metrics.json"
    metadata_path = dataset_dir / "metadata.json"
    _write_json(metrics_path, metrics)
    _write_json(metadata_path, metadata)

    return {
        "dataset_name": dataset,
        "split": split,
        "attempted": attempted,
        "completed": completed,
        "skipped": skipped,
        "failed": len(failures),
        "samples": sample_results,
        "failures": failures,
        "metrics": metrics,
        "metadata": metadata,
        "artifact_paths": {
            "dataset_metrics": str(metrics_path.resolve()),
            "dataset_metadata": str(metadata_path.resolve()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MoA main-results pipeline.")
    parser.add_argument("--dataset", default="multi_news")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--overlay-config", action="append", default=[])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_dataset(
        dataset=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        sample_ids=args.sample_ids,
        resume=args.resume,
        skip_existing=args.skip_existing,
        output_root=args.output_root,
        overlay_configs=args.overlay_config,
        continue_on_error=args.continue_on_error,
    )
    summary = {
        "moa_pipeline_status": "ok" if result["failed"] == 0 else "partial",
        "dataset_name": result["dataset_name"],
        "split": result["split"],
        "attempted": result["attempted"],
        "completed": result["completed"],
        "skipped": result["skipped"],
        "failed": result["failed"],
        "metrics": result["metrics"],
        "artifact_paths": result["artifact_paths"],
        "sample_artifacts": [
            sample.get("artifact_paths", {}) for sample in result["samples"]
        ],
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"moa_pipeline_status={summary['moa_pipeline_status']}")
    for key, value in summary.items():
        if key == "moa_pipeline_status":
            continue
        print(f"{key}={json.dumps(value, ensure_ascii=False)}")


def _is_completed_case(case_dir: Path) -> bool:
    metadata_path = case_dir / "metadata.json"
    agent_log_path = case_dir / "agent_log.json"
    final_summary_path = case_dir / "final_summary.txt"
    if not (metadata_path.exists() and agent_log_path.exists() and final_summary_path.exists()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return metadata.get("status") == "completed"


def _load_completed_case(sample: Mapping[str, Any], case_dir: Path) -> Dict[str, Any]:
    agent_log = json.loads((case_dir / "agent_log.json").read_text(encoding="utf-8"))
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    final_summary = (case_dir / "final_summary.txt").read_text(encoding="utf-8")
    return {
        "sample_id": sample.get("sample_id"),
        "dataset_name": sample.get("dataset_name"),
        "final_summary_length": len(final_summary),
        "agents": sorted(agent_log.get("agents", {}).keys()),
        "metrics": agent_log.get("sample_metrics", {}),
        "kg_metrics": _extract_kg_metrics(agent_log),
        "metadata": metadata,
        "artifact_paths": _case_artifact_paths(case_dir),
        "skipped": True,
    }


def _sample_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    metadata_path = result.get("artifact_paths", {}).get("metadata")
    metadata = {}
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return {
        "sample_id": result["sample_id"],
        "dataset_name": result["dataset_name"],
        "final_summary_length": len(result["final_summary"]),
        "agents": sorted(result["agents"]),
        "metrics": result["metrics"],
        "kg_metrics": _extract_kg_metrics_from_result(result),
        "metadata": metadata,
        "artifact_paths": result["artifact_paths"],
        "skipped": False,
    }


def _case_artifact_paths(case_dir: Path) -> Dict[str, str]:
    return {
        "agent_log": str((case_dir / "agent_log.json").resolve()),
        "metadata": str((case_dir / "metadata.json").resolve()),
        "final_summary": str((case_dir / "final_summary.txt").resolve()),
        "reference_summary": str((case_dir / "reference_summary.txt").resolve()),
        "knowledge_graph": str((case_dir / "knowledge_graph.json").resolve()),
    }


def _extract_kg_metrics(agent_log: Mapping[str, Any]) -> Dict[str, Any]:
    kgsum = agent_log.get("agents", {}).get("kgsum", {})
    metadata = kgsum.get("metadata", {}).get("M_kg", {})
    return dict(metadata.get("graph_metrics", {}))


def _extract_kg_metrics_from_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    kgsum = result.get("agents", {}).get("kgsum", {})
    metadata = kgsum.get("metadata", {}).get("M_kg", {})
    return dict(metadata.get("graph_metrics", {}))


def _aggregate_metrics(sample_results: List[Mapping[str, Any]]) -> Dict[str, Any]:
    rouge = {}
    for metric_name in ["rouge1", "rouge2", "rougeL"]:
        for score_name in ["precision", "recall", "f1"]:
            values = [
                sample.get("metrics", {}).get(metric_name, {}).get(score_name)
                for sample in sample_results
            ]
            values = [float(value) for value in values if value is not None]
            rouge[f"{metric_name}_{score_name}"] = _mean(values)

    kg_metrics = {}
    for metric_name in [
        "num_entities",
        "num_relations",
        "num_communities",
        "modularity",
        "contrastive_relation_count",
    ]:
        values = [sample.get("kg_metrics", {}).get(metric_name) for sample in sample_results]
        values = [float(value) for value in values if value is not None]
        kg_metrics[f"avg_{metric_name}"] = _mean(values)

    return {
        "sample_count": len(sample_results),
        "rouge": rouge,
        "kg": kg_metrics,
    }


def _dataset_metadata(
    *,
    config: Config,
    dataset: str,
    split: str,
    attempted: int,
    completed: int,
    skipped: int,
    failed: int,
    sample_results: List[Mapping[str, Any]],
    failures: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    sample_metadata = [sample.get("metadata", {}) for sample in sample_results]
    total_tokens = [_int_or_none(metadata.get("total_tokens")) for metadata in sample_metadata]
    input_tokens = [_int_or_none(metadata.get("total_input_tokens")) for metadata in sample_metadata]
    output_tokens = [_int_or_none(metadata.get("total_output_tokens")) for metadata in sample_metadata]
    runtimes = [_float_or_none(metadata.get("runtime_seconds")) for metadata in sample_metadata]
    costs = [
        _float_or_none((metadata.get("cost") or metadata.get("estimated_cost") or {}).get("total_cost"))
        for metadata in sample_metadata
    ]

    return {
        "dataset_name": dataset,
        "split": split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if failed == 0 else "partial",
        "attempted_samples": attempted,
        "completed_samples": completed,
        "skipped_samples": skipped,
        "failed_samples": failed,
        "model_id": config.get("llm.model_id"),
        "base_url": config.get("llm.base_url"),
        "total_input_tokens": _sum(input_tokens),
        "total_output_tokens": _sum(output_tokens),
        "total_tokens": _sum(total_tokens),
        "avg_input_tokens_per_sample": _mean_present(input_tokens),
        "avg_output_tokens_per_sample": _mean_present(output_tokens),
        "avg_total_tokens_per_sample": _mean_present(total_tokens),
        "total_cost_usd": _sum(costs),
        "avg_cost_per_sample_usd": _mean_present(costs),
        "avg_runtime_seconds_per_sample": _mean_present(runtimes),
        "failures": list(failures),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_overlay_configs(dataset: str, overlay_configs: Optional[Iterable[str]]) -> List[str]:
    defaults = [
        "configs/agents/default.yaml",
        f"configs/evaluations/{dataset}.yaml",
    ]
    paths = [path for path in defaults if Path(path).exists()]
    paths.extend(list(overlay_configs or []))
    return paths


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.fmean(values), 8)


def _mean_present(values: List[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return _mean(present)


def _sum(values: List[Optional[float]]) -> float:
    return round(sum(float(value) for value in values if value is not None), 8)


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    main()
