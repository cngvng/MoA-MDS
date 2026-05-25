from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DATASETS = ["multi_news", "multi_xscience", "vn_mds", "vims"]
DATASET_DISPLAY_NAMES = {
    "multi_news": "Multi-News",
    "multi_xscience": "Multi-XScience",
    "vn_mds": "VN-MDS",
    "vims": "ViMs",
}
DEFAULT_INPUT_ROOTS = [Path("/data/moa/outputs"), Path("outputs")]
DEFAULT_OUTPUT_DIR = Path("outputs/paper_results/main_results")


def main() -> None:
    args = parse_args()
    input_roots = [Path(root) for root in args.input_root] if args.input_root else DEFAULT_INPUT_ROOTS
    output_dir = Path(args.output_dir)
    package = package_main_results(input_roots=input_roots, output_dir=output_dir)

    print("package_main_results_status=ok")
    for name, path in package["artifact_paths"].items():
        print(f"{name}={path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package saved MoA outputs into paper-ready main-result tables."
    )
    parser.add_argument(
        "--input-root",
        action="append",
        default=[],
        help="Saved output root to scan. May be repeated. Defaults to /data/moa/outputs and outputs.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def package_main_results(input_roots: Iterable[Path], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_payloads = {
        dataset: find_dataset_payload(dataset, input_roots)
        for dataset in DATASETS
    }

    main_rows = [build_main_row(dataset, payload) for dataset, payload in dataset_payloads.items()]
    cost_rows = [build_cost_row(dataset, payload) for dataset, payload in dataset_payloads.items()]

    main_csv = output_dir / "main_results.csv"
    main_md = output_dir / "main_results.md"
    cost_csv = output_dir / "cost_summary.csv"
    cost_md = output_dir / "cost_summary.md"
    dataset_metadata_json = output_dir / "dataset_metadata.json"

    write_csv(main_csv, main_rows)
    write_markdown_table(
        main_md,
        "Main Results",
        main_rows,
        note="Values are aggregated from saved dataset-level outputs. No LLM calls are made by this packager.",
    )
    write_csv(cost_csv, cost_rows)
    write_markdown_table(
        cost_md,
        "Cost Summary",
        cost_rows,
        note="Token and cost values are copied from saved metadata artifacts.",
    )
    write_json(
        dataset_metadata_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_roots": [str(root) for root in input_roots],
            "output_dir": str(output_dir),
            "paper": {
                "title": "A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs",
                "journal": "Neural Computing and Applications",
                "year": 2025,
            },
            "datasets": {
                dataset: {
                    "display_name": DATASET_DISPLAY_NAMES[dataset],
                    "source_dir": str(payload["source_dir"]),
                    "metrics_path": str(payload["metrics_path"]),
                    "metadata_path": str(payload["metadata_path"]),
                    "case_studies": payload["case_studies"],
                    "metrics": payload["metrics"],
                    "metadata": payload["metadata"],
                }
                for dataset, payload in dataset_payloads.items()
            },
        },
    )

    return {
        "main_rows": main_rows,
        "cost_rows": cost_rows,
        "dataset_payloads": dataset_payloads,
        "artifact_paths": {
            "main_results_csv": str(main_csv.resolve()),
            "main_results_md": str(main_md.resolve()),
            "cost_summary_csv": str(cost_csv.resolve()),
            "cost_summary_md": str(cost_md.resolve()),
            "dataset_metadata_json": str(dataset_metadata_json.resolve()),
        },
    }


def find_dataset_payload(dataset: str, input_roots: Iterable[Path]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for root in input_roots:
        dataset_dir = root / dataset
        metrics_path = dataset_dir / "metrics.json"
        metadata_path = dataset_dir / "metadata.json"
        if metrics_path.exists() and metadata_path.exists():
            metrics = read_json(metrics_path)
            metadata = read_json(metadata_path)
            candidates.append(
                {
                    "source_dir": dataset_dir,
                    "metrics_path": metrics_path,
                    "metadata_path": metadata_path,
                    "metrics": metrics,
                    "metadata": metadata,
                    "case_studies": list_case_studies(dataset_dir),
                }
            )

    if not candidates:
        raise FileNotFoundError(
            f"No saved dataset-level metrics/metadata found for {dataset} under: "
            + ", ".join(str(root) for root in input_roots)
        )

    return sorted(
        candidates,
        key=lambda item: (
            int(item["metadata"].get("completed_samples") or 0),
            int(item["metrics"].get("sample_count") or 0),
        ),
        reverse=True,
    )[0]


def build_main_row(dataset: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload["metrics"]
    metadata = payload["metadata"]
    rouge = metrics.get("rouge", {})
    kg = metrics.get("kg", {})
    return {
        "dataset": dataset,
        "dataset_name": DATASET_DISPLAY_NAMES[dataset],
        "split": metadata.get("split"),
        "model_id": metadata.get("model_id"),
        "sample_count": metrics.get("sample_count"),
        "completed_samples": metadata.get("completed_samples"),
        "failed_samples": metadata.get("failed_samples"),
        "rouge1_f1": format_float(rouge.get("rouge1_f1")),
        "rouge2_f1": format_float(rouge.get("rouge2_f1")),
        "rougeL_f1": format_float(rouge.get("rougeL_f1")),
        "avg_num_entities": format_float(kg.get("avg_num_entities")),
        "avg_num_relations": format_float(kg.get("avg_num_relations")),
        "avg_num_communities": format_float(kg.get("avg_num_communities")),
        "avg_modularity": format_float(kg.get("avg_modularity")),
        "avg_contrastive_relation_count": format_float(kg.get("avg_contrastive_relation_count")),
        "source_dir": str(payload["source_dir"]),
    }


def build_cost_row(dataset: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload["metadata"]
    return {
        "dataset": dataset,
        "dataset_name": DATASET_DISPLAY_NAMES[dataset],
        "model_id": metadata.get("model_id"),
        "completed_samples": metadata.get("completed_samples"),
        "total_input_tokens": format_float(metadata.get("total_input_tokens")),
        "total_output_tokens": format_float(metadata.get("total_output_tokens")),
        "total_tokens": format_float(metadata.get("total_tokens")),
        "avg_total_tokens_per_sample": format_float(metadata.get("avg_total_tokens_per_sample")),
        "total_cost_usd": format_float(metadata.get("total_cost_usd")),
        "avg_cost_per_sample_usd": format_float(metadata.get("avg_cost_per_sample_usd")),
        "avg_runtime_seconds_per_sample": format_float(metadata.get("avg_runtime_seconds_per_sample")),
    }


def list_case_studies(dataset_dir: Path) -> List[Dict[str, Any]]:
    case_root = dataset_dir / "case_studies"
    if not case_root.exists():
        return []
    cases = []
    for case_dir in sorted(path for path in case_root.iterdir() if path.is_dir()):
        artifacts = {
            "agent_log": case_dir / "agent_log.json",
            "metadata": case_dir / "metadata.json",
            "final_summary": case_dir / "final_summary.txt",
            "reference_summary": case_dir / "reference_summary.txt",
            "knowledge_graph": case_dir / "knowledge_graph.json",
        }
        cases.append(
            {
                "sample_id": case_dir.name,
                "case_dir": str(case_dir),
                "artifacts": {
                    name: str(path) for name, path in artifacts.items() if path.exists()
                },
            }
        )
    return cases


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, title: str, rows: List[Dict[str, Any]], note: str) -> None:
    headers = list(rows[0].keys()) if rows else []
    lines = [f"# {title}", "", note, ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_float(value: Any) -> Optional[str]:
    if value is None:
        return None
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    main()
