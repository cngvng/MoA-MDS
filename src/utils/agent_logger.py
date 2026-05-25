from __future__ import annotations

import json
import os
import tempfile
import traceback as traceback_lib
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Union


JsonDict = Dict[str, Any]
CostCalculator = Callable[[str, Dict[str, Optional[int]]], Dict[str, Any]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: JsonDict) -> None:
    """
    Write JSON atomically to avoid corrupted logs if the process is interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp_file.name)

    os.replace(tmp_path, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        tmp_file.write(text or "")
        tmp_path = Path(tmp_file.name)

    os.replace(tmp_path, path)


def to_plain_dict(obj: Any) -> JsonDict:
    """
    Convert dataclass/object/dict into JSON-serializable dict as safely as possible.
    """
    if obj is None:
        return {}

    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, dict):
        return deepcopy(obj)

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    # Fallback for simple result objects.
    if hasattr(obj, "__dict__"):
        return deepcopy(vars(obj))

    return {"value": str(obj)}


def sanitize_for_logging(obj: Any) -> Any:
    """
    Remove secrets from config snapshots and raw metadata before writing logs.
    """
    allowed_token_fields = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "total_input_tokens",
        "total_output_tokens",
        "per_agent_token_usage",
        "token_usage",
        "cached_input_tokens",
    }
    secret_markers = [
        "api_key",
        "apikey",
        "secret",
        "access_token",
        "refresh_token",
        "bearer_token",
        "id_token",
        "password",
        "authorization",
        "auth",
    ]

    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            key_lower = str(key).lower()

            if (
                key_lower not in allowed_token_fields
                and any(marker in key_lower for marker in secret_markers)
            ):
                cleaned[key] = "<REDACTED>"
            else:
                cleaned[key] = sanitize_for_logging(value)

        return cleaned

    if isinstance(obj, list):
        return [sanitize_for_logging(item) for item in obj]

    return obj


def normalize_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """
    Normalize different provider token usage schemas into one schema.
    """
    if not usage:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("input")
    )

    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("output")
    )

    total_tokens = usage.get("total_tokens") or usage.get("total")

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)

    return {
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
    }


def sum_usages(usages: List[Dict[str, Optional[int]]]) -> Dict[str, Optional[int]]:
    """
    Sum token usages. Missing values are ignored.
    If all values for a field are missing, the field remains None.
    """
    result: Dict[str, Optional[int]] = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }

    for field in result.keys():
        values = [u.get(field) for u in usages if u.get(field) is not None]
        result[field] = sum(values) if values else None

    return result


class AgentLogger:
    """
    Sample-level structured logger for MoA case studies.

    One AgentLogger instance should be created for one dataset sample.

    Output layout:

    outputs/{dataset_name}/case_studies/{sample_id}/
    ├── agent_log.json
    ├── metadata.json
    ├── final_summary.txt
    └── reference_summary.txt
    """

    def __init__(
        self,
        dataset_name: str,
        sample_id: Union[str, int],
        output_root: Union[str, Path] = "outputs",
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        config_snapshot: Optional[JsonDict] = None,
        cost_calculator: Optional[CostCalculator] = None,
        flush_on_update: bool = True,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.sample_id = str(sample_id)
        self.output_root = Path(output_root)
        self.model_id = model_id
        self.base_url = base_url
        self.config_snapshot = sanitize_for_logging(config_snapshot or {})
        self.cost_calculator = cost_calculator
        self.flush_on_update = flush_on_update

        self.case_dir = (
            self.output_root
            / self.dataset_name
            / "case_studies"
            / self.sample_id
        )

        self.agent_log_path = self.case_dir / "agent_log.json"
        self.metadata_path = self.case_dir / "metadata.json"
        self.final_summary_path = self.case_dir / "final_summary.txt"
        self.reference_summary_path = self.case_dir / "reference_summary.txt"

        self._lock = RLock()
        self.started_at = utc_now_iso()
        self.finished_at: Optional[str] = None

        self.agent_log: JsonDict = {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "started_at": self.started_at,
            "finished_at": None,
            "status": "running",
            "agents": {},
            "final_summary": None,
            "reference_summary": None,
            "sample_metrics": {},
            "errors": [],
        }

        self.metadata: JsonDict = {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "started_at": self.started_at,
            "finished_at": None,
            "status": "running",
            "runtime_seconds": None,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "cost": None,
            "estimated_cost": None,
            "per_agent_token_usage": {},
            "per_agent_cost": {},
            "config_snapshot": self.config_snapshot,
        }

        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.flush()

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def log_agent_start(
        self,
        agent_name: str,
        extra_metadata: Optional[JsonDict] = None,
    ) -> None:
        """
        Mark an agent as started for this sample.
        """
        with self._lock:
            agent_name = self._normalize_agent_name(agent_name)

            self.agent_log["agents"][agent_name] = {
                "agent_name": agent_name,
                "status": "running",
                "started_at": utc_now_iso(),
                "finished_at": None,
                "output_text": None,
                "metadata": sanitize_for_logging(extra_metadata or {}),
                "llm_calls": [],
                "token_usage": {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                },
                "cost": None,
                "estimated_cost": None,
                "runtime_seconds": None,
                "error": None,
            }

            if self.flush_on_update:
                self.flush()

    def log_agent_result(
        self,
        agent_name: str,
        result: Any,
        extra_metadata: Optional[JsonDict] = None,
    ) -> None:
        """
        Log a real agent result.

        The result can be:
        - dataclass AgentResult
        - dict
        - object with .to_dict()
        - simple object with __dict__

        Expected fields if available:
        - output_text
        - metadata
        - llm_calls
        - token_usage
        - runtime_seconds
        """
        with self._lock:
            agent_name = self._normalize_agent_name(agent_name)
            result_dict = to_plain_dict(result)

            output_text = (
                result_dict.get("output_text")
                or result_dict.get("text")
                or result_dict.get("summary")
                or ""
            )

            metadata = result_dict.get("metadata") or {}
            if extra_metadata:
                metadata = {
                    **metadata,
                    **extra_metadata,
                }

            llm_calls = result_dict.get("llm_calls") or []
            token_usage = self._extract_agent_usage(result_dict)

            estimated_cost = self._calculate_cost(token_usage)

            previous = self.agent_log["agents"].get(agent_name, {})
            started_at = previous.get("started_at") or result_dict.get("started_at")

            self.agent_log["agents"][agent_name] = {
                "agent_name": agent_name,
                "status": "completed",
                "started_at": started_at,
                "finished_at": result_dict.get("finished_at") or utc_now_iso(),
                "output_text": output_text,
                "metadata": sanitize_for_logging(metadata),
                "llm_calls": sanitize_for_logging(llm_calls),
                "token_usage": token_usage,
                "cost": estimated_cost,
                "estimated_cost": estimated_cost,
                "runtime_seconds": result_dict.get("runtime_seconds"),
                "error": None,
            }

            self._refresh_metadata()

            if self.flush_on_update:
                self.flush()

    def log_agent_error(
        self,
        agent_name: str,
        error: Union[str, Exception],
        extra_metadata: Optional[JsonDict] = None,
        save_traceback: bool = True,
    ) -> None:
        """
        Log an agent failure without crashing the whole dataset run.
        """
        with self._lock:
            agent_name = self._normalize_agent_name(agent_name)

            error_message = str(error)
            error_payload: JsonDict = {
                "agent_name": agent_name,
                "message": error_message,
                "time": utc_now_iso(),
                "metadata": sanitize_for_logging(extra_metadata or {}),
            }

            if save_traceback:
                error_payload["traceback"] = traceback_lib.format_exc()

            previous = self.agent_log["agents"].get(agent_name, {})

            self.agent_log["agents"][agent_name] = {
                "agent_name": agent_name,
                "status": "failed",
                "started_at": previous.get("started_at"),
                "finished_at": utc_now_iso(),
                "output_text": previous.get("output_text"),
                "metadata": previous.get("metadata", {}),
                "llm_calls": previous.get("llm_calls", []),
                "token_usage": previous.get(
                    "token_usage",
                    {
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    },
                ),
                "cost": previous.get("cost") or previous.get("estimated_cost"),
                "estimated_cost": previous.get("estimated_cost"),
                "runtime_seconds": previous.get("runtime_seconds"),
                "error": error_payload,
            }

            self.agent_log["errors"].append(error_payload)
            self.agent_log["status"] = "failed_partial"
            self.metadata["status"] = "failed_partial"

            self._refresh_metadata()

            if self.flush_on_update:
                self.flush()

    # ------------------------------------------------------------------
    # Sample-level outputs
    # ------------------------------------------------------------------

    def save_final_summary(self, final_summary: str) -> None:
        with self._lock:
            self.agent_log["final_summary"] = final_summary or ""
            atomic_write_text(self.final_summary_path, final_summary or "")

            if self.flush_on_update:
                self.flush()

    def save_reference_summary(self, reference_summary: Optional[str]) -> None:
        with self._lock:
            self.agent_log["reference_summary"] = reference_summary or ""
            atomic_write_text(self.reference_summary_path, reference_summary or "")

            if self.flush_on_update:
                self.flush()

    def log_sample_metrics(self, metrics: Optional[JsonDict]) -> None:
        with self._lock:
            self.agent_log["sample_metrics"] = sanitize_for_logging(metrics or {})

            if self.flush_on_update:
                self.flush()

    def finalize_sample(
        self,
        final_summary: Optional[str] = None,
        reference_summary: Optional[str] = None,
        sample_metrics: Optional[JsonDict] = None,
        status: str = "completed",
    ) -> None:
        """
        Finalize the current sample and write final JSON/text files.
        """
        with self._lock:
            if final_summary is not None:
                self.agent_log["final_summary"] = final_summary
                atomic_write_text(self.final_summary_path, final_summary)

            if reference_summary is not None:
                self.agent_log["reference_summary"] = reference_summary
                atomic_write_text(self.reference_summary_path, reference_summary)

            if sample_metrics is not None:
                self.agent_log["sample_metrics"] = sanitize_for_logging(sample_metrics)

            self.finished_at = utc_now_iso()
            self.agent_log["finished_at"] = self.finished_at
            self.agent_log["status"] = status

            self.metadata["finished_at"] = self.finished_at
            self.metadata["status"] = status
            self.metadata["runtime_seconds"] = self._runtime_seconds()

            self._refresh_metadata()
            self.flush()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """
        Persist current in-memory logs to disk.
        """
        with self._lock:
            atomic_write_json(self.agent_log_path, sanitize_for_logging(self.agent_log))
            atomic_write_json(self.metadata_path, sanitize_for_logging(self.metadata))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_agent_name(agent_name: str) -> str:
        name = str(agent_name).strip().lower()

        aliases = {
            "extractor_agent": "extractor",
            "sentence_agent": "extractor",
            "sentence_selection_agent": "extractor",
            "kgsum_agent": "kgsum",
            "kg_sum": "kgsum",
            "abstractor_agent": "abstractor",
            "abstractive_agent": "abstractor",
            "mixture_agent": "amf",
            "fusion": "amf",
        }

        return aliases.get(name, name)

    def _extract_agent_usage(self, result_dict: JsonDict) -> Dict[str, Optional[int]]:
        direct_usage = normalize_usage(result_dict.get("token_usage"))

        llm_calls = result_dict.get("llm_calls") or []
        llm_call_usages = []

        for call in llm_calls:
            call_dict = to_plain_dict(call)
            usage = (
                call_dict.get("usage")
                or call_dict.get("token_usage")
                or call_dict.get("metadata", {}).get("usage")
            )
            llm_call_usages.append(normalize_usage(usage))

        if any(
            direct_usage.get(field) is not None
            for field in ["input_tokens", "output_tokens", "total_tokens"]
        ):
            return direct_usage

        if llm_call_usages:
            return sum_usages(llm_call_usages)

        return direct_usage

    def _calculate_cost(
        self,
        usage: Dict[str, Optional[int]],
    ) -> Optional[Dict[str, Any]]:
        if self.cost_calculator is None or not self.model_id:
            return None

        try:
            return self.cost_calculator(self.model_id, usage)
        except Exception as exc:
            return {
                "status": "failed",
                "error": str(exc),
            }

    def _refresh_metadata(self) -> None:
        agents = self.agent_log.get("agents", {})

        per_agent_usage: Dict[str, Dict[str, Optional[int]]] = {}
        per_agent_cost: Dict[str, Any] = {}

        usages: List[Dict[str, Optional[int]]] = []

        for agent_name, agent_payload in agents.items():
            usage = normalize_usage(agent_payload.get("token_usage"))
            per_agent_usage[agent_name] = usage
            usages.append(usage)

            if agent_payload.get("estimated_cost") is not None:
                per_agent_cost[agent_name] = agent_payload["estimated_cost"]
            elif agent_payload.get("cost") is not None:
                per_agent_cost[agent_name] = agent_payload["cost"]

        total_usage = sum_usages(usages)

        self.metadata["per_agent_token_usage"] = per_agent_usage
        self.metadata["per_agent_cost"] = per_agent_cost

        self.metadata["total_input_tokens"] = total_usage["input_tokens"] or 0
        self.metadata["total_output_tokens"] = total_usage["output_tokens"] or 0
        self.metadata["total_tokens"] = total_usage["total_tokens"] or 0

        if self.cost_calculator is not None and self.model_id:
            cost = self._calculate_cost(total_usage)
            self.metadata["cost"] = cost
            self.metadata["estimated_cost"] = cost

    def _runtime_seconds(self) -> Optional[float]:
        if self.finished_at is None:
            return None

        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.finished_at)
            return round((end - start).total_seconds(), 4)
        except Exception:
            return None
