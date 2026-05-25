#!/usr/bin/env bash
set -euo pipefail

uv run python - <<'PY'
import json
import shutil
from pathlib import Path

from src.utils.agent_logger import AgentLogger
from src.utils.cost import calculate_token_cost


output_root = Path("outputs")
dataset_name = "logger_smoke"
sample_id = "phase4_fake_sample"
case_dir = output_root / dataset_name / "case_studies" / sample_id

if case_dir.exists():
    shutil.rmtree(case_dir)


def cost_calculator(model_id, usage):
    return calculate_token_cost(model_id, usage).to_dict()


logger = AgentLogger(
    dataset_name=dataset_name,
    sample_id=sample_id,
    output_root=output_root,
    model_id="gpt-5-nano",
    base_url=None,
    config_snapshot={
        "llm": {
            "model_id": "gpt-5-nano",
            "api_key_env": "OPENAI_API_KEY",
            "api_key": "must-not-be-written",
        }
    },
    cost_calculator=cost_calculator,
)

fake_results = {
    "extractor": {
        "agent_name": "extractor",
        "output_text": "Extractor summary.",
        "metadata": {"output_key": "r_ext"},
        "llm_calls": [],
        "token_usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        "runtime_seconds": 0.11,
    },
    "kgsum": {
        "agent_name": "kgsum",
        "output_text": "KG summary.",
        "metadata": {"output_key": "r_kg"},
        "llm_calls": [
            {
                "model_id": "gpt-5-nano",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_tokens": 28,
                },
                "api_key": "must-not-be-written",
            }
        ],
        "runtime_seconds": 0.22,
    },
    "abstractor": {
        "agent_name": "abstractor",
        "output_text": "Abstractor summary.",
        "metadata": {"output_key": "r_abs"},
        "llm_calls": [],
        "token_usage": {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
        "runtime_seconds": 0.33,
    },
    "amf": {
        "agent_name": "amf",
        "output_text": "Final fused summary.",
        "metadata": {"output_key": "S_MoA"},
        "llm_calls": [],
        "token_usage": {"input_tokens": 40, "output_tokens": 16, "total_tokens": 56},
        "runtime_seconds": 0.44,
    },
}

for agent_name, result in fake_results.items():
    logger.log_agent_start(agent_name)
    logger.log_agent_result(agent_name, result)

logger.finalize_sample(
    final_summary=fake_results["amf"]["output_text"],
    reference_summary="Reference summary.",
    sample_metrics={"rouge1_f1": 50.0},
    status="completed",
)

required_paths = [
    logger.agent_log_path,
    logger.metadata_path,
    logger.final_summary_path,
    logger.reference_summary_path,
]

for path in required_paths:
    if not path.exists():
        raise AssertionError(f"Missing artifact: {path}")

agent_log = json.loads(logger.agent_log_path.read_text(encoding="utf-8"))
metadata = json.loads(logger.metadata_path.read_text(encoding="utf-8"))

expected_schema = {
    "sample_id",
    "dataset_name",
    "model_id",
    "base_url",
    "status",
    "runtime_seconds",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "estimated_cost",
    "per_agent_token_usage",
    "per_agent_cost",
}
missing = sorted(expected_schema - set(metadata))
if missing:
    raise AssertionError(f"metadata.json missing fields: {missing}")

if set(agent_log["agents"]) != {"extractor", "kgsum", "abstractor", "amf"}:
    raise AssertionError(f"Unexpected agents: {sorted(agent_log['agents'])}")

if metadata["status"] != "completed":
    raise AssertionError(f"Unexpected status: {metadata['status']}")

if metadata["model_id"] != "gpt-5-nano":
    raise AssertionError(f"Unexpected model_id: {metadata['model_id']}")

if metadata["total_input_tokens"] != 100:
    raise AssertionError(f"Unexpected total_input_tokens: {metadata['total_input_tokens']}")

if metadata["total_output_tokens"] != 40:
    raise AssertionError(f"Unexpected total_output_tokens: {metadata['total_output_tokens']}")

if metadata["total_tokens"] != 140:
    raise AssertionError(f"Unexpected total_tokens: {metadata['total_tokens']}")

if not isinstance(metadata["estimated_cost"], dict):
    raise AssertionError("estimated_cost must be a dict when pricing is configured")

serialized_agent_log = json.dumps(agent_log, ensure_ascii=False)
serialized_metadata = json.dumps(metadata, ensure_ascii=False)
if "must-not-be-written" in serialized_agent_log or "must-not-be-written" in serialized_metadata:
    raise AssertionError("Secret value was written to structured logs")

print("agent_logger_smoke_status=ok")
for path in required_paths:
    print(f"artifact={path.resolve()}")
print(f"metadata_status={metadata['status']}")
print(f"metadata_model_id={metadata['model_id']}")
print(f"metadata_base_url={metadata['base_url']}")
print(f"metadata_total_input_tokens={metadata['total_input_tokens']}")
print(f"metadata_total_output_tokens={metadata['total_output_tokens']}")
print(f"metadata_total_tokens={metadata['total_tokens']}")
print(f"metadata_estimated_cost={metadata['estimated_cost']['total_cost']}")
PY
