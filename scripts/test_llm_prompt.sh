#!/usr/bin/env bash
set -euo pipefail

uv run python - <<'PY'
from src.llms.factory import build_llm, load_llm_factory_config
from src.utils.cost import calculate_token_cost

prompt = (
    "Reply with exactly one short sentence confirming that the MoA LLM "
    "smoke test is working."
)

factory_config = load_llm_factory_config()
llm = build_llm()
result = llm.generate_answer(
    system_prompt="You are a concise test assistant.",
    user_prompt=prompt,
)

usage = result.usage.to_dict()
usage_returned = any(value is not None for value in usage.values())
cost = calculate_token_cost(result.model_id, usage).to_dict()

print("llm_smoke_status=ok")
print(f"model_id={result.model_id}")
print(f"base_url={result.raw_metadata.get('base_url') if result.raw_metadata else None}")
print(f"api_key_env={factory_config.api_key_env}")
print(f"api_key_loaded={factory_config.safe_dict()['api_key_loaded']}")
print(f"finish_reason={result.raw_metadata.get('finish_reason') if result.raw_metadata else None}")
print(f"usage_returned={str(usage_returned).lower()}")
print(f"input_tokens={usage['input_tokens']}")
print(f"output_tokens={usage['output_tokens']}")
print(f"total_tokens={usage['total_tokens']}")
print(f"estimated_cost_usd={cost['total_cost']}")
print(f"response_text={result.text.strip()}")
PY
