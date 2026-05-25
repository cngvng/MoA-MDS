from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


@dataclass(frozen=True)
class TokenCost:
    model_id: str
    currency: str
    input_cost: Optional[float]
    output_cost: Optional[float]
    total_cost: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    pricing_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_usage(usage: Any) -> Dict[str, Optional[int]]:
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    if hasattr(usage, "to_dict"):
        usage = usage.to_dict()
    elif hasattr(usage, "__dict__") and not isinstance(usage, Mapping):
        usage = vars(usage)

    if not isinstance(usage, Mapping):
        raise TypeError(f"Unsupported usage type: {type(usage)!r}")

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)

    return {
        "input_tokens": _optional_int(input_tokens),
        "output_tokens": _optional_int(output_tokens),
        "total_tokens": _optional_int(total_tokens),
    }


def load_pricing(path: str | Path = "configs/pricing.yaml") -> Dict[str, Any]:
    pricing_path = Path(path)
    if not pricing_path.exists():
        raise FileNotFoundError(f"Pricing config not found: {pricing_path}")

    with pricing_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def calculate_token_cost(
    model_id: str,
    usage: Any,
    pricing: Optional[Mapping[str, Any]] = None,
    pricing_path: str | Path = "configs/pricing.yaml",
) -> TokenCost:
    pricing_config = dict(pricing or load_pricing(pricing_path))
    usage_values = normalize_usage(usage)
    model_pricing = _find_model_pricing(model_id, pricing_config)
    currency = str(pricing_config.get("currency", "USD"))

    if model_pricing is None:
        return TokenCost(
            model_id=model_id,
            currency=currency,
            input_cost=None,
            output_cost=None,
            total_cost=None,
            pricing_source=None,
            **usage_values,
        )

    input_tokens = usage_values["input_tokens"]
    output_tokens = usage_values["output_tokens"]
    input_cost = _cost_for_tokens(input_tokens, model_pricing.get("input_per_1m"))
    output_cost = _cost_for_tokens(output_tokens, model_pricing.get("output_per_1m"))

    total_cost = None
    if input_cost is not None or output_cost is not None:
        total_cost = (input_cost or 0.0) + (output_cost or 0.0)

    return TokenCost(
        model_id=model_id,
        currency=currency,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        pricing_source=str(model_pricing.get("source", "")) or None,
        **usage_values,
    )


def _find_model_pricing(
    model_id: str,
    pricing_config: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    models = pricing_config.get("models", {})
    if not isinstance(models, Mapping):
        return None

    if model_id in models and isinstance(models[model_id], Mapping):
        return models[model_id]

    aliases = pricing_config.get("aliases", {})
    if isinstance(aliases, Mapping):
        canonical_id = aliases.get(model_id)
        if canonical_id in models and isinstance(models[canonical_id], Mapping):
            return models[canonical_id]

    return None


def _cost_for_tokens(
    tokens: Optional[int],
    price_per_1m: Optional[float],
) -> Optional[float]:
    if tokens is None or price_per_1m is None:
        return None
    return (tokens / 1_000_000) * float(price_per_1m)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)
