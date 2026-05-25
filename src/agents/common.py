from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TokenUsage = Dict[str, Optional[int]]
AgentOutput = Dict[str, Any]


TOKEN_USAGE_EMPTY: TokenUsage = {
    "input_tokens": None,
    "output_tokens": None,
    "total_tokens": None,
}


def get_config_value(config: Any, path: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get") and not isinstance(config, Mapping):
        return config.get(path, default)

    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def get_method_config(config: Any, agent_key: str) -> Mapping[str, Any]:
    value = get_config_value(config, f"method.{agent_key}", {})
    return value if isinstance(value, Mapping) else {}


def documents_from_input(sample_or_documents: Any) -> List[str]:
    if isinstance(sample_or_documents, Mapping):
        documents = sample_or_documents.get("documents", [])
    else:
        documents = sample_or_documents

    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise TypeError("Agents expect a unified sample or list[str] documents.")

    cleaned = [normalize_whitespace(str(doc)) for doc in documents if normalize_whitespace(str(doc))]
    if not cleaned:
        raise ValueError("No non-empty documents provided to agent.")
    return cleaned


def sample_metadata(sample_or_documents: Any) -> Dict[str, Any]:
    if not isinstance(sample_or_documents, Mapping):
        return {}
    return {
        "sample_id": sample_or_documents.get("sample_id"),
        "dataset_name": sample_or_documents.get("dataset_name"),
        "split": sample_or_documents.get("split"),
        "language": sample_or_documents.get("language"),
    }


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text: str) -> List[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", normalized)
    sentences = [part.strip() for part in parts if part.strip()]
    if len(sentences) == 1 and len(sentences[0].split()) > 80:
        chunks = re.split(r"(?<=,)\s+|;\s+", sentences[0])
        sentences = [chunk.strip() for chunk in chunks if chunk.strip()]
    return sentences


def tokenize(text: str) -> List[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def vectorize(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_counter(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def merge_counters(counters: Iterable[Counter]) -> Counter:
    merged: Counter = Counter()
    for counter in counters:
        merged.update(counter)
    return merged


def make_agent_output(
    agent_name: str,
    output_text: str,
    metadata: Dict[str, Any],
    llm_calls: Optional[List[Dict[str, Any]]] = None,
    runtime_seconds: Optional[float] = None,
    token_usage: Optional[TokenUsage] = None,
) -> AgentOutput:
    calls = llm_calls or []
    usage = token_usage or sum_token_usage(
        [normalize_usage(call.get("usage") or call.get("token_usage")) for call in calls]
    )
    return {
        "agent_name": agent_name,
        "output_text": output_text or "",
        "metadata": metadata,
        "llm_calls": calls,
        "token_usage": usage,
        "runtime_seconds": runtime_seconds,
    }


def timed() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def call_llm(
    llm: Any,
    *,
    prompt_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    if llm is None:
        raise ValueError(f"{prompt_id} requires a configured LLM wrapper.")

    result = llm.generate_answer(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )

    result_dict = result.to_dict() if hasattr(result, "to_dict") else _object_to_dict(result)
    raw_metadata = result_dict.get("raw_metadata") or {}
    usage = normalize_usage(result_dict.get("usage"))
    call_record = {
        "prompt_id": prompt_id,
        "model_id": result_dict.get("model_id"),
        "base_url": raw_metadata.get("base_url"),
        "finish_reason": raw_metadata.get("finish_reason"),
        "usage": usage,
        "request_params": raw_metadata.get("request_params", {}),
    }
    return str(result_dict.get("text", "")), call_record


def normalize_usage(usage: Any) -> TokenUsage:
    if not usage:
        return dict(TOKEN_USAGE_EMPTY)
    if is_dataclass(usage):
        usage = asdict(usage)
    elif hasattr(usage, "to_dict"):
        usage = usage.to_dict()
    elif hasattr(usage, "__dict__") and not isinstance(usage, Mapping):
        usage = vars(usage)
    if not isinstance(usage, Mapping):
        return dict(TOKEN_USAGE_EMPTY)

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


def sum_token_usage(usages: List[TokenUsage]) -> TokenUsage:
    total: TokenUsage = dict(TOKEN_USAGE_EMPTY)
    for field in total:
        values = [usage[field] for usage in usages if usage.get(field) is not None]
        total[field] = sum(values) if values else None
    return total


def truncate_documents(documents: List[str], max_chars: int) -> str:
    chunks: List[str] = []
    remaining = max_chars
    for index, document in enumerate(documents, start=1):
        if remaining <= 0:
            break
        prefix = f"[Document {index}]\n"
        body = document[: max(0, remaining - len(prefix))]
        chunks.append(prefix + body)
        remaining -= len(prefix) + len(body)
    return "\n\n".join(chunks)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _object_to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {"text": str(obj)}
