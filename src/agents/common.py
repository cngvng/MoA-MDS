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
    language = sample_or_documents.get("language")
    return {
        "sample_id": sample_or_documents.get("sample_id"),
        "dataset_name": sample_or_documents.get("dataset_name"),
        "split": sample_or_documents.get("split"),
        "language": language,
        "target_language": language,
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


KG_SURFACE_MARKERS = (
    "cộng đồng",
    "knowledge graph",
    "đồ thị tri thức",
    "centrality",
    "metadata",
    "mức độ liên kết",
    "cấu trúc trung tâm",
    "mối liên hệ",
    "thực thể",
    "các nhóm",
    "liên kết chặt chẽ",
    "quan hệ giữa",
    "community",
)


def clean_summary_surface(text: str, *, remove_kg_markers: bool = True) -> str:
    normalized = normalize_whitespace(text)
    normalized = re.sub(r"^(tóm tắt tích hợp|bản tóm tắt|tổng hợp)\s*:\s*", "", normalized, flags=re.IGNORECASE)
    sentences = split_sentences(normalized)
    cleaned: List[str] = []
    seen = set()
    for sentence in sentences:
        candidate = sentence.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if remove_kg_markers and any(marker in lowered for marker in KG_SURFACE_MARKERS):
            continue
        if re.fullmatch(r"[\wÀ-ỹ]{1,3}\.", candidate, flags=re.UNICODE):
            continue
        key = " ".join(tokenize(candidate)[:18])
        if key and key in seen:
            continue
        seen.add(key)
        cleaned.append(candidate)
    if cleaned:
        return normalize_whitespace(" ".join(cleaned))
    if remove_kg_markers and sentences:
        return ""
    return normalized


def repetition_ratio(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return round(1.0 - (len(set(tokens)) / len(tokens)), 6)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


VIETNAMESE_DIACRITICS = set(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
VIETNAMESE_STOPWORDS = {
    "và",
    "của",
    "là",
    "các",
    "những",
    "trong",
    "được",
    "cho",
    "với",
    "một",
    "này",
    "đã",
    "khi",
    "tại",
    "theo",
    "từ",
    "về",
    "có",
    "không",
    "người",
    "năm",
}
ENGLISH_STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "for",
    "with",
    "on",
    "as",
    "by",
    "that",
    "is",
    "are",
    "was",
    "were",
    "from",
    "this",
    "these",
}


def language_from_input(sample_or_documents: Any, default: str = "en") -> str:
    if isinstance(sample_or_documents, Mapping):
        return str(sample_or_documents.get("language") or default).lower()
    return default


def language_name(language: str) -> str:
    normalized = (language or "en").lower()
    if normalized == "vi":
        return "Vietnamese"
    if normalized == "en":
        return "English"
    return normalized


def language_instruction(language: str) -> str:
    name = language_name(language)
    if (language or "").lower() == "vi":
        return (
            "Target language: Vietnamese (vi). You MUST write the entire output in Vietnamese. "
            "Do not translate Vietnamese source content into English. Keep Vietnamese names and terms as written."
        )
    return f"Target language: {name}. Write the entire output in {name}."


def language_retry_instruction(language: str) -> str:
    return (
        "\n\n[Mandatory Language Correction]\n"
        f"The previous answer violated the target language contract. {language_instruction(language)} "
        "Rewrite the answer completely in the target language, preserving the same factual content."
    )


def is_language_mismatch(text: str, target_language: Optional[str]) -> bool:
    if (target_language or "").lower() != "vi":
        return False

    normalized = normalize_whitespace(text)
    if not normalized:
        return False

    tokens = tokenize(normalized)
    if len(tokens) < 6:
        return False

    lower_text = normalized.lower()
    diacritic_count = sum(1 for char in lower_text if char in VIETNAMESE_DIACRITICS)
    vi_stopword_count = sum(1 for token in tokens if token in VIETNAMESE_STOPWORDS)
    en_stopword_count = sum(1 for token in tokens if token in ENGLISH_STOPWORDS)
    token_count = max(len(tokens), 1)

    has_vietnamese_signal = diacritic_count >= 8 or (vi_stopword_count / token_count) >= 0.035
    has_english_signal = (en_stopword_count / token_count) >= 0.09 and diacritic_count < 8
    return has_english_signal or not has_vietnamese_signal


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


def trim_to_word_budget(text: str, max_words: Optional[int]) -> str:
    if max_words is None or max_words <= 0:
        return normalize_whitespace(text)

    sentences = split_sentences(text)
    selected: List[str] = []
    used_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue
        if selected and used_words + len(sentence_words) > max_words:
            break
        if not selected and len(sentence_words) > max_words:
            return " ".join(sentence_words[:max_words])
        selected.append(sentence)
        used_words += len(sentence_words)

    if selected:
        return normalize_whitespace(" ".join(selected))
    return " ".join(normalize_whitespace(text).split()[:max_words])


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
