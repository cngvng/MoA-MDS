from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from src.agents.common import (
    cosine_counter,
    documents_from_input,
    elapsed,
    get_method_config,
    make_agent_output,
    merge_counters,
    sample_metadata,
    split_sentences,
    timed,
    tokenize,
    vectorize,
)


@dataclass(frozen=True)
class SentenceRecord:
    index: int
    document_index: int
    position: int
    text: str
    score: float
    semantic_score: float
    positional_score: float


class ExtractorAgent:
    """Extractor Agent from the MoA paper."""

    agent_name = "extractor"

    def __init__(self, config: Any = None) -> None:
        method = get_method_config(config, "extractor")
        self.output_key = str(method.get("output_key", "r_ext"))
        self.metadata_key = str(method.get("metadata_key", "M_ext"))
        self.lambda_weight = float(method.get("lambda", 0.7))
        self.top_sentence_proportion = float(method.get("top_sentence_proportion", 0.15))
        self.redundancy_threshold = float(method.get("redundancy_threshold", 0.9))
        self.min_sentences = int(method.get("min_sentences", 3))
        self.max_sentences = method.get("max_sentences")
        self.embedding_model = str(method.get("model_name", "lexical_centroid"))
        self.max_length = int(method.get("max_length", 4096))
        self.embedding_batch_size = int(method.get("embedding_batch_size", 8))
        self.allow_lexical_fallback = bool(method.get("allow_lexical_fallback", True))
        self._embedding_backend = "lexical_centroid"
        self._fallback_reason: Optional[str] = None
        self._tokenizer = None
        self._model = None

        if self.embedding_model != "lexical_centroid":
            self._load_transformer_backend()

    def process(self, sample_or_documents: Any) -> Dict[str, Any]:
        start = timed()
        documents = documents_from_input(sample_or_documents)
        dataset_name = ""
        document_roles: List[str] = []
        if isinstance(sample_or_documents, Mapping):
            dataset_name = str(sample_or_documents.get("dataset_name", "")).lower()
            metadata = sample_or_documents.get("metadata", {})
            if isinstance(metadata, Mapping):
                raw_roles = metadata.get("document_roles", [])
                if isinstance(raw_roles, list):
                    document_roles = [str(role).lower() for role in raw_roles]

        sentence_records = self._score_sentences(
            documents,
            dataset_name=dataset_name,
            document_roles=document_roles,
        )
        selected_records = self._select_sentences(sentence_records)
        output_text = " ".join(record.text for record in sorted(selected_records, key=lambda item: item.index))

        metadata = {
            self.metadata_key: {
                "sentence_scores": [record.__dict__ for record in sentence_records],
                "selected_indices": [record.index for record in selected_records],
                "selected_document_indices": sorted({record.document_index for record in selected_records}),
                "salience": _mean([record.score for record in selected_records]),
                "density": self._density(output_text, documents),
                "coverage": self._coverage(selected_records, documents),
                "redundancy_filtered": True,
                "theta_r": self.redundancy_threshold,
                "params": {
                    "lambda": self.lambda_weight,
                    "top_sentence_proportion": self.top_sentence_proportion,
                    "min_sentences": self.min_sentences,
                    "max_sentences": self.max_sentences,
                    "embedding_model": self.embedding_model,
                    "embedding_backend": self._embedding_backend,
                    "fallback_reason": self._fallback_reason,
                },
            },
            "output_key": self.output_key,
            **sample_metadata(sample_or_documents),
        }

        return make_agent_output(
            agent_name=self.agent_name,
            output_text=output_text,
            metadata=metadata,
            llm_calls=[],
            runtime_seconds=elapsed(start),
        )

    def _score_sentences(
        self,
        documents: List[str],
        *,
        dataset_name: str = "",
        document_roles: Optional[List[str]] = None,
    ) -> List[SentenceRecord]:
        sentence_items: List[Dict[str, Any]] = []
        document_sentence_counts: Dict[int, int] = {}
        for doc_index, document in enumerate(documents):
            sentences = split_sentences(document)
            document_sentence_counts[doc_index] = len(sentences)
            for position, sentence in enumerate(sentences):
                sentence_text = re.sub(r"^\[(?:Target|Cited)[^\]]*\]\s*", "", sentence, flags=re.IGNORECASE).strip()
                if not sentence_text:
                    continue
                sentence_items.append(
                    {
                        "document_index": doc_index,
                        "position": position,
                        "text": sentence_text,
                        "vector": vectorize(sentence_text),
                    }
                )

        if not sentence_items:
            return []

        transformer_embeddings = self._sentence_embeddings([item["text"] for item in sentence_items])
        total_sentences = len(sentence_items)
        records: List[SentenceRecord] = []
        lexical_centroid = merge_counters(item["vector"] for item in sentence_items)
        if transformer_embeddings is not None:
            document_centroid_vector = _mean_vector(transformer_embeddings)

        for index, item in enumerate(sentence_items):
            lexical_semantic = cosine_counter(item["vector"], lexical_centroid)
            if transformer_embeddings is None:
                semantic = lexical_semantic
            else:
                transformer_semantic = _cosine_vector(transformer_embeddings[index], document_centroid_vector)
                semantic = 0.5 * transformer_semantic + 0.5 * lexical_semantic
            document_sentence_count = max(document_sentence_counts.get(item["document_index"], 1), 1)
            positional = 1.0 - (item["position"] / document_sentence_count)
            score = self.lambda_weight * semantic + (1.0 - self.lambda_weight) * positional
            if dataset_name == "multi_xscience":
                role_bonus = _multixscience_role_bonus(
                    sentence=item["text"],
                    role=(document_roles[item["document_index"]] if document_roles and item["document_index"] < len(document_roles) else ""),
                )
                score = score * role_bonus["multiplier"] + role_bonus["additive"]
            records.append(
                SentenceRecord(
                    index=index,
                    document_index=item["document_index"],
                    position=item["position"],
                    text=item["text"],
                    score=round(score, 6),
                    semantic_score=round(semantic, 6),
                    positional_score=round(positional, 6),
                )
            )
        return records

    def _load_transformer_backend(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            if not self.allow_lexical_fallback:
                raise RuntimeError(
                    "Transformer extractor requested but torch/transformers are not available."
                ) from exc
            self._fallback_reason = f"transformer_import_failed:{type(exc).__name__}"
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
            self._model = AutoModel.from_pretrained(self.embedding_model)
            self._model.eval()
            self._embedding_backend = "transformer_cls"
        except Exception as exc:
            if not self.allow_lexical_fallback:
                raise RuntimeError(f"Failed to load extractor model: {self.embedding_model}") from exc
            self._fallback_reason = f"transformer_load_failed:{type(exc).__name__}"
            self._tokenizer = None
            self._model = None

    def _sentence_embeddings(self, sentences: List[str]) -> Optional[List[List[float]]]:
        if self._tokenizer is None or self._model is None:
            return None

        import torch

        embeddings: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(sentences), self.embedding_batch_size):
                batch = sentences[start : start + self.embedding_batch_size]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                outputs = self._model(**inputs)
                cls_vectors = outputs.last_hidden_state[:, 0, :]
                embeddings.extend(cls_vectors.cpu().tolist())
        return embeddings

    def _select_sentences(self, records: List[SentenceRecord]) -> List[SentenceRecord]:
        if not records:
            return []

        candidate_count = max(
            self.min_sentences,
            math.ceil(len(records) * self.top_sentence_proportion),
        )
        if self.max_sentences is not None:
            candidate_count = min(candidate_count, int(self.max_sentences))
        candidate_count = min(candidate_count, len(records))

        candidates = sorted(records, key=lambda item: item.score, reverse=True)[:candidate_count]
        selected: List[SentenceRecord] = []
        selected_vectors = []

        for candidate in candidates:
            candidate_vector = vectorize(candidate.text)
            if all(
                cosine_counter(candidate_vector, selected_vector) <= self.redundancy_threshold
                for selected_vector in selected_vectors
            ):
                selected.append(candidate)
                selected_vectors.append(candidate_vector)

        if not selected:
            selected = candidates[:1]
        return sorted(selected, key=lambda item: item.index)

    @staticmethod
    def _density(summary: str, documents: List[str]) -> float:
        source_tokens = sum(len(tokenize(document)) for document in documents)
        summary_tokens = len(tokenize(summary))
        if source_tokens == 0:
            return 0.0
        return round(summary_tokens / source_tokens, 6)

    @staticmethod
    def _coverage(selected: List[SentenceRecord], documents: List[str]) -> float:
        if not documents:
            return 0.0
        covered = len({record.document_index for record in selected})
        return round(covered / len(documents), 6)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _multixscience_role_bonus(sentence: str, role: str) -> Dict[str, float]:
    lowered = sentence.lower()
    multiplier = 1.0
    if role == "target_paper":
        multiplier = 0.92
    elif role == "cited_paper":
        multiplier = 1.08

    relation_terms = (
        "however",
        "while",
        "whereas",
        "in contrast",
        "compared",
        "similar",
        "prior work",
        "previous work",
        "related work",
        "in this work",
        "our work",
    )
    additive = 0.0
    if any(term in lowered for term in relation_terms):
        additive += 0.07
    if "we " in lowered and role == "target_paper":
        additive += 0.03
    return {"multiplier": multiplier, "additive": additive}


def _mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def _cosine_vector(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(l_value * r_value for l_value, r_value in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
