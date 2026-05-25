from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

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

    def process(self, sample_or_documents: Any) -> Dict[str, Any]:
        start = timed()
        documents = documents_from_input(sample_or_documents)
        sentence_records = self._score_sentences(documents)
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

    def _score_sentences(self, documents: List[str]) -> List[SentenceRecord]:
        sentence_items: List[Dict[str, Any]] = []
        for doc_index, document in enumerate(documents):
            for position, sentence in enumerate(split_sentences(document)):
                sentence_items.append(
                    {
                        "document_index": doc_index,
                        "position": position,
                        "text": sentence,
                        "vector": vectorize(sentence),
                    }
                )

        if not sentence_items:
            return []

        document_centroid = merge_counters(item["vector"] for item in sentence_items)
        total_sentences = len(sentence_items)
        records: List[SentenceRecord] = []
        for index, item in enumerate(sentence_items):
            semantic = cosine_counter(item["vector"], document_centroid)
            positional = 1.0 - (index / max(total_sentences, 1))
            score = self.lambda_weight * semantic + (1.0 - self.lambda_weight) * positional
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
