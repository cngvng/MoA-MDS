from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from src.agents.common import (
    call_llm,
    documents_from_input,
    elapsed,
    get_method_config,
    make_agent_output,
    normalize_whitespace,
    sample_metadata,
    sum_token_usage,
    timed,
    tokenize,
    truncate_documents,
)
from src.agents.prompts import PI_INIT_SYSTEM_PROMPT, PI_INIT_USER_PROMPT, PI_REF_SYSTEM_PROMPT, PI_REF_USER_PROMPT


@dataclass(frozen=True)
class MDSFeedback:
    factual_accuracy: float
    coverage: float
    coherence: float
    specificity: float
    redundancy: float

    @property
    def quality_score(self) -> float:
        score = (
            self.factual_accuracy * 0.30
            + self.coverage * 0.30
            + self.coherence * 0.20
            + self.specificity * 0.15
            + (1.0 - self.redundancy) * 0.05
        )
        return round(max(0.0, min(1.0, score)), 6)


class AbstractorAgent:
    """Abstractor Agent from the MoA paper."""

    agent_name = "abstractor"

    def __init__(self, llm: Any, config: Any = None) -> None:
        method = get_method_config(config, "abstractor")
        self.llm = llm
        self.output_key = str(method.get("output_key", "r_abs"))
        self.metadata_key = str(method.get("metadata_key", "M_abs"))
        self.max_refinement_iterations = int(method.get("max_refinement_iterations", 2))
        self.quality_threshold = float(method.get("quality_threshold", 0.7))
        self.max_input_chars = int(method.get("max_input_chars", 8000))
        self.max_summary_tokens = int(method.get("max_summary_tokens", 512))

    def process(self, sample_or_documents: Any) -> Dict[str, Any]:
        start = timed()
        documents = documents_from_input(sample_or_documents)
        llm_calls: List[Dict[str, Any]] = []

        initial_summary, initial_call = self._generate_initial_summary(documents)
        llm_calls.append(initial_call)

        current_summary = normalize_whitespace(initial_summary) or self._fallback_initial_summary(documents)
        feedback_history: List[Dict[str, Any]] = []
        refined_summaries: List[str] = []

        for iteration in range(1, self.max_refinement_iterations + 1):
            feedback = self._evaluate_mds_quality(current_summary, documents)
            feedback_payload = {
                "iteration": iteration - 1,
                "feedback": feedback.__dict__,
                "quality_score": feedback.quality_score,
            }
            feedback_history.append(feedback_payload)
            if feedback.quality_score >= self.quality_threshold:
                break

            refined_summary, refine_call = self._apply_refinement_prompt(
                current_summary=current_summary,
                feedback=feedback,
                documents=documents,
                iteration=iteration,
            )
            llm_calls.append(refine_call)
            current_summary = normalize_whitespace(refined_summary) or current_summary
            refined_summaries.append(current_summary)

        final_feedback = self._evaluate_mds_quality(current_summary, documents)
        metadata = {
            self.metadata_key: {
                "initial_summary": normalize_whitespace(initial_summary),
                "feedback": feedback_history,
                "refined_summaries": refined_summaries,
                "quality_scores": {
                    "factual_accuracy": final_feedback.factual_accuracy,
                    "coverage": final_feedback.coverage,
                    "coherence": final_feedback.coherence,
                    "specificity": final_feedback.specificity,
                    "redundancy": final_feedback.redundancy,
                    "final_quality_score": final_feedback.quality_score,
                },
                "refinement_iterations": len(refined_summaries),
                "quality_threshold": self.quality_threshold,
                "params": {
                    "max_refinement_iterations": self.max_refinement_iterations,
                    "quality_threshold": self.quality_threshold,
                },
            },
            "output_key": self.output_key,
            **sample_metadata(sample_or_documents),
        }

        return make_agent_output(
            agent_name=self.agent_name,
            output_text=current_summary,
            metadata=metadata,
            llm_calls=llm_calls,
            runtime_seconds=elapsed(start),
            token_usage=sum_token_usage([call["usage"] for call in llm_calls]),
        )

    def _generate_initial_summary(self, documents: List[str]) -> tuple[str, Dict[str, Any]]:
        return call_llm(
            self.llm,
            prompt_id="Pi_init",
            system_prompt=PI_INIT_SYSTEM_PROMPT,
            user_prompt=PI_INIT_USER_PROMPT.format(
                documents=truncate_documents(documents, self.max_input_chars)
            ),
            max_tokens=self.max_summary_tokens,
        )

    @staticmethod
    def _fallback_initial_summary(documents: List[str]) -> str:
        selected = []
        for document in documents:
            sentences = [part.strip() for part in document.split(".") if part.strip()]
            if sentences:
                selected.append(sentences[0])
            if len(selected) >= 5:
                break
        return ". ".join(selected) + ("." if selected else "")

    def _apply_refinement_prompt(
        self,
        *,
        current_summary: str,
        feedback: MDSFeedback,
        documents: List[str],
        iteration: int,
    ) -> tuple[str, Dict[str, Any]]:
        return call_llm(
            self.llm,
            prompt_id="Pi_ref",
            system_prompt=PI_REF_SYSTEM_PROMPT,
            user_prompt=PI_REF_USER_PROMPT.format(
                current_summary_text=current_summary,
                feedback_object_from_FMDS=json.dumps(
                    {
                        "iteration": iteration,
                        "feedback": feedback.__dict__,
                        "source_documents": truncate_documents(documents, self.max_input_chars),
                    },
                    ensure_ascii=False,
                ),
            ),
            max_tokens=self.max_summary_tokens,
        )

    def _evaluate_mds_quality(self, summary: str, documents: List[str]) -> MDSFeedback:
        source_tokens = set()
        for document in documents:
            source_tokens.update(tokenize(document))
        summary_tokens = tokenize(summary)
        summary_token_set = set(summary_tokens)

        overlap = len(source_tokens & summary_token_set)
        coverage = overlap / max(len(source_tokens), 1)
        factual_accuracy = overlap / max(len(summary_token_set), 1)
        coherence = _coherence_score(summary)
        specificity = _specificity_score(summary)
        redundancy = _redundancy_score(summary_tokens)

        return MDSFeedback(
            factual_accuracy=round(min(1.0, factual_accuracy), 6),
            coverage=round(min(1.0, coverage * 2.5), 6),
            coherence=coherence,
            specificity=specificity,
            redundancy=redundancy,
        )


def _coherence_score(summary: str) -> float:
    sentences = [sentence for sentence in summary.split(".") if sentence.strip()]
    if not sentences:
        return 0.0
    avg_len = sum(len(tokenize(sentence)) for sentence in sentences) / len(sentences)
    if 10 <= avg_len <= 35:
        return 0.85
    if 6 <= avg_len <= 45:
        return 0.7
    return 0.55


def _specificity_score(summary: str) -> float:
    tokens = tokenize(summary)
    if not tokens:
        return 0.0
    numeric_bonus = 0.15 if any(any(char.isdigit() for char in token) for token in tokens) else 0.0
    named_like = sum(1 for token in summary.split() if token[:1].isupper())
    return round(min(1.0, 0.55 + numeric_bonus + min(named_like / 25, 0.3)), 6)


def _redundancy_score(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    return round(1.0 - unique_ratio, 6)
