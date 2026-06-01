from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from src.agents.common import (
    call_llm,
    clean_summary_surface,
    documents_from_input,
    elapsed,
    get_method_config,
    is_language_mismatch,
    language_from_input,
    language_instruction,
    language_retry_instruction,
    make_agent_output,
    normalize_whitespace,
    repetition_ratio,
    sample_metadata,
    sum_token_usage,
    timed,
    tokenize,
    trim_to_word_budget,
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
        self.max_summary_words = method.get("max_summary_words")
        self.max_summary_words = int(self.max_summary_words) if self.max_summary_words else None
        self.dataset_guidance = str(method.get("dataset_guidance", "") or "").strip()

    def process(self, sample_or_documents: Any) -> Dict[str, Any]:
        start = timed()
        documents = documents_from_input(sample_or_documents)
        target_language = language_from_input(sample_or_documents)
        llm_calls: List[Dict[str, Any]] = []

        initial_summary, initial_calls = self._generate_initial_summary(documents, target_language)
        llm_calls.extend(initial_calls)

        current_summary = trim_to_word_budget(
            clean_summary_surface(initial_summary, remove_kg_markers=False) or self._fallback_initial_summary(documents),
            self.max_summary_words,
        )
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
                target_language=target_language,
            )
            llm_calls.extend(refine_call)
            current_summary = trim_to_word_budget(
                clean_summary_surface(refined_summary, remove_kg_markers=False) or current_summary,
                self.max_summary_words,
            )
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
                    "max_summary_words": self.max_summary_words,
                    "target_language": target_language,
                    "dataset_guidance": self.dataset_guidance,
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

    def _generate_initial_summary(self, documents: List[str], target_language: str) -> tuple[str, List[Dict[str, Any]]]:
        user_prompt = PI_INIT_USER_PROMPT.format(
            documents=truncate_documents(documents, self.max_input_chars)
            + _length_instruction(self.max_summary_words),
            target_language_instruction=language_instruction(target_language),
            dataset_guidance=_dataset_guidance_instruction(self.dataset_guidance),
        )
        text, call = call_llm(
            self.llm,
            prompt_id="Pi_init",
            system_prompt=PI_INIT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=self.max_summary_tokens,
        )
        calls = [call]
        if is_language_mismatch(text, target_language):
            text, retry_call = call_llm(
                self.llm,
                prompt_id="Pi_init_language_retry",
                system_prompt=PI_INIT_SYSTEM_PROMPT,
                user_prompt=user_prompt + language_retry_instruction(target_language),
                max_tokens=self.max_summary_tokens,
            )
            calls.append(retry_call)
        return text, calls

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
        target_language: str,
    ) -> tuple[str, List[Dict[str, Any]]]:
        user_prompt = PI_REF_USER_PROMPT.format(
            current_summary_text=current_summary,
            feedback_object_from_FMDS=json.dumps(
                {
                    "iteration": iteration,
                    "feedback": feedback.__dict__,
                    "surface_constraints": {
                        "remove_duplicate_openings": True,
                        "remove_partial_sentence_fragments": True,
                        "rewrite_as_coherent_summary": True,
                    },
                    "length_constraint": {
                        "max_summary_words": self.max_summary_words,
                    },
                    "target_language": target_language,
                    "dataset_guidance": self.dataset_guidance,
                    "source_documents": truncate_documents(documents, self.max_input_chars),
                },
                ensure_ascii=False,
            ),
            target_language_instruction=language_instruction(target_language),
            dataset_guidance=_dataset_guidance_instruction(self.dataset_guidance),
        )
        text, call = call_llm(
            self.llm,
            prompt_id="Pi_ref",
            system_prompt=PI_REF_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=self.max_summary_tokens,
        )
        calls = [call]
        if is_language_mismatch(text, target_language):
            text, retry_call = call_llm(
                self.llm,
                prompt_id="Pi_ref_language_retry",
                system_prompt=PI_REF_SYSTEM_PROMPT,
                user_prompt=user_prompt + language_retry_instruction(target_language),
                max_tokens=self.max_summary_tokens,
            )
            calls.append(retry_call)
        return text, calls

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
        repetition = repetition_ratio(summary)

        return MDSFeedback(
            factual_accuracy=round(min(1.0, factual_accuracy), 6),
            coverage=round(min(1.0, coverage * 2.5), 6),
            coherence=coherence,
            specificity=specificity,
            redundancy=round(max(redundancy, repetition), 6),
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


def _length_instruction(max_summary_words: int | None) -> str:
    if not max_summary_words:
        return ""
    return f"\n\nLength constraint: the summary must not exceed {max_summary_words} words."


def _dataset_guidance_instruction(dataset_guidance: str) -> str:
    return dataset_guidance or "No additional dataset-level guidance."
