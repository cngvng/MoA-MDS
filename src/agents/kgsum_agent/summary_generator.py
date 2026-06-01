from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from src.agents.common import (
    call_llm,
    clean_summary_surface,
    is_language_mismatch,
    language_instruction,
    language_retry_instruction,
    normalize_whitespace,
)
from src.agents.prompts import PI_F_SYSTEM_PROMPT, PI_F_USER_PROMPT, PI_S_SYSTEM_PROMPT, PI_S_USER_PROMPT


class MetaAwareSummaryGenerator:
    """Community summarization and meta-aware fusion for KGSum Agent."""

    def __init__(self, llm: Any, max_summary_tokens: int = 512, dataset_guidance: str = "") -> None:
        self.llm = llm
        self.max_summary_tokens = max_summary_tokens
        self.dataset_guidance = dataset_guidance.strip()

    def generate_summary(
        self,
        communities: Sequence[Dict[str, Any]],
        target_language: str = "en",
    ) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        llm_calls: List[Dict[str, Any]] = []
        community_summaries = self._community_summaries(communities, llm_calls, target_language)
        if len(community_summaries) == 1:
            return community_summaries[0]["summary"], community_summaries, llm_calls
        fused_summary, fusion_calls = self._meta_aware_fusion(community_summaries, target_language)
        llm_calls.extend(fusion_calls)
        return fused_summary, community_summaries, llm_calls

    def _community_summaries(
        self,
        communities: Sequence[Dict[str, Any]],
        llm_calls: List[Dict[str, Any]],
        target_language: str,
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for community in communities:
            linearized = self._linearize_community(community)
            user_prompt = PI_S_USER_PROMPT.format(
                community_linearized_text=linearized,
                target_language_instruction=language_instruction(target_language),
                dataset_guidance=_dataset_guidance_instruction(self.dataset_guidance),
            )
            response_text, call = call_llm(
                self.llm,
                prompt_id="Pi_s",
                system_prompt=PI_S_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=256,
            )
            llm_calls.append(call)
            summary = clean_summary_surface(response_text)
            if not summary:
                summary = self._evidence_summary([community], max_sentences=2)
            if is_language_mismatch(summary, target_language):
                response_text, retry_call = call_llm(
                    self.llm,
                    prompt_id="Pi_s_language_retry",
                    system_prompt=PI_S_SYSTEM_PROMPT,
                    user_prompt=user_prompt + language_retry_instruction(target_language),
                    max_tokens=256,
                )
                llm_calls.append(retry_call)
                summary = clean_summary_surface(response_text) or summary
            if not summary:
                raise RuntimeError(
                    f"Pi_s returned an empty summary for {community.get('community_id')}; KGSum release path does not use silent fallback summaries."
                )
            summaries.append(
                {
                    "community_id": community["community_id"],
                    "summary": summary,
                    "linearized_graph": linearized,
                    "metadata": community["metadata"],
                }
            )
        return summaries

    def _meta_aware_fusion(
        self,
        community_summaries: Sequence[Dict[str, Any]],
        target_language: str,
    ) -> tuple[str, List[Dict[str, Any]]]:
        payload = [
            {
                "summary": item["summary"],
                "metadata": {
                    "centrality": item["metadata"].get("centrality_score", item["metadata"].get("importance_score")),
                    "has_contrasts": item["metadata"].get("has_contrasts", False),
                    "key_entities": item["metadata"].get("key_entities", []),
                },
            }
            for item in community_summaries
        ]
        user_prompt = PI_F_USER_PROMPT.format(
            communities=json.dumps(payload, ensure_ascii=False),
            target_language_instruction=language_instruction(target_language),
            dataset_guidance=_dataset_guidance_instruction(self.dataset_guidance),
        )
        response_text, call = call_llm(
            self.llm,
            prompt_id="Pi_f",
            system_prompt=PI_F_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=self.max_summary_tokens,
        )
        calls = [call]
        summary = clean_summary_surface(response_text)
        if not summary:
            summary = self._evidence_summary(
                [item for item in community_summaries],
                max_sentences=5,
                summary_key="summary",
            )
        if is_language_mismatch(summary, target_language):
            response_text, retry_call = call_llm(
                self.llm,
                prompt_id="Pi_f_language_retry",
                system_prompt=PI_F_SYSTEM_PROMPT,
                user_prompt=user_prompt + language_retry_instruction(target_language),
                max_tokens=self.max_summary_tokens,
            )
            calls.append(retry_call)
            summary = clean_summary_surface(response_text) or summary
        if not summary:
            raise RuntimeError("Pi_f returned an empty KGSum summary; release path requires visible LLM output.")
        return summary, calls

    @staticmethod
    def _evidence_summary(
        communities: Sequence[Dict[str, Any]],
        max_sentences: int,
        summary_key: str | None = None,
    ) -> str:
        evidence: List[str] = []
        for community in communities:
            if summary_key and community.get(summary_key):
                evidence.append(str(community[summary_key]))
                continue
            for relation in community.get("relations", [])[:8]:
                text = normalize_whitespace(str(relation.get("evidence", "")))
                if text and text not in evidence:
                    evidence.append(text)
                if len(evidence) >= max_sentences:
                    break
            if len(evidence) >= max_sentences:
                break
        return normalize_whitespace(" ".join(evidence[:max_sentences]))

    @staticmethod
    def _linearize_community(community: Dict[str, Any]) -> str:
        relation_lines = [
            f"- {relation['source']} --{relation['relation_type']}--> {relation['target']} "
            f"(confidence={relation['confidence']}; evidence={relation.get('evidence', '')})"
            for relation in community.get("relations", [])[:20]
        ]
        return "\n".join(
            [
                f"Community: {community['community_id']}",
                f"Entities: {', '.join(community.get('nodes', []))}",
                "Relations:",
                *relation_lines,
                f"Metadata: {json.dumps(community.get('metadata', {}), ensure_ascii=False)}",
            ]
        )


def _dataset_guidance_instruction(dataset_guidance: str) -> str:
    return dataset_guidance or "No additional dataset-level guidance."
