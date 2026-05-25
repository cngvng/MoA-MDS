from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from src.agents.common import call_llm, normalize_whitespace
from src.agents.prompts import PI_F_SYSTEM_PROMPT, PI_F_USER_PROMPT, PI_S_SYSTEM_PROMPT, PI_S_USER_PROMPT


class MetaAwareSummaryGenerator:
    """Community summarization and meta-aware fusion for KGSum Agent."""

    def __init__(self, llm: Any, max_summary_tokens: int = 512) -> None:
        self.llm = llm
        self.max_summary_tokens = max_summary_tokens

    def generate_summary(
        self,
        communities: Sequence[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        llm_calls: List[Dict[str, Any]] = []
        community_summaries = self._community_summaries(communities, llm_calls)
        if len(community_summaries) == 1:
            return community_summaries[0]["summary"], community_summaries, llm_calls
        fused_summary, fusion_call = self._meta_aware_fusion(community_summaries)
        llm_calls.append(fusion_call)
        return fused_summary, community_summaries, llm_calls

    def _community_summaries(
        self,
        communities: Sequence[Dict[str, Any]],
        llm_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for community in communities:
            linearized = self._linearize_community(community)
            response_text, call = call_llm(
                self.llm,
                prompt_id="Pi_s",
                system_prompt=PI_S_SYSTEM_PROMPT,
                user_prompt=PI_S_USER_PROMPT.format(community_linearized_text=linearized),
                max_tokens=256,
            )
            llm_calls.append(call)
            summary = normalize_whitespace(response_text)
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
    ) -> tuple[str, Dict[str, Any]]:
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
        response_text, call = call_llm(
            self.llm,
            prompt_id="Pi_f",
            system_prompt=PI_F_SYSTEM_PROMPT,
            user_prompt=PI_F_USER_PROMPT.format(communities=json.dumps(payload, ensure_ascii=False)),
            max_tokens=self.max_summary_tokens,
        )
        summary = normalize_whitespace(response_text)
        if not summary:
            raise RuntimeError("Pi_f returned an empty KGSum summary; release path requires visible LLM output.")
        return summary, call

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
