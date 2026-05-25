from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any, Dict, List, Mapping, Sequence

from src.agents.common import call_llm, normalize_whitespace, split_sentences, truncate_documents
from src.agents.kgsum_agent.types import Entity, Relation
from src.agents.prompts import PI_REL_SYSTEM_PROMPT, PI_REL_USER_PROMPT


class ContrastiveRelationExtractor:
    """Relation extraction for KGSum Agent with contrastive relation support."""

    def __init__(
        self,
        llm: Any,
        confidence_threshold: float = 0.6,
        max_input_chars: int = 6000,
    ) -> None:
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.max_input_chars = max_input_chars

    def extract_relations(
        self,
        documents: Sequence[str],
        entities: Sequence[Entity],
    ) -> tuple[List[Relation], List[Dict[str, Any]]]:
        llm_calls: List[Dict[str, Any]] = []
        relations = [
            *self._llm_relations(documents, entities, llm_calls),
            *self._cooccurrence_relations(documents, entities),
        ]
        return self._dedupe_and_filter(relations), llm_calls

    def _llm_relations(
        self,
        documents: Sequence[str],
        entities: Sequence[Entity],
        llm_calls: List[Dict[str, Any]],
    ) -> List[Relation]:
        entity_names = [entity.text for entity in entities[:50]]
        user_prompt = PI_REL_USER_PROMPT.format(
            text=truncate_documents(list(documents), self.max_input_chars),
            entities=json.dumps(entity_names, ensure_ascii=False),
        )
        response_text, call = call_llm(
            self.llm,
            prompt_id="Pi_rel",
            system_prompt=PI_REL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=512,
        )
        llm_calls.append(call)

        entity_lookup = {entity.text.lower(): entity.text for entity in entities}
        relations: List[Relation] = []
        parsed = _parse_json_array(response_text)
        for item in parsed:
            if not isinstance(item, Mapping):
                continue
            source = _match_entity(str(item.get("head", "")), entity_lookup)
            target = _match_entity(str(item.get("tail", "")), entity_lookup)
            if not source or not target or source == target:
                continue
            relation_type = normalize_whitespace(str(item.get("relation", "related to"))) or "related to"
            relations.append(
                Relation(
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    confidence=_confidence(item.get("confidence", 0.65)),
                    evidence=normalize_whitespace(str(item.get("evidence", "")))[:300],
                    is_contrastive=relation_type == "contrastive relation",
                )
            )
        return relations

    def _cooccurrence_relations(
        self,
        documents: Sequence[str],
        entities: Sequence[Entity],
    ) -> List[Relation]:
        names = [entity.text for entity in entities[:50]]
        relations: List[Relation] = []
        for document in documents:
            for sentence in split_sentences(document):
                present = [name for name in names if name.lower() in sentence.lower()]
                for source, target in combinations(present[:8], 2):
                    contrastive = _contains_contrast(sentence)
                    relations.append(
                        Relation(
                            source=source,
                            target=target,
                            relation_type="contrastive relation" if contrastive else "related to",
                            confidence=0.68 if contrastive else 0.62,
                            evidence=sentence[:300],
                            is_contrastive=contrastive,
                        )
                    )
        return relations

    def _dedupe_and_filter(self, relations: List[Relation]) -> List[Relation]:
        deduped: Dict[tuple[str, str, str], Relation] = {}
        for relation in relations:
            if relation.confidence < self.confidence_threshold:
                continue
            key = (relation.source.lower(), relation.target.lower(), relation.relation_type.lower())
            if key not in deduped or relation.confidence > deduped[key].confidence:
                deduped[key] = relation
        return list(deduped.values())[:120]


def _parse_json_array(text: str) -> List[Any]:
    match = re.search(r"\[[\s\S]*\]", text.strip())
    payload = match.group(0) if match else text.strip()
    try:
        parsed = json.loads(payload)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _match_entity(candidate: str, entity_lookup: Mapping[str, str]) -> str | None:
    normalized = candidate.strip().lower()
    if normalized in entity_lookup:
        return entity_lookup[normalized]
    for key, value in entity_lookup.items():
        if normalized and (normalized in key or key in normalized):
            return value
    return None


def _confidence(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 6)
    except Exception:
        return 0.0


def _contains_contrast(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(
        marker in lowered
        for marker in ["however", "but", "although", "whereas", "while", "unlike", "in contrast", "contrary"]
    )
