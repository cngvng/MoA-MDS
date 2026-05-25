from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence, Set

from src.agents.common import call_llm, normalize_whitespace, split_sentences, tokenize, truncate_documents
from src.agents.kgsum_agent.types import Entity
from src.agents.prompts import PI_E_SYSTEM_PROMPT, PI_E_USER_PROMPT


class HybridEntityExtractor:
    """Hybrid entity extraction for KGSum Agent: spaCy/rules/LLM, then merge."""

    def __init__(
        self,
        llm: Any,
        similarity_threshold: float = 0.85,
        max_input_chars: int = 6000,
        spacy_model: str | None = None,
    ) -> None:
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.max_input_chars = max_input_chars
        self.nlp = self._load_spacy(spacy_model)

    def extract_entities(self, documents: Sequence[str]) -> tuple[List[Entity], List[Dict[str, Any]]]:
        llm_calls: List[Dict[str, Any]] = []
        entities = [
            *self._spacy_entities(documents),
            *self._rule_entities(documents),
            *self._llm_entities(documents, llm_calls),
        ]
        return self._merge_entities(entities), llm_calls

    def _load_spacy(self, model_name: str | None) -> Any:
        if not model_name:
            return None
        try:
            import spacy

            return spacy.load(model_name)
        except Exception:
            return None

    def _spacy_entities(self, documents: Sequence[str]) -> List[Entity]:
        if self.nlp is None:
            return []
        entities: List[Entity] = []
        for document in documents:
            for ent in self.nlp(document[: self.max_input_chars]).ents:
                entities.append(
                    Entity(
                        text=normalize_whitespace(ent.text),
                        entity_type=str(ent.label_),
                        confidence=0.78,
                        extraction_source="spacy",
                        context=normalize_whitespace(ent.sent.text)[:240],
                    )
                )
        return entities

    def _rule_entities(self, documents: Sequence[str]) -> List[Entity]:
        pattern = re.compile(
            r"\b(?:[A-Z][\w&.-]+(?:\s+[A-Z][\w&.-]+){0,4}|\d+(?:[.,]\d+)*(?:\s?(?:%|percent|million|billion|USD|VND))?)\b"
        )
        entities: List[Entity] = []
        for document in documents:
            for sentence in split_sentences(document):
                for match in pattern.finditer(sentence):
                    text = normalize_whitespace(match.group(0))
                    if len(text) < 2:
                        continue
                    entity_type = "NUMERIC" if any(char.isdigit() for char in text) else "NAMED_ENTITY"
                    entities.append(
                        Entity(
                            text=text,
                            entity_type=entity_type,
                            confidence=0.82 if entity_type == "NUMERIC" else 0.72,
                            extraction_source="rule",
                            context=sentence[:240],
                        )
                    )
        return entities

    def _llm_entities(self, documents: Sequence[str], llm_calls: List[Dict[str, Any]]) -> List[Entity]:
        text = truncate_documents(list(documents), self.max_input_chars)
        seed_entities = [entity.text for entity in self._rule_entities(documents)[:80]]
        user_prompt = PI_E_USER_PROMPT.format(
            text=text,
            entities=json.dumps(seed_entities, ensure_ascii=False),
        )
        response_text, call = call_llm(
            self.llm,
            prompt_id="Pi_e",
            system_prompt=PI_E_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=384,
        )
        llm_calls.append(call)

        pairs = _parse_json(response_text)
        related_entities: Set[str] = set()
        if isinstance(pairs, list):
            for item in pairs:
                if not isinstance(item, str) or "|" not in item:
                    continue
                left, right = [normalize_whitespace(part) for part in item.split("|", 1)]
                related_entities.update([left, right])

        return [
            Entity(
                text=entity,
                entity_type="LLM_RELATED_ENTITY",
                confidence=0.84,
                extraction_source="llm",
                context="Pi_e related entity pair",
            )
            for entity in related_entities
            if entity
        ]

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        merged: List[Entity] = []
        for entity in sorted(entities, key=lambda item: item.confidence, reverse=True):
            tokens = set(tokenize(entity.text))
            if not tokens:
                continue
            duplicate_index = None
            for index, existing in enumerate(merged):
                if _jaccard(tokens, set(tokenize(existing.text))) >= self.similarity_threshold:
                    duplicate_index = index
                    break
            if duplicate_index is None:
                merged.append(entity)
            elif entity.confidence > merged[duplicate_index].confidence:
                merged[duplicate_index] = entity
        return merged[:80]


def _parse_json(text: str) -> Any:
    match = re.search(r"\[[\s\S]*\]", text.strip())
    payload = match.group(0) if match else text.strip()
    try:
        return json.loads(payload)
    except Exception:
        return []


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
