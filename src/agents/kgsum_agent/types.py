from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    text: str
    entity_type: str
    confidence: float
    extraction_source: str
    context: str


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    relation_type: str
    confidence: float
    evidence: str
    is_contrastive: bool
