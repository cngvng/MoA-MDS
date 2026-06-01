from __future__ import annotations

from typing import Any, Dict, List

from src.agents.common import (
    documents_from_input,
    elapsed,
    get_method_config,
    language_from_input,
    make_agent_output,
    sample_metadata,
    sum_token_usage,
    timed,
    tokenize,
)
from src.agents.kgsum_agent.community_detector import CommunityDetector
from src.agents.kgsum_agent.entity_extractor import HybridEntityExtractor
from src.agents.kgsum_agent.relation_extractor import ContrastiveRelationExtractor
from src.agents.kgsum_agent.summary_generator import MetaAwareSummaryGenerator
from src.agents.kgsum_agent.types import Entity, Relation


class KGSumAgent:
    """KGSum Agent: dynamic KG construction and knowledge-enhanced summary generation."""

    agent_name = "kgsum"

    def __init__(self, llm: Any, config: Any = None) -> None:
        method = get_method_config(config, "kgsum")
        self.llm = llm
        self.output_key = str(method.get("output_key", "r_kg"))
        self.metadata_key = str(method.get("metadata_key", "M_kg"))
        self.entity_similarity_threshold = float(method.get("entity_similarity_threshold", 0.85))
        self.relation_confidence_threshold = float(method.get("relation_confidence_threshold", 0.6))
        self.top_k_communities = int(method.get("top_k_communities", 5))
        self.min_community_size = int(method.get("min_community_size", 3))
        self.max_cluster_size = int(method.get("max_cluster_size", 50))
        self.leiden_resolution = float(method.get("leiden_resolution", 1.0))
        self.max_input_chars = int(method.get("max_input_chars", 6000))
        self.max_summary_tokens = int(method.get("max_summary_tokens", 1024))
        self.spacy_model = method.get("spacy_model")
        self.dataset_guidance = str(method.get("dataset_guidance", "") or "").strip()

        self.entity_extractor = HybridEntityExtractor(
            llm=llm,
            similarity_threshold=self.entity_similarity_threshold,
            max_input_chars=self.max_input_chars,
            spacy_model=self.spacy_model,
        )
        self.relation_extractor = ContrastiveRelationExtractor(
            llm=llm,
            confidence_threshold=self.relation_confidence_threshold,
            max_input_chars=self.max_input_chars,
        )
        self.community_detector = CommunityDetector(
            top_k=self.top_k_communities,
            min_community_size=self.min_community_size,
            max_cluster_size=self.max_cluster_size,
            leiden_resolution=self.leiden_resolution,
        )
        self.summary_generator = MetaAwareSummaryGenerator(
            llm=llm,
            max_summary_tokens=self.max_summary_tokens,
            dataset_guidance=self.dataset_guidance,
        )

    def process(self, sample_or_documents: Any) -> Dict[str, Any]:
        start = timed()
        documents = documents_from_input(sample_or_documents)
        target_language = language_from_input(sample_or_documents)

        entities, entity_calls = self.entity_extractor.extract_entities(documents)
        relations, relation_calls = self.relation_extractor.extract_relations(documents, entities)
        graph, communities = self.community_detector.detect_communities(entities, relations)
        output_text, community_summaries, summary_calls = self.summary_generator.generate_summary(
            communities,
            target_language=target_language,
        )

        llm_calls = [*entity_calls, *relation_calls, *summary_calls]
        graph_metrics = self.community_detector.graph_metrics(graph, communities, relations)
        metadata = {
            self.metadata_key: {
                "entities": [entity.__dict__ for entity in entities],
                "entity_count": len(entities),
                "relations": [relation.__dict__ for relation in relations],
                "relation_confidence": {
                    "threshold": self.relation_confidence_threshold,
                    "average": _mean([relation.confidence for relation in relations]),
                },
                "contrast_flags": [relation.relation_type for relation in relations if relation.is_contrastive],
                "contrastive_relation_count": graph_metrics["contrastive_relation_count"],
                "communities": communities,
                "community_summaries": community_summaries,
                "centrality": self.community_detector.centrality_payload(graph),
                "modularity": graph_metrics["modularity"],
                "fact_density": self._fact_density(relations, output_text),
                "key_entities": self.community_detector.key_entities(graph),
                "graph_metrics": graph_metrics,
                "params": {
                    "entity_similarity_threshold": self.entity_similarity_threshold,
                    "relation_confidence_threshold": self.relation_confidence_threshold,
                    "community_algorithm": "leiden",
                    "top_k_communities": self.top_k_communities,
                    "min_community_size": self.min_community_size,
                    "max_cluster_size": self.max_cluster_size,
                    "leiden_resolution": self.leiden_resolution,
                    "spacy_model": self.spacy_model,
                    "target_language": target_language,
                    "dataset_guidance": self.dataset_guidance,
                },
            },
            "output_key": self.output_key,
            **sample_metadata(sample_or_documents),
        }

        return make_agent_output(
            agent_name=self.agent_name,
            output_text=output_text,
            metadata=metadata,
            llm_calls=llm_calls,
            runtime_seconds=elapsed(start),
            token_usage=sum_token_usage([call["usage"] for call in llm_calls]),
        )

    @staticmethod
    def _fact_density(relations: List[Relation], summary: str) -> float:
        summary_tokens = max(len(tokenize(summary)), 1)
        return round(len(relations) / summary_tokens, 6)


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)
