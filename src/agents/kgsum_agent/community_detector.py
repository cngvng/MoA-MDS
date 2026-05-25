from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Set

import igraph as ig
import leidenalg
import networkx as nx

from src.agents.common import tokenize
from src.agents.kgsum_agent.types import Entity, Relation


class CommunityDetector:
    """Leiden community detection and metadata extraction for KGSum Agent."""

    def __init__(
        self,
        top_k: int = 5,
        min_community_size: int = 3,
        max_cluster_size: int = 50,
        leiden_resolution: float = 1.0,
    ) -> None:
        self.top_k = top_k
        self.min_community_size = min_community_size
        self.max_cluster_size = max_cluster_size
        self.leiden_resolution = leiden_resolution

    def detect_communities(
        self,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
    ) -> tuple[nx.Graph, List[Dict[str, Any]]]:
        graph = self._build_graph(entities, relations)
        raw_communities = self._leiden_communities(graph)
        if not raw_communities and graph.number_of_nodes() > 0:
            raw_communities = [set(graph.nodes())]

        communities: List[Dict[str, Any]] = []
        for index, nodes in enumerate(raw_communities):
            selected_nodes = sorted(str(node) for node in nodes)
            if len(selected_nodes) < self.min_community_size and len(raw_communities) > 1:
                continue
            selected_nodes = selected_nodes[: self.max_cluster_size]
            community_relations = [
                relation
                for relation in relations
                if relation.source in selected_nodes and relation.target in selected_nodes
            ]
            community = self._community_payload(index, graph, selected_nodes, community_relations)
            communities.append(community)

        ranked = sorted(communities, key=lambda item: item["importance_score"], reverse=True)
        return graph, ranked[: self.top_k]

    def _build_graph(self, entities: Sequence[Entity], relations: Sequence[Relation]) -> nx.Graph:
        graph = nx.Graph()
        for entity in entities:
            graph.add_node(entity.text, entity_type=entity.entity_type, confidence=entity.confidence)
        for relation in relations:
            graph.add_edge(
                relation.source,
                relation.target,
                weight=relation.confidence,
                relation_type=relation.relation_type,
                is_contrastive=relation.is_contrastive,
                evidence=relation.evidence,
            )
        return graph

    def _leiden_communities(self, graph: nx.Graph) -> List[Set[str]]:
        if graph.number_of_nodes() == 0:
            return []
        nodes = list(graph.nodes())
        node_to_index = {node: index for index, node in enumerate(nodes)}
        edges = [(node_to_index[source], node_to_index[target]) for source, target in graph.edges()]
        igraph_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        igraph_graph.vs["name"] = nodes
        if edges:
            igraph_graph.es["weight"] = [
                float(graph[source][target].get("weight", 1.0))
                for source, target in graph.edges()
            ]
        partition = leidenalg.find_partition(
            igraph_graph,
            leidenalg.ModularityVertexPartition,
            weights=igraph_graph.es["weight"] if edges else None,
        )
        return [{str(nodes[index]) for index in community} for community in partition]

    def _community_payload(
        self,
        index: int,
        graph: nx.Graph,
        nodes: List[str],
        relations: Sequence[Relation],
    ) -> Dict[str, Any]:
        subgraph = graph.subgraph(nodes)
        centrality = nx.degree_centrality(subgraph) if subgraph.number_of_nodes() > 1 else {nodes[0]: 1.0}
        contrastive_relations = [relation.__dict__ for relation in relations if relation.is_contrastive]
        numerical_facts = [node for node in nodes if any(char.isdigit() for char in node)]
        relation_payload = [relation.__dict__ for relation in relations]
        importance = min(
            1.0,
            0.35
            + 0.08 * len(nodes)
            + 0.08 * len(relation_payload)
            + 0.12 * len(contrastive_relations)
            + 0.05 * len(numerical_facts),
        )
        return {
            "community_id": f"C{index + 1}",
            "nodes": nodes,
            "relations": relation_payload,
            "metadata": {
                "algorithm": "leiden",
                "implementation": "leidenalg.ModularityVertexPartition",
                "centrality": {node: round(score, 6) for node, score in centrality.items()},
                "centrality_score": round(max(centrality.values()) if centrality else 0.0, 6),
                "has_contrasts": bool(contrastive_relations),
                "contrasts": contrastive_relations,
                "contrast_flags": [relation["relation_type"] for relation in contrastive_relations],
                "key_entities": sorted(centrality, key=centrality.get, reverse=True)[:5],
                "fact_density": self._fact_density(nodes, relation_payload),
                "numerical_facts": numerical_facts,
                "importance_score": round(importance, 6),
            },
            "importance_score": round(importance, 6),
        }

    def graph_metrics(
        self,
        graph: nx.Graph,
        communities: Sequence[Dict[str, Any]],
        relations: Sequence[Relation],
    ) -> Dict[str, Any]:
        node_sets = [set(community["nodes"]) for community in communities if community.get("nodes")]
        modularity = None
        if node_sets and graph.number_of_edges() > 0:
            covered_nodes = set().union(*node_sets)
            subgraph = graph.subgraph(covered_nodes)
            if subgraph.number_of_edges() > 0:
                modularity = round(nx.algorithms.community.modularity(subgraph, node_sets, weight="weight"), 6)
        return {
            "num_entities": graph.number_of_nodes(),
            "num_relations": len(relations),
            "num_communities": len(communities),
            "modularity": modularity,
            "contrastive_relation_count": sum(1 for relation in relations if relation.is_contrastive),
        }

    @staticmethod
    def centrality_payload(graph: nx.Graph) -> Dict[str, float]:
        if graph.number_of_nodes() == 0:
            return {}
        return {node: round(score, 6) for node, score in nx.degree_centrality(graph).items()}

    @staticmethod
    def key_entities(graph: nx.Graph) -> List[str]:
        if graph.number_of_nodes() == 0:
            return []
        centrality = nx.degree_centrality(graph)
        return sorted(centrality, key=centrality.get, reverse=True)[:10]

    @staticmethod
    def _fact_density(nodes: Sequence[str], relations: Sequence[Dict[str, Any]]) -> float:
        token_count = max(sum(len(tokenize(node)) for node in nodes), 1)
        return round(len(relations) / token_count, 6)
