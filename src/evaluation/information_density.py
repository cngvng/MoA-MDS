"""
Information Density Metric for KGSum Agent

Implementation of the Information Density metric used to evaluate
the quality of community detection in knowledge graphs.

Based on information-theoretic approaches to graph analysis.
"""

import math
import networkx as nx
from typing import Dict, Any, List
from collections import Counter
from loguru import logger


class InformationDensityCalculator:
    """
    Calculator for Information Density metric in knowledge graph communities.
    
    Based on:
    - Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks on complex networks. PNAS.
    - Fortunato, S., & Hric, D. (2016). Community detection in networks: A user guide. Physics Reports.
    """
    
    def __init__(self, debug_mode: bool = True):
        """
        Initialize the Information Density calculator.
        
        Args:
            debug_mode: Enable debug logging
        """
        self.debug_mode = debug_mode
        
    def calculate_community_information_density(
        self, 
        community_data: Dict[str, Any]
    ) -> float:
        """
        Calculate Information Density for a single community.
        
        ID(C_j) = (|E_j| / |V_j|^2) × H(C_j)
        
        Where:
        - |E_j| = number of edges (relations) in community j
        - |V_j| = number of nodes (entities) in community j  
        - H(C_j) = Shannon entropy of relation types in community j
        
        Args:
            community_data: Community data with nodes and edges
            
        Returns:
            Information density score for the community
        """
        nodes = community_data.get('nodes', [])
        edges = community_data.get('edges', [])
        
        if len(nodes) == 0:
            return 0.0
        
        # Calculate structural density component: |E_j| / |V_j|^2
        num_edges = len(edges)
        num_nodes = len(nodes)
        structural_density = num_edges / (num_nodes ** 2)
        
        # Calculate Shannon entropy of relation types
        relation_entropy = self._calculate_relation_entropy(edges)
        
        # Information Density = structural_density × relation_entropy
        information_density = structural_density * relation_entropy
        
        if self.debug_mode:
            logger.debug(f"Community {community_data.get('id')}: "
                        f"nodes={num_nodes}, edges={num_edges}, "
                        f"structural_density={structural_density:.4f}, "
                        f"entropy={relation_entropy:.4f}, "
                        f"info_density={information_density:.4f}")
        
        return information_density
    
    def _calculate_relation_entropy(self, edges: List[Dict[str, Any]]) -> float:
        """
        Calculate Shannon entropy of relation types in a community.
        
        H(C_j) = -Σ p(r) log₂ p(r)
        
        Where p(r) is the probability of relation type r in the community.
        
        Args:
            edges: List of edges/relations in the community
            
        Returns:
            Shannon entropy of relation types
        """
        if not edges:
            return 0.0
        
        # Count relation types
        relation_types = [edge.get('relation_type', 'unknown') for edge in edges]
        relation_counts = Counter(relation_types)
        
        total_relations = len(edges)
        entropy = 0.0
        
        for relation_type, count in relation_counts.items():
            probability = count / total_relations
            if probability > 0:  # Avoid log(0)
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def calculate_overall_information_density(
        self, 
        communities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate weighted average Information Density across all communities.
        
        Args:
            communities: List of community data
            
        Returns:
            Dictionary with overall metrics and per-community breakdown
        """
        if not communities:
            return {
                'overall_information_density': 0.0,
                'weighted_average': 0.0,
                'community_scores': [],
                'total_communities': 0,
                'total_nodes': 0,
                'total_edges': 0
            }
        
        community_scores = []
        total_nodes = 0
        total_edges = 0
        weighted_sum = 0.0
        total_weight = 0.0
        
        for community in communities:
            # Calculate Information Density for this community
            info_density = self.calculate_community_information_density(community)
            
            # Use community size as weight
            community_size = len(community.get('nodes', []))
            community_edges = len(community.get('edges', []))
            
            community_scores.append({
                'community_id': community.get('id'),
                'information_density': info_density,
                'size': community_size,
                'edges': community_edges
            })
            
            # Weighted sum for overall calculation
            weighted_sum += info_density * community_size
            total_weight += community_size
            
            total_nodes += community_size
            total_edges += community_edges
        
        # Calculate weighted average
        weighted_average = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Simple average for comparison
        simple_average = sum(score['information_density'] for score in community_scores) / len(community_scores)
        
        result = {
            'overall_information_density': weighted_average,
            'simple_average': simple_average,
            'weighted_average': weighted_average,
            'community_scores': community_scores,
            'total_communities': len(communities),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'avg_community_size': total_nodes / len(communities) if communities else 0.0,
            'avg_edges_per_community': total_edges / len(communities) if communities else 0.0
        }
        
        if self.debug_mode:
            logger.info(f"Overall Information Density: {weighted_average:.4f}")
            logger.info(f"Total communities: {len(communities)}, "
                       f"Total nodes: {total_nodes}, "
                       f"Total edges: {total_edges}")
        
        return result
    
    def compare_community_detection_methods(
        self,
        baseline_communities: List[Dict[str, Any]],
        louvain_communities: List[Dict[str, Any]], 
        leiden_communities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare Information Density across different community detection methods.
        
        Args:
            baseline_communities: Communities from baseline (no detection)
            louvain_communities: Communities from Louvain algorithm
            leiden_communities: Communities from Leiden algorithm
            
        Returns:
            Comparative analysis of Information Density scores
        """
        logger.info("Comparing community detection methods using Information Density")
        
        # Calculate Information Density for each method
        baseline_metrics = self.calculate_overall_information_density(baseline_communities)
        louvain_metrics = self.calculate_overall_information_density(louvain_communities)
        leiden_metrics = self.calculate_overall_information_density(leiden_communities)
        
        # Calculate relative improvements
        baseline_score = baseline_metrics['overall_information_density']
        louvain_score = louvain_metrics['overall_information_density'] 
        leiden_score = leiden_metrics['overall_information_density']
        
        louvain_improvement = ((louvain_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0.0
        leiden_improvement = ((leiden_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0.0
        leiden_vs_louvain = ((leiden_score - louvain_score) / louvain_score * 100) if louvain_score > 0 else 0.0
        
        comparison = {
            'methods': {
                'baseline': {
                    'information_density': baseline_score,
                    'communities': baseline_metrics['total_communities'],
                    'avg_size': baseline_metrics['avg_community_size']
                },
                'louvain': {
                    'information_density': louvain_score,
                    'communities': louvain_metrics['total_communities'],
                    'avg_size': louvain_metrics['avg_community_size']
                },
                'leiden': {
                    'information_density': leiden_score,
                    'communities': leiden_metrics['total_communities'],
                    'avg_size': leiden_metrics['avg_community_size']
                }
            },
            'relative_improvements': {
                'louvain_vs_baseline': louvain_improvement,
                'leiden_vs_baseline': leiden_improvement,
                'leiden_vs_louvain': leiden_vs_louvain
            },
            'detailed_metrics': {
                'baseline': baseline_metrics,
                'louvain': louvain_metrics,
                'leiden': leiden_metrics
            }
        }
        
        # Log results
        logger.info("=== Information Density Comparison Results ===")
        logger.info(f"Baseline: {baseline_score:.4f}")
        logger.info(f"Louvain:  {louvain_score:.4f} (+{louvain_improvement:.1f}%)")
        logger.info(f"Leiden:   {leiden_score:.4f} (+{leiden_improvement:.1f}% vs baseline, +{leiden_vs_louvain:.1f}% vs Louvain)")
        
        return comparison


def evaluate_kgsum_information_density(kgsum_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate Information Density for KGSum Agent results.
    
    Args:
        kgsum_result: Result from KGSum Agent processing
        
    Returns:
        Information density analysis
    """
    calculator = InformationDensityCalculator()
    
    # Extract communities from KGSum result
    detailed_result = kgsum_result.get('detailed_result', {})
    if isinstance(detailed_result, dict):
        communities = detailed_result.get('communities', [])
    else:
        communities = []
    
    # Calculate Information Density
    id_metrics = calculator.calculate_overall_information_density(communities)
    
    # Add to result
    evaluation = {
        'information_density_metrics': id_metrics,
        'summary_quality': {
            'confidence_score': kgsum_result.get('confidence_score', 0.0),
            'processing_time': kgsum_result.get('processing_time', 0.0),
            'summary_length': len(kgsum_result.get('summary', '').split())
        },
        'correlation_analysis': {
            'density_confidence_correlation': id_metrics['overall_information_density'] * kgsum_result.get('confidence_score', 0.0)
        }
    }
    
    return evaluation
