from typing import Dict, Any
from rouge_score import rouge_scorer
from .base import BaseMetric

class RougeMetric(BaseMetric):
    """Class for calculating ROUGE scores"""
    
    def __init__(self, use_stemmer: bool = False):
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=use_stemmer,
        )

    def calculate(self, generated_text: str, reference_text: str) -> Dict[str, Any]:
        """Calculate ROUGE scores
        
        Args:
            generated_text (str): Generated summary text
            reference_text (str): Reference summary text
            
        Returns:
            Dict[str, Any]: Dictionary containing ROUGE-1, ROUGE-2 and ROUGE-L scores
                           with precision, recall and f1 for each
        """
        scores = self.scorer.score(reference_text, generated_text)
        
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'f1': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'f1': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'f1': scores['rougeL'].fmeasure
            }
        }
