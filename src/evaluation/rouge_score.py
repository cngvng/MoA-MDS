import re
from typing import Dict, Any, Optional, Sequence

from rouge_score import rouge_scorer

from .base import BaseMetric


class RegexTokenizer:
    def __init__(self, pattern: str) -> None:
        self.pattern = re.compile(pattern, flags=re.UNICODE)

    def tokenize(self, text: str):
        return self.pattern.findall((text or "").lower())


class CharacterTokenizer:
    def tokenize(self, text: str):
        normalized = re.sub(r"\s+", "", (text or "").lower())
        return list(normalized)


class RougeMetric(BaseMetric):
    """Class for calculating ROUGE scores"""
    
    def __init__(
        self,
        use_stemmer: bool = False,
        tokenizer_name: str = "default",
        language: Optional[str] = None,
    ):
        tokenizer = None
        if tokenizer_name == "unicode_char" or language == "vi":
            tokenizer = CharacterTokenizer()
        elif tokenizer_name == "vietnamese_aware":
            tokenizer = RegexTokenizer(r"[\wÀ-ỹ]+")

        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=use_stemmer,
            tokenizer=tokenizer,
        )
        self.tokenizer_name = tokenizer_name
        self.language = language

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

    def calculate_multi_reference(
        self,
        generated_text: str,
        reference_texts: Sequence[str],
        aggregation: str = "max",
    ) -> Dict[str, Any]:
        references = [reference for reference in reference_texts if reference]
        if not references:
            return {}
        if len(references) == 1:
            result = self.calculate(generated_text, references[0])
            result["reference_count"] = 1
            result["reference_aggregation"] = "single"
            result["selected_reference_index"] = 0
            return result

        scores = [self.calculate(generated_text, reference) for reference in references]
        if aggregation == "average":
            result = {
                metric_name: {
                    score_name: sum(score[metric_name][score_name] for score in scores) / len(scores)
                    for score_name in ["precision", "recall", "f1"]
                }
                for metric_name in ["rouge1", "rouge2", "rougeL"]
            }
            selected_index = None
        else:
            selected_index, result = max(
                enumerate(scores),
                key=lambda item: (
                    item[1]["rouge1"]["f1"],
                    item[1]["rouge2"]["f1"],
                    item[1]["rougeL"]["f1"],
                ),
            )

        result["reference_count"] = len(references)
        result["reference_aggregation"] = aggregation
        result["selected_reference_index"] = selected_index
        return result
