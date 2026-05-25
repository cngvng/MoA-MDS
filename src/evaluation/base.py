from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMetric(ABC):
    """Base class for evaluation metrics"""
    
    @abstractmethod
    def calculate(self, generated_text: str, reference_text: str) -> Dict[str, Any]:
        """Calculate metric scores
        
        Args:
            generated_text (str): Generated summary text
            reference_text (str): Reference summary text
            
        Returns:
            Dict[str, Any]: Dictionary containing metric scores
        """
        pass
