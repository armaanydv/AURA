"""
Base Recommender Interface for AURA
All recommendation models implement this contract.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseRecommender(ABC):
    def __init__(self, name: str = "BaseRecommender"):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseRecommender":
        """Train or index the recommendation model."""
        pass

    @abstractmethod
    def recommend(self, user_id: str, top_k: int = 10, exclude_known: bool = True) -> List[Dict[str, Any]]:
        """
        Generate top-K item recommendations for a given user.
        Returns a list of dicts: [{"product_id": str, "score": float, "reason": str}]
        """
        pass

    @abstractmethod
    def get_similar_items(self, product_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find top-K most similar items to a target item.
        Returns a list of dicts: [{"product_id": str, "score": float}]
        """
        pass
