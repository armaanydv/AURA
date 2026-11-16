"""
Collaborative Filtering Recommendation Engine
Implements Item-Based and User-Based Collaborative Filtering using Cosine and Pearson correlations.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.recommenders.base import BaseRecommender

class CollaborativeRecommender(BaseRecommender):
    def __init__(self, mode: str = "item"):
        """
        mode: 'item' for Item-Based Collaborative Filtering, 'user' for User-Based Collaborative Filtering.
        """
        super().__init__(name=f"CollaborativeFiltering_{mode.capitalize()}")
        self.mode = mode
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.user_ids: List[str] = []
        self.item_ids: List[str] = []
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.idx_to_user: Dict[int, str] = {}

    def fit(self, interactions_df: pd.DataFrame, value_col: str = "rating") -> "CollaborativeRecommender":
        """
        Construct user-item matrix and compute pairwise similarity matrix.
        """
        pivot = interactions_df.pivot_table(
            index="user_id",
            columns="product_id",
            values=value_col,
            fill_value=0.0
        )
        self.user_item_matrix = pivot
        self.user_ids = list(pivot.index)
        self.item_ids = list(pivot.columns)
        self.user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        self.idx_to_item = {i: iid for i, iid in enumerate(self.item_ids)}
        self.idx_to_user = {i: uid for i, uid in enumerate(self.user_ids)}

        R = pivot.values.astype(float)

        if self.mode == "item":
            # Item-Item similarity (Columns)
            # Subtract item means for Pearson correlation approximation
            item_sums = np.sum(R, axis=0)
            item_counts = np.sum(R > 0, axis=0)
            item_counts[item_counts == 0] = 1
            item_means = item_sums / item_counts
            R_centered = np.where(R > 0, R - item_means[np.newaxis, :], 0.0)
            self.similarity_matrix = cosine_similarity(R_centered.T)
        else:
            # User-User similarity (Rows)
            user_sums = np.sum(R, axis=1)
            user_counts = np.sum(R > 0, axis=1)
            user_counts[user_counts == 0] = 1
            user_means = user_sums / user_counts
            R_centered = np.where(R > 0, R - user_means[:, np.newaxis], 0.0)
            self.similarity_matrix = cosine_similarity(R_centered)

        self.is_fitted = True
        return self

    def get_similar_items(self, product_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Item-Item similarity scores based on co-occurrence and rating patterns."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if product_id not in self.item_to_idx or self.mode != "item":
            return []

        idx = self.item_to_idx[product_id]
        sims = self.similarity_matrix[idx]
        sorted_indices = np.argsort(sims)[::-1]

        results = []
        for s_idx in sorted_indices:
            if s_idx == idx:
                continue
            iid = self.idx_to_item[s_idx]
            score = float(sims[s_idx])
            results.append({
                "product_id": iid,
                "score": round(score, 4),
                "reason": f"Frequently co-rated item ({int(score * 100)}% correlation)"
            })
            if len(results) >= top_k:
                break
        return results

    def recommend(self, user_id: str, top_k: int = 10, exclude_known: bool = True) -> List[Dict[str, Any]]:
        """Recommend items using collaborative neighborhood aggregation."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if user_id not in self.user_to_idx:
            # Cold start
            return []

        u_idx = self.user_to_idx[user_id]
        user_ratings = self.user_item_matrix.values[u_idx]

        if self.mode == "item":
            # Item-based prediction: score = (sim_matrix @ user_ratings) / sum(sim_matrix)
            sim_weights = np.dot(self.similarity_matrix, user_ratings)
            denom = np.dot(np.abs(self.similarity_matrix), (user_ratings > 0).astype(float))
            denom[denom == 0] = 1.0
            scores = sim_weights / denom
        else:
            # User-based prediction: score = (sim_matrix[u] @ R) / sum(sims)
            user_sims = self.similarity_matrix[u_idx]
            scores = np.dot(user_sims, self.user_item_matrix.values)
            denom = np.sum(np.abs(user_sims))
            denom = denom if denom != 0 else 1.0
            scores = scores / denom

        if exclude_known:
            scores = np.where(user_ratings > 0, -np.inf, scores)

        sorted_indices = np.argsort(scores)[::-1]
        results = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if score == -np.inf or score <= 0:
                continue
            iid = self.idx_to_item[idx]
            results.append({
                "product_id": iid,
                "score": round(score, 4),
                "reason": f"Collaborative filtering {self.mode}-neighborhood affinity"
            })
            if len(results) >= top_k:
                break
        return results
